"""Capture healthy-only splits and three additional damaged-gear test sets."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from rotation import object_geometry, setup
from three_dents import trace_frame
from low_light import DisplayMapping, block_sum
import mitsuba as mi
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossing_angle(defect, geometry, teeth):
    point = np.asarray(defect["surface_point_mm"])*0.001
    origin = np.asarray(geometry["laser_origin_m"])
    fan = np.cross(np.asarray(geometry["line_tangent"]), origin-np.asarray(geometry["patch_m"]))
    fan /= np.linalg.norm(fan)
    a = fan[0]*point[0]+fan[1]*point[1]
    b = -fan[0]*point[1]+fan[1]*point[0]
    c = fan[2]*point[2]-fan@origin
    phase = math.atan2(b, a)
    ratio = -c/math.hypot(a, b)
    if abs(ratio) > 1:
        raise ValueError(f"The laser fan plane misses {defect['id']} through rotation.")
    roots = np.degrees([phase+math.acos(ratio), phase-math.acos(ratio)]) % 360
    nominal = (-defect["tooth"]*360/teeth) % 360
    distance = abs((roots-nominal+180) % 360-180)
    return float(roots[distance.argmin()])


def labels_at(angle, gear, offsets):
    half_width = max(abs(value) for value in offsets)
    return [d["id"] for d in gear["defects"]
            if abs((angle-d["crossing_angle_deg"]+180) % 360-180) <= half_width+1e-7]


def preprocess(out, config, manifest):
    started = time.perf_counter()
    rows = manifest["frames"]
    settings = config["display"]
    trains = [record for record in rows if record["case"] == "healthy_train"]
    train_images = [block_sum(np.asarray(mi.Bitmap(str(out/record["raw"]))).squeeze().astype(np.float64),
                              settings["block_size"]) for record in trains]
    mapping = DisplayMapping(train_images, settings)
    np.savez_compressed(out/"histogram_mapping.npz", intensity_edges=mapping.edges,
                        cumulative_probability=mapping.cdf, healthy_histogram_counts=mapping.counts,
                        peak=mapping.peak, floor=mapping.floor)
    binned = {}
    for record in rows:
        case = record["case"]
        image = block_sum(np.asarray(mi.Bitmap(str(out/record["raw"]))).squeeze().astype(np.float64),
                          settings["block_size"])
        binned.setdefault(case, []).append(image.astype(np.float32))
        equalized = np.rint(255*mapping.transform(image, "histogram")).astype(np.uint8)
        destination = out/"frames"/case/f"frame_{record['index']:03d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(equalized).save(destination)
        record["image"] = str(destination.relative_to(out))
    np.savez_compressed(out/"screen_arrays.npz", angle_deg=np.asarray(manifest["angles_deg"]),
                        **{case: np.stack(images) for case, images in binned.items()})
    fields = ("case", "index", "angle_deg", "raw", "image", "defect_ids", "first_seed", "raw_source")
    with (out/"dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ";".join(row[key]) if key == "defect_ids" else row[key] for key in fields})
    manifest["histogram"] = {"training_case": "healthy_train", "peak": mapping.peak, "floor": mapping.floor,
                              "image_shape": list(train_images[0].shape), "mode": "L", "precision_bits": 8,
                              "mapping": "One healthy-training CDF, reused for every split"}
    manifest["preprocessing_seconds"] = time.perf_counter()-started
    manifest["status"] = "captures_complete"
    write_json(out/"manifest.json", manifest)


def main(config_path):
    config = json.loads(config_path.read_text())
    out = PROJECT/config["output"]
    out.mkdir(parents=True, exist_ok=True)
    config_file, manifest_file = out/"config.json", out/"manifest.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        old_config = json.loads(config_file.read_text())
        old_manifest = json.loads(manifest_file.read_text())
        same_optics = {k: v for k, v in old_config.items() if k != "gears"} == {k: v for k, v in config.items() if k != "gears"}
        if not same_optics or any(row["case"].startswith("gear_") for row in old_manifest["frames"]):
            raise ValueError("Changed test geometry must precede damaged captures and preserve the optical settings.")
        write_json(out/"initial_config.json", old_config)
    write_json(config_file, config)
    reference = PROJECT/config["healthy_reference"]
    reference_manifest = json.loads((reference/"manifest.json").read_text())
    trace = json.loads((PROJECT/config["trace_config"]).read_text())
    mi.set_variant(trace["variant"])
    model_file = PROJECT/config["model_output"]/"models.json"
    models = json.loads(model_file.read_text())
    geometry = reference_manifest["geometry"]
    gears = copy.deepcopy(models["gears"])
    extra = []
    for gear in gears:
        for defect in gear["defects"]:
            angle = crossing_angle(defect, geometry, reference_manifest["gear_tooth_count"])
            defect["crossing_angle_deg"] = angle
            defect["sample_angles_deg"] = [(angle+delta) % 360 for delta in config["defect_sample_offsets_deg"]]
            extra.extend(defect["sample_angles_deg"])
    angles = np.unique(np.round(np.r_[np.arange(0, 360, config["rotation_step_deg"]), extra], 8))
    write_json(out/"defects.json", {"gears": gears, "passage_half_width_deg": max(abs(x) for x in config["defect_sample_offsets_deg"]),
                                   "definition": "A defect is detected when a sampled passage capture exceeds the fixed healthy-calibrated threshold"})
    cases = ["healthy_train", "healthy_calibration", "healthy_test"]+[g["id"] for g in gears]
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["status"] == "captures_complete":
            print("Existing captures and preprocessing are complete.", flush=True)
            return
        saved_angles = np.asarray(manifest["angles_deg"])
        additions = [angle for angle in angles if not np.any(abs((saved_angles-angle+180) % 360-180) < 1e-7)]
        angles = np.r_[saved_angles, additions]
        manifest["angles_deg"] = angles.tolist()
        manifest.setdefault("source_revisions", []).append({"utc": datetime.now(timezone.utc).isoformat(),
            "completed_frames": len(manifest["frames"]), "added_angles_deg": additions,
            "source_sha256_previous": manifest["source_sha256"],
            "reason": "Corrected test-gear edge geometry before damaged captures; existing optical settings and healthy indices preserved"})
        manifest["source_sha256"] = {key: digest(PROJECT/key) for key in manifest["source_sha256"]}
        write_json(manifest_file, manifest)
    else:
        sources = [Path(__file__), config_path, PROJECT/config["trace_config"], model_file,
                   PROJECT/"collector/run.py", PROJECT/"collector/rotation.py", PROJECT/"collector/three_dents.py",
                   PROJECT/"collector/low_light.py", reference/"manifest.json"]
        manifest = {"status": "rendering", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "geometry": geometry, "trace_config": trace, "angles_deg": angles.tolist(), "cases": cases,
                    "source_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in sources},
                    "frames": [], "evaluation": "15 individual defects; false positives on held-out healthy_test frames"}
        write_json(manifest_file, manifest)
    done = {(row["case"], row["index"]) for row in manifest["frames"]}
    reference_angles = np.asarray([f["angle_deg"] for f in reference_manifest["frames"]])
    setup_config = copy.deepcopy(trace)
    setup_config["mesh"] = str(PROJECT/config["healthy_mesh"])
    setup_config["alignment_mesh"] = setup_config["mesh"]
    healthy_scene, healthy_bsdf, _, _ = setup(setup_config)
    started = time.perf_counter()
    total = len(cases)*len(angles)
    for case in cases:
        gear = next((g for g in gears if g["id"] == case), None)
        if gear:
            setup_config["mesh"] = str(PROJECT/gear["mesh"])
            scene, bsdf, _, _ = setup(setup_config)
            manifest.setdefault("test_mesh_sha256", {})[case] = digest(PROJECT/gear["mesh"])
        else:
            scene, bsdf = healthy_scene, healthy_bsdf
        for index, value in enumerate(angles):
            if (case, index) in done:
                continue
            angle, frame_started = float(value), time.perf_counter()
            path = out/"raw"/case/f"frame_{index:03d}.exr"
            path.parent.mkdir(parents=True, exist_ok=True)
            match = np.flatnonzero(abs(reference_angles-angle) < 1e-7)
            if case == "healthy_train" and len(match):
                reference_index = int(match[0])
                source = reference/"raw"/f"frame_{reference_index:03d}.exr"
                if not path.exists():
                    path.symlink_to(os.path.relpath(source, path.parent))
                power = reference_manifest["frames"][reference_index]["power"]
                first_seed = trace["sampling"]["seed"]+reference_index*trace["sampling"]["batches"]
                raw_source = "saved_healthy_reference"
            else:
                first_seed = (trace["sampling"]["seed"]+config["seed_offsets"][case]
                              +(index+len(reference_angles))*trace["sampling"]["batches"])
                image, _, record = trace_frame(scene, bsdf, trace, geometry, angle, first_seed)
                mi.Bitmap(image.astype(np.float32)).write(str(path))
                power, raw_source = record["power"], "new_independent_capture"
            manifest["frames"].append({"case": case, "index": index, "angle_deg": angle,
                                       "raw": str(path.relative_to(out)), "first_seed": first_seed,
                                       "raw_source": raw_source, "defect_ids": labels_at(angle, gear, config["defect_sample_offsets_deg"]) if gear else [],
                                       "power": power, "elapsed_seconds": time.perf_counter()-frame_started})
            if len(manifest["frames"]) % 10 == 0 or index == len(angles)-1:
                manifest["elapsed_render_seconds_this_session"] = time.perf_counter()-started
                write_json(manifest_file, manifest)
                print(f"{len(manifest['frames'])}/{total} | {case} | {angle:.4f}° | {time.perf_counter()-started:.1f} s", flush=True)
    manifest["frames"].sort(key=lambda row: (cases.index(row["case"]), row["index"]))
    manifest["status"] = "rendered"
    write_json(manifest_file, manifest)
    preprocess(out, config, manifest)
    print("Healthy-only training/calibration, healthy test, and three damaged test sets are ready.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("prototype.json"))
    main(parser.parse_args().config.resolve())
