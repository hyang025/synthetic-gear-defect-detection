"""Capture the separate damaged validation gear with the baseline optics."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from prototype_capture import PROJECT, crossing_angle, digest, labels_at, write_json
from rotation import setup
from three_dents import trace_frame
from low_light import block_sum
import mitsuba as mi
import numpy as np
from PIL import Image


def main(config_path):
    config = json.loads(config_path.read_text())
    source, out = PROJECT/config["source_run"], PROJECT/config["output"]
    out.mkdir(parents=True, exist_ok=True)
    capture_path = out/"capture_manifest.json"
    if capture_path.exists():
        manifest = json.loads(capture_path.read_text())
        if manifest["status"] == "complete":
            print("Saved validation captures are complete.", flush=True)
            return
        if json.loads((out/"config.json").read_text()) != config:
            raise ValueError("Preserve the capture configuration while resuming.")
    else:
        manifest = None
    model_file = PROJECT/config["model_output"]/"models.json"
    models = json.loads(model_file.read_text())
    gear = copy.deepcopy(models["gears"][0])
    reference = json.loads((source/"manifest.json").read_text())
    trace = reference["trace_config"]
    geometry = reference["geometry"]
    tooth_count = json.loads((PROJECT/"gear/model_parameters.json").read_text())["supplied"]["tooth_count"]
    extra = []
    for defect in gear["defects"]:
        center = crossing_angle(defect, geometry, tooth_count)
        defect["crossing_angle_deg"] = center
        defect["sample_angles_deg"] = [(center+delta) % 360 for delta in config["defect_sample_offsets_deg"]]
        extra.extend(defect["sample_angles_deg"])
    angles = np.unique(np.round(np.r_[np.arange(0, 360, config["rotation_step_deg"]), extra], 8))
    write_json(out/"config.json", config)
    write_json(out/"defects.json", {"gears": [gear], "passage_half_width_deg": max(abs(x) for x in config["defect_sample_offsets_deg"])})
    if manifest is None:
        sources = [config_path, Path(__file__), PROJECT/"collector/prototype_capture.py", PROJECT/"collector/rotation.py",
                   PROJECT/"collector/run.py", PROJECT/"collector/three_dents.py", model_file, PROJECT/gear["mesh"],
                   source/"manifest.json", source/"histogram_mapping.npz"]
        manifest = {"status": "rendering", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "geometry": geometry, "trace_config": trace, "angles_deg": angles.tolist(), "frames": [],
                    "purpose": "Damaged validation captures for original-bank threshold selection",
                    "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}}
        write_json(capture_path, manifest)
    elif not np.array_equal(angles, np.asarray(manifest["angles_deg"])):
        raise ValueError("Preserve the encoder grid while resuming.")
    mi.set_variant(trace["variant"])
    setup_config = copy.deepcopy(trace)
    setup_config["mesh"] = str(PROJECT/gear["mesh"])
    setup_config["alignment_mesh"] = str(PROJECT/"gear/model/damage_test/healthy_control.obj")
    scene, bsdf, _, _ = setup(setup_config)
    with np.load(source/"histogram_mapping.npz") as mapping:
        edges = mapping["intensity_edges"].copy()
        cdf = mapping["cumulative_probability"].copy()
        floor = float(mapping["floor"])
    block_size = json.loads((source/"config.json").read_text())["display"]["block_size"]
    for folder in ("raw/damaged_validation", "frames/damaged_validation", "images"):
        (out/folder).mkdir(parents=True, exist_ok=True)
    done = {row["index"] for row in manifest["frames"]}
    started = time.perf_counter()
    for index, value in enumerate(angles):
        if index in done:
            continue
        frame_started = time.perf_counter()
        seed = trace["sampling"]["seed"]+config["validation_seed_offset"]+(index+len(reference["angles_deg"]))*trace["sampling"]["batches"]
        image, _, record = trace_frame(scene, bsdf, trace, geometry, float(value), seed)
        raw = out/"raw/damaged_validation"/f"frame_{index:03d}.exr"
        mi.Bitmap(image.astype(np.float32)).write(str(raw))
        # Use the same saved EXR precision and healthy-trained CDF as the baseline.
        binned = block_sum(np.asarray(mi.Bitmap(str(raw))).squeeze().astype(np.float64), block_size)
        equalized = np.interp(np.log10(np.maximum(binned, floor)), np.log10(edges), cdf, left=0, right=1)
        display = out/"frames/damaged_validation"/f"frame_{index:03d}.png"
        Image.fromarray(np.rint(255*equalized).astype(np.uint8)).save(display)
        manifest["frames"].append({"case": "damaged_validation", "index": index, "angle_deg": float(value),
            "raw": str(raw.relative_to(out)), "image": str(display.relative_to(out)),
            "defect_ids": ";".join(labels_at(float(value), gear, config["defect_sample_offsets_deg"])),
            "first_seed": seed, "raw_source": "new_independent_validation_capture",
            "power": record["power"], "elapsed_seconds": time.perf_counter()-frame_started})
        if (index+1) % 10 == 0 or index == len(angles)-1:
            manifest["elapsed_render_seconds_this_session"] = time.perf_counter()-started
            write_json(capture_path, manifest)
            print(f"Validation capture {index+1}/{len(angles)} · {value:.4f}° · {time.perf_counter()-started:.1f} s", flush=True)
    manifest["frames"].sort(key=lambda row: row["index"])
    fields = ("case", "index", "angle_deg", "raw", "image", "defect_ids", "first_seed", "raw_source")
    with (out/"validation_dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in manifest["frames"]:
            writer.writerow({key: row[key] for key in fields})
    manifest.update(status="complete", captures=len(manifest["frames"]), histogram_source=str((source/"histogram_mapping.npz").relative_to(PROJECT)))
    write_json(capture_path, manifest)
    print("Separate validation gear captures and fixed-CDF grayscale inputs complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("validation_patchcore.json"))
    args = parser.parse_args()
    main(args.config.resolve())
