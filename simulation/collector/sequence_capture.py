"""Capture encoder-spaced sequences: defect windows on damaged gears and two full healthy revolutions."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from prototype_capture import PROJECT, crossing_angle, digest, write_json
from rotation import setup
from three_dents import trace_frame
import mitsuba as mi
import numpy as np


def window_indices(center, step, half_width, count):
    """Grid indices whose angle lies within half_width of center, measured around the circle."""
    angles = np.arange(count)*step
    distance = np.abs((angles-center+180) % 360-180)
    return np.flatnonzero(distance <= half_width+1e-9)


def plan_cases(config, geometry, tooth_count):
    count = config["frames_per_revolution"]
    step = 360/count
    cases = []
    for healthy in config["healthy_cases"]:
        cases.append({"case": healthy["case"], "role": healthy["role"], "mesh": config["healthy_mesh"],
                      "seed_offset": healthy["seed_offset"], "indices": list(range(count)), "defects": []})
    offset = config["damaged_seed_offset_start"]
    damaged = []
    for group in config["damaged_sets"]:
        models = json.loads((PROJECT/group["model_output"]/"models.json").read_text())
        for gear in models["gears"]:
            if gear["id"] not in group["roles"]:
                continue
            defects, indices = [], set()
            for defect in gear["defects"]:
                center = crossing_angle(defect, geometry, tooth_count)
                window = window_indices(center, step, config["window_half_width_deg"], count)
                indices.update(window.tolist())
                defects.append({"id": defect["id"], "tooth": defect["tooth"], "kind": defect["kind"],
                                "diameter_mm": defect["diameter_mm"], "depth_mm": defect["depth_mm"],
                                "crossing_angle_deg": center, "window_indices": window.tolist()})
            damaged.append({"case": f"{group['prefix']}_{gear['id']}", "role": group["roles"][gear["id"]],
                            "mesh": gear["mesh"], "seed_offset": offset, "indices": sorted(indices),
                            "defects": defects})
            offset += 1000000
    # Defect windows render first so classifier development can start before the healthy revolutions finish.
    return damaged+cases


def main(config_path):
    config = json.loads(config_path.read_text())
    source, out = PROJECT/config["source_run"], PROJECT/config["output"]
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((source/"manifest.json").read_text())
    trace, geometry = reference["trace_config"], reference["geometry"]
    tooth_count = json.loads((PROJECT/"gear/model_parameters.json").read_text())["supplied"]["tooth_count"]
    step = 360/config["frames_per_revolution"]
    cases = plan_cases(config, geometry, tooth_count)
    capture_path = out/"capture_manifest.json"
    manifest = json.loads(capture_path.read_text()) if capture_path.exists() else None
    if manifest and json.loads((out/"config.json").read_text()) != config:
        raise ValueError("Preserve the capture configuration while resuming.")
    if manifest and manifest["status"] == "complete":
        print("Saved sequence captures are complete.", flush=True)
        return
    write_json(out/"config.json", config)
    write_json(out/"cases.json", {"step_deg": step, "cases": cases})
    if manifest is None:
        sources = [config_path, Path(__file__), PROJECT/"collector/prototype_capture.py", PROJECT/"collector/rotation.py",
                   PROJECT/"collector/run.py", PROJECT/"collector/three_dents.py", source/"manifest.json"]
        sources += [PROJECT/case["mesh"] for case in cases]
        manifest = {"status": "rendering", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "step_deg": step, "geometry": geometry, "trace_config": trace, "frames": [],
                    "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in dict.fromkeys(sources)}}
        write_json(capture_path, manifest)
    mi.set_variant(trace["variant"])
    done = {(row["case"], row["index"]) for row in manifest["frames"]}
    total = sum(len(case["indices"]) for case in cases)
    started = time.perf_counter()
    for case in cases:
        name = case["case"]
        todo = [index for index in case["indices"] if (name, index) not in done]
        if not todo:
            continue
        profile_path = out/"profiles"/f"{name}.npz"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        saved = dict(np.load(profile_path)) if profile_path.exists() else {}
        rows = {int(i): (u, v) for i, u, v in zip(saved.get("index", []), saved.get("u", []), saved.get("v", []))}
        setup_config = copy.deepcopy(trace)
        setup_config["mesh"] = str(PROJECT/case["mesh"])
        setup_config["alignment_mesh"] = str(PROJECT/config["healthy_mesh"])
        scene, bsdf, _, _ = setup(setup_config)
        (out/"raw"/name).mkdir(parents=True, exist_ok=True)
        for count, index in enumerate(todo):
            frame_started = time.perf_counter()
            angle = index*step
            seed = trace["sampling"]["seed"]+case["seed_offset"]+index*trace["sampling"]["batches"]
            image, profiles, record = trace_frame(scene, bsdf, trace, geometry, angle, seed)
            raw = out/"raw"/name/f"frame_{index:04d}.exr"
            mi.Bitmap(image.astype(np.float32)).write(str(raw))
            rows[index] = (np.mean(profiles["u"], axis=0), np.mean(profiles["v"], axis=0))
            manifest["frames"].append({"case": name, "role": case["role"], "index": index, "angle_deg": angle,
                                       "raw": str(raw.relative_to(out)), "first_seed": seed, "power": record["power"],
                                       "elapsed_seconds": time.perf_counter()-frame_started})
            if (count+1) % 20 == 0 or count == len(todo)-1:
                order = sorted(rows)
                np.savez_compressed(profile_path, index=np.asarray(order), angle_deg=np.asarray(order)*step,
                                    u=np.stack([rows[i][0] for i in order]), v=np.stack([rows[i][1] for i in order]))
                manifest["elapsed_render_seconds_this_session"] = time.perf_counter()-started
                write_json(capture_path, manifest)
                print(f"Sequence {len(manifest['frames'])}/{total} · {name} · {angle:.4f}° · "
                      f"{time.perf_counter()-started:.1f} s", flush=True)
    manifest.update(status="complete", captures=len(manifest["frames"]),
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(capture_path, manifest)
    print(f"Completed {len(cases)} sequence cases.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("sequence.json"))
    args = parser.parse_args()
    main(args.config.resolve())
