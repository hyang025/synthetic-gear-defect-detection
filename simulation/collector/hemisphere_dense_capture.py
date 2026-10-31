"""Add 0.2-degree healthy training captures while preserving evaluation splits."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from prototype_capture import PROJECT, digest, write_json
from rotation import setup
from three_dents import trace_frame
import mitsuba as mi
import numpy as np


def main(config_path):
    config = json.loads(config_path.read_text())
    source, out = PROJECT/config["source_run"], PROJECT/config["output"]
    out.mkdir(parents=True, exist_ok=True)
    with (source/"dataset.csv").open() as stream:
        original = [row for row in csv.DictReader(stream) if row["case"] == "healthy_train"]
    angles = np.round(np.arange(0, 360, config["rotation_step_deg"]), 8)
    present = {round(float(row["angle_deg"]), 8) for row in original}
    additions = [float(angle) for angle in angles if float(angle) not in present]
    source_manifest = json.loads((source/"manifest.json").read_text())
    source_config = json.loads((source/"config.json").read_text())
    trace = copy.deepcopy(source_manifest["trace_config"])
    geometry = source_manifest["geometry"]
    sources = [config_path, Path(__file__), source/"dataset.csv", source/"manifest.json",
               PROJECT/"collector/rotation.py", PROJECT/"collector/run.py", PROJECT/"collector/three_dents.py",
               PROJECT/"healthy/scene.py", PROJECT/"gear/machining_marks.py", PROJECT/source_config["healthy_mesh"]]
    hashes = {str(path.relative_to(PROJECT)): digest(path) for path in sources}
    manifest_path = out/"capture_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != hashes or json.loads((out/"config.json").read_text()) != config:
            raise RuntimeError("Resume with the saved capture configuration and source files.")
        if manifest["status"] == "complete":
            print("Dense healthy training captures are complete.", flush=True)
            return
    else:
        frames = []
        for row in original:
            frames.append({"case": "healthy_train", "role": "healthy_train", "index": int(row["index"]),
                "angle_deg": float(row["angle_deg"]), "raw": os.path.relpath(source/row["raw"], out),
                "defect_ids": "", "first_seed": int(row["first_seed"]),
                "raw_source": "preserved_healthy_training_capture"})
        manifest = {"status": "rendering", "created_utc": datetime.now(timezone.utc).isoformat(),
            "rotation_step_deg": config["rotation_step_deg"], "preserved_training_frames": len(original),
            "new_training_frames": len(additions), "total_training_frames": len(original)+len(additions),
            "new_angles_deg": additions, "geometry": geometry, "trace_config": trace,
            "rays_per_frame": trace["sampling"]["batches"]*trace["sampling"]["rays_per_batch"],
            "source_sha256": hashes, "frames": frames, "elapsed_render_seconds": 0.0}
        write_json(out/"config.json", config)
        write_json(manifest_path, manifest)
    mi.set_variant(trace["variant"])
    setup_config = copy.deepcopy(trace)
    setup_config["mesh"] = str(PROJECT/source_config["healthy_mesh"])
    setup_config["alignment_mesh"] = setup_config["mesh"]
    scene, bsdf, _, _ = setup(setup_config)
    raw_folder = out/"raw/healthy_train"
    raw_folder.mkdir(parents=True, exist_ok=True)
    done = {row["index"] for row in manifest["frames"]}
    started, previous_seconds = time.perf_counter(), manifest["elapsed_render_seconds"]
    for offset, angle in enumerate(additions):
        index = len(original)+offset
        if index in done:
            continue
        frame_started = time.perf_counter()
        seed = trace["sampling"]["seed"]+config["seed_offset"]+offset*trace["sampling"]["batches"]
        image, _, record = trace_frame(scene, bsdf, trace, geometry, angle, seed)
        raw = raw_folder/f"frame_{index:04d}.exr"
        mi.Bitmap(image.astype(np.float32)).write(str(raw))
        manifest["frames"].append({"case": "healthy_train", "role": "healthy_train", "index": index,
            "angle_deg": angle, "raw": str(raw.relative_to(out)), "defect_ids": "", "first_seed": seed,
            "raw_source": "new_dense_healthy_training_capture", "raw_sha256": digest(raw),
            "power": record["power"], "elapsed_seconds": time.perf_counter()-frame_started})
        if (offset+1) % 10 == 0 or offset == len(additions)-1:
            manifest["elapsed_render_seconds"] = previous_seconds+time.perf_counter()-started
            write_json(manifest_path, manifest)
            print(f"Dense healthy: {len(manifest['frames'])-len(original)}/{len(additions)} new · "
                  f"{angle:.2f}° · {manifest['elapsed_render_seconds']:.1f} s", flush=True)
    fields = ("case", "role", "index", "angle_deg", "raw", "defect_ids", "first_seed", "raw_source")
    manifest["frames"].sort(key=lambda row: row["index"])
    with (out/"training_dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in manifest["frames"])
    manifest.update(status="complete", elapsed_render_seconds=previous_seconds+time.perf_counter()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path, manifest)
    print(f"Ready: {len(manifest['frames'])} healthy training captures.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("hemisphere_dense.json"))
    main(parser.parse_args().config.resolve())
