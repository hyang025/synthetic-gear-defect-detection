"""Prepare shared-scale grayscale inputs for one spherical-defect test set."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from prototype_capture import PROJECT, digest, write_json
from low_light import block_sum
import mitsuba as mi
import numpy as np
from PIL import Image


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    source, spherical, out = (PROJECT/config[key] for key in ("source_run", "spherical_run", "output"))
    transform = config.get("transform", config.get("linear"))
    mode = transform.get("mode", "linear")
    if mode not in ("linear", "gamma", "asinh", "histogram") or transform["precision_bits"] != 8:
        raise ValueError("Use a configured linear, gamma, asinh, or histogram mapping with 8-bit inputs.")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"preprocessing_manifest.json").exists():
        raise FileExistsError("Preserve the completed display dataset before a new run.")
    capture_path = spherical/"capture_manifest.json"
    cases = (config["validation_gear"], config["test_gear"])
    while True:
        try:
            manifest = json.loads(capture_path.read_text())
            ready = all(sum(row["case"] == case for row in manifest["frames"]) == len(manifest["angles_deg"][case]) for case in cases)
            if ready:
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if not wait:
            raise RuntimeError("Complete the validation and first test-gear captures first.")
        time.sleep(5)
    write_json(out/"config.json", config)
    started = time.perf_counter()
    source_settings = json.loads((source/"config.json").read_text())
    block_size = source_settings["display"]["block_size"]
    with np.load(source/"histogram_mapping.npz") as mapping:
        healthy_peak = float(mapping["peak"])
        histogram_edges = mapping["intensity_edges"].copy()
        histogram_cdf = mapping["cumulative_probability"].copy()
        histogram_floor = float(mapping["floor"])
    white = transform["headroom_multiplier"]*healthy_peak
    formula = {"linear": "round(255 * clip(I / white_reference, 0, 1))",
               "gamma": "round(255 * clip(I / white_reference, 0, 1) ** gamma)",
               "asinh": "round(255 * asinh(k * clip(I / white_reference, 0, 1)) / asinh(k))",
               "histogram": "round(255 * healthy_training_log_intensity_CDF(I))"}[mode]
    mapping_name = "display_mapping.json" if "transform" in config else "linear_mapping.json"
    write_json(out/mapping_name, {"healthy_training_peak": healthy_peak, "white_reference": white,
        "gain_to_unit_range": 1/white, "gain_to_8bit": 255/white, "black_offset": 0,
        "formula": formula, "mode": mode, "parameters": transform, "block_sum": block_size,
        "training_source": str((source/"histogram_mapping.npz").relative_to(PROJECT)),
        "shared_across_all_images": True})
    with (source/"dataset.csv").open() as stream:
        original = list(csv.DictReader(stream))
    records = []
    for row in original:
        if row["case"] in ("healthy_train", "healthy_calibration", "healthy_test"):
            if row["case"] == "healthy_train" and config.get("healthy_training_run"):
                continue
            records.append({**row, "role": row["case"], "source_folder": source})
    training_sources = []
    if config.get("healthy_training_run"):
        dense = PROJECT/config["healthy_training_run"]
        if json.loads((dense/"capture_manifest.json").read_text())["status"] != "complete":
            raise RuntimeError("Complete the dense healthy training captures first.")
        with (dense/"training_dataset.csv").open() as stream:
            for row in csv.DictReader(stream):
                records.append({**row, "source_folder": dense})
        training_sources = [dense/"capture_manifest.json", dense/"training_dataset.csv"]
    for row in manifest["frames"]:
        if row["case"] in cases:
            records.append({**row, "source_folder": spherical})
    fields = ("case", "role", "index", "angle_deg", "raw", "image", "defect_ids", "first_seed", "raw_source")
    rows, statistics = [], {}
    for record in records:
        case = record["case"]
        index = int(record["index"])
        raw = record["source_folder"]/record["raw"]
        intensity = block_sum(np.asarray(mi.Bitmap(str(raw))).squeeze().astype(np.float64), block_size)
        q = np.clip(intensity/white, 0, 1)
        if mode == "gamma":
            q = q**transform["gamma"]
        elif mode == "asinh":
            span = transform["asinh_dynamic_range"]
            q = np.arcsinh(span*q)/np.arcsinh(span)
        elif mode == "histogram":
            q = np.interp(np.log10(np.maximum(intensity, histogram_floor)),
                          np.log10(histogram_edges), histogram_cdf, left=0, right=1)
        display = np.rint(255*q).astype(np.uint8)
        path = out/"frames"/case/f"frame_{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(display).save(path)
        row = {key: record.get(key, "") for key in fields}
        row.update(index=index, angle_deg=float(record["angle_deg"]), raw=os.path.relpath(raw, out),
                   image=str(path.relative_to(out)))
        rows.append(row)
        counts = statistics.setdefault(case, {"images": 0, "pixels": 0, "clipped_pixels": 0,
            "positive_intensity_pixels": 0, "positive_pixels_rounded_to_zero": 0})
        counts["images"] += 1
        counts["pixels"] += intensity.size
        counts["clipped_pixels"] += int(np.count_nonzero(intensity > (healthy_peak if mode == "histogram" else white)))
        counts["positive_intensity_pixels"] += int(np.count_nonzero(intensity > 0))
        counts["positive_pixels_rounded_to_zero"] += int(np.count_nonzero((intensity > 0) & (display == 0)))
    with (out/"dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    defects = json.loads((spherical/"defects.json").read_text())
    defects["gears"] = [gear for gear in defects["gears"] if gear["id"] in cases]
    write_json(out/"defects.json", defects)
    sources = [config_path, Path(__file__), source/"dataset.csv", source/"histogram_mapping.npz", spherical/"defects.json"]+training_sources
    write_json(out/"preprocessing_manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(rows), "display_mapping": mapping_name, "transform": transform, "per_case": statistics,
        "raw_capture_reuse": True, "healthy_training_run": config.get("healthy_training_run"),
        "mapping_reference_preserved": True, "elapsed_seconds": time.perf_counter()-started,
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
    print(f"Prepared {len(rows)} {mode} images for one test gear.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("hemisphere_linear.json"))
    parser.add_argument("--wait-for-captures", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_captures)
