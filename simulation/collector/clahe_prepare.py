"""Apply fixed log-input CLAHE and prepare two additional test gears."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(config_path, additional=False, wait=False, evaluate=False):
    config = json.loads(config_path.read_text())
    root, source = PROJECT/config["output"], PROJECT/config["source_run"]
    pack = root/"inputs"
    name = "additional" if additional else "reference"
    pack_manifest = pack/f"{name}_manifest.json"
    while True:
        try:
            if json.loads(pack_manifest.read_text())["status"] == "complete":
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if not wait:
            raise RuntimeError("Complete the corresponding summed EXR intensity pack first.")
        time.sleep(5)
    transform = config["transform"]
    if transform["precision_bits"] != 8:
        raise ValueError("Use the configured 8-bit CLAHE input and output.")
    cv2.setNumThreads(1)
    clahe = cv2.createCLAHE(clipLimit=transform["clip_limit"], tileGridSize=tuple(transform["tile_grid"]))
    with np.load(source/"histogram_mapping.npz") as mapping:
        peak = float(mapping["peak"])
        edges, cdf = mapping["intensity_edges"].copy(), mapping["cumulative_probability"].copy()
        histogram_floor = float(mapping["floor"])
    white, floor = peak*transform["headroom_multiplier"], peak*transform["floor_relative_to_healthy_peak"]
    with (pack/f"{name}_dataset.csv").open() as stream:
        records = list(csv.DictReader(stream))
    destinations = [("clahe", "CLAHE · log input")]
    if additional:
        destinations.append(("histogram", "Histogram equalized"))
    started = time.perf_counter()
    for method, label in destinations:
        out = root/method
        out.mkdir(parents=True, exist_ok=True)
        manifest_name = "additional_preprocessing_manifest.json" if additional else "preprocessing_manifest.json"
        if (out/manifest_name).exists():
            raise FileExistsError("Preserve the completed image preprocessing.")
        if not additional:
            branch = {key: config[key] for key in ("source_run", "spherical_run", "healthy_training_run", "test_gear",
                "validation_gear", "exclude_validation_defects", "transform", "threshold_policy", "display_label")}
            branch["output"] = str(out.relative_to(PROJECT))
            write_json(out/"config.json", branch)
            write_json(out/"display_mapping.json", {"mode": "clahe", "parameters": transform,
                "healthy_reference_peak": peak, "log_floor": floor, "log_white_reference": white,
                "precompression": "round(255 * clip(log(max(I, floor) / floor) / log(white / floor), 0, 1))",
                "local_mapping": "OpenCV CLAHE on the 8-bit log-intensity input",
                "fixed_parameters_across_all_images": True, "per_image_local_histograms": True,
                "opencv_version": cv2.__version__, "block_sum": 4})
        arrays = np.load(pack/f"{name}_arrays.npz")
        case_arrays = {case: arrays[case] for case in arrays.files}
        arrays.close()
        rows, statistics = [], {}
        for record in records:
            case, index = record["case"], int(record["index"])
            intensity = case_arrays[case][int(record["array_index"])]
            if method == "clahe":
                q = np.clip(np.log(np.maximum(intensity, floor)/floor)/np.log(white/floor), 0, 1)
                display = clahe.apply(np.rint(255*q).astype(np.uint8))
            else:
                q = np.interp(np.log10(np.maximum(intensity, histogram_floor)), np.log10(edges), cdf, left=0, right=1)
                display = np.rint(255*q).astype(np.uint8)
            image = out/"frames"/case/f"frame_{index:03d}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(display).save(image)
            row = {**record, "index": index, "angle_deg": float(record["angle_deg"]),
                "raw": os.path.relpath(pack/record["raw"], out), "image": str(image.relative_to(out))}
            rows.append(row)
            counts = statistics.setdefault(case, {"images": 0, "pixels": 0, "positive_intensity_pixels": 0,
                "positive_pixels_rounded_to_zero": 0, "zero_intensity_pixels_lifted": 0})
            counts["images"] += 1
            counts["pixels"] += intensity.size
            counts["positive_intensity_pixels"] += int(np.count_nonzero(intensity > 0))
            counts["positive_pixels_rounded_to_zero"] += int(np.count_nonzero((intensity > 0) & (display == 0)))
            counts["zero_intensity_pixels_lifted"] += int(np.count_nonzero((intensity == 0) & (display > 0)))
        table = out/("additional_dataset.csv" if additional else "dataset.csv")
        with table.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        defect_source = root/"defects.json" if additional else PROJECT/config["spherical_run"]/"defects.json"
        defects = json.loads(defect_source.read_text())
        cases = {row["case"] for row in records}
        defects["gears"] = [gear for gear in defects["gears"] if gear["id"] in cases]
        write_json(out/("additional_defects.json" if additional else "defects.json"), defects)
        sources = [config_path, Path(__file__), pack_manifest, pack/f"{name}_dataset.csv",
                   source/"histogram_mapping.npz", defect_source]
        write_json(out/manifest_name, {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "images": len(rows), "method": label, "transform": transform if method == "clahe" else {"mode": "histogram"},
            "per_case": statistics, "full_screen_retained": True, "angular_weighting": False,
            "peak_mask": False, "opencv_version": cv2.__version__, "elapsed_seconds": time.perf_counter()-started,
            "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
        print(f"Prepared {len(rows)} {label} {name} images.", flush=True)
    if evaluate:
        if additional:
            raise ValueError("Fit the reference bank before evaluating the additional gears.")
        from hemisphere_linear_evaluate import main as fit_reference
        fit_reference(root/"clahe/config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("clahe.json"))
    parser.add_argument("--additional", action="store_true")
    parser.add_argument("--wait-for-pack", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.additional, args.wait_for_pack, args.evaluate)
