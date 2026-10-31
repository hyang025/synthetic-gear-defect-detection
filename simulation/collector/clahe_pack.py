"""Read EXRs into summed intensity arrays for the CLAHE environment."""

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


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(config_path, additional=False, wait=False):
    config = json.loads(config_path.read_text())
    out, source = PROJECT/config["output"], PROJECT/config["source_run"]
    pack = out/"inputs"
    pack.mkdir(parents=True, exist_ok=True)
    name = "additional" if additional else "reference"
    manifest_path = pack/f"{name}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Preserve the saved intensity pack.")
    records, sources = [], [config_path, Path(__file__), PROJECT/"collector/low_light.py"]
    if additional:
        capture_path = out/"capture_manifest.json"
        while True:
            try:
                captures = json.loads(capture_path.read_text())
                if captures["status"] == "complete":
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if not wait:
                raise RuntimeError("Complete both new test-gear revolutions first.")
            time.sleep(5)
        records = [{**row, "source_folder": out} for row in captures["frames"]]
        sources.extend([capture_path, out/"defects.json"])
    else:
        dense = PROJECT/config["healthy_training_run"]
        spherical = PROJECT/config["spherical_run"]
        for row in read_rows(source/"dataset.csv"):
            if row["case"] in ("healthy_calibration", "healthy_test"):
                records.append({**row, "role": row["case"], "source_folder": source})
        for row in read_rows(dense/"training_dataset.csv"):
            records.append({**row, "source_folder": dense})
        old = json.loads((spherical/"capture_manifest.json").read_text())
        for row in old["frames"]:
            if row["case"] in (config["validation_gear"], config["test_gear"]):
                records.append({**row, "source_folder": spherical})
        sources.extend([source/"dataset.csv", dense/"training_dataset.csv", dense/"capture_manifest.json",
                        spherical/"capture_manifest.json", spherical/"defects.json"])
    block_size = json.loads((source/"config.json").read_text())["display"]["block_size"]
    arrays, rows = {}, []
    fields = ("case", "role", "index", "angle_deg", "raw", "defect_ids", "first_seed", "raw_source", "array_index")
    started = time.perf_counter()
    for record in records:
        raw = record["source_folder"]/record["raw"]
        array = block_sum(np.asarray(mi.Bitmap(str(raw))).squeeze().astype(np.float64), block_size)
        values = arrays.setdefault(record["case"], [])
        row = {key: record.get(key, "") for key in fields}
        row.update(index=int(record["index"]), angle_deg=float(record["angle_deg"]),
                   raw=os.path.relpath(raw, pack), array_index=len(values))
        rows.append(row)
        values.append(array)
    data_path, table_path = pack/f"{name}_arrays.npz", pack/f"{name}_dataset.csv"
    np.savez_compressed(data_path, **{case: np.stack(values) for case, values in arrays.items()})
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(manifest_path, {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(rows), "block_sum": block_size, "dtype": "float64", "raw_capture_reuse": True,
        "per_case": {case: len(values) for case, values in arrays.items()},
        "arrays_sha256": digest(data_path), "dataset_sha256": digest(table_path),
        "elapsed_seconds": time.perf_counter()-started,
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
    print(f"Packed {name}: {len(rows)} summed EXR intensity arrays.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("clahe.json"))
    parser.add_argument("--additional", action="store_true")
    parser.add_argument("--wait-for-captures", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.additional, args.wait_for_captures)
