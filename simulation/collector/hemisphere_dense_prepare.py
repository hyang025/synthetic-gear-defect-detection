"""Prepare four brightness mappings for the denser healthy-training experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from prototype_capture import PROJECT, write_json
from hemisphere_linear_prepare import main as prepare


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    out = PROJECT/config["output"]
    out.mkdir(parents=True, exist_ok=True)
    branches = []
    for method in config["methods"]:
        branch = out/method["id"]
        branch.mkdir(parents=True, exist_ok=True)
        path = branch/"config.json"
        settings = {key: config[key] for key in ("source_run", "spherical_run", "test_gear", "validation_gear",
                    "exclude_validation_defects", "threshold_policy")}
        settings.update(output=str(branch.relative_to(PROJECT)), healthy_training_run=config["output"],
                        display_label=method["label"], transform=method["transform"])
        if path.exists() and json.loads(path.read_text()) != settings:
            raise RuntimeError("Preserve the existing branch configuration.")
        if not path.exists():
            write_json(path, settings)
        if not (branch/"preprocessing_manifest.json").exists():
            branches.append(path)
    while True:
        try:
            ready = json.loads((out/"capture_manifest.json").read_text())["status"] == "complete"
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
        if ready:
            break
        if not wait:
            raise RuntimeError("Complete the denser healthy capture grid first.")
        time.sleep(5)
    with ThreadPoolExecutor(max_workers=len(config["methods"])) as pool:
        list(pool.map(prepare, branches))
    print("Four matched grayscale datasets are ready.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("hemisphere_dense.json"))
    parser.add_argument("--wait-for-captures", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_captures)
