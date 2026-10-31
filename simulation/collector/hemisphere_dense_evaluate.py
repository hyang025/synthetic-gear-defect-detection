"""Fit and evaluate the four requested denser-training PatchCore branches."""

import argparse
import json
from pathlib import Path

from hemisphere_linear_evaluate import PROJECT, main as evaluate


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    out = PROJECT/config["output"]
    for method in config["methods"]:
        branch = out/method["id"]
        if (branch/"evaluation_manifest.json").exists():
            print(f"Preserved completed branch: {method['label']}", flush=True)
            continue
        evaluate(branch/"config.json", wait=wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("hemisphere_dense.json"))
    parser.add_argument("--wait-for-preprocessing", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_preprocessing)
