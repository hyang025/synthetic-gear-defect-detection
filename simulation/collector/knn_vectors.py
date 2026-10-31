"""Turn each saved screen image into a 2D angular-distribution vector for the KNN and PCA detector."""

import argparse
import json
from pathlib import Path

import mitsuba as mi
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent


def angle_bins(trace, half_range, step):
    """Flat bin index for every screen pixel, or -1 outside the angular window."""
    width_mm, pixels = trace["screen"]["width_mm"], trace["screen"]["width_px"]
    position = (np.arange(pixels)+0.5)/pixels*width_mm-width_mm/2
    angle = np.degrees(np.arctan(position/trace["collector"]["focal_length_mm"]))
    count = int(round(2*half_range/step))
    column = np.floor((angle+half_range)/step).astype(int)
    column[(column < 0) | (column >= count)] = -1
    v, u = np.meshgrid(column, column, indexing="ij")
    index = np.where((u >= 0) & (v >= 0), v*count+u, -1)
    return index.ravel(), count


def main(config_path):
    config = json.loads(config_path.read_text())
    source = PROJECT/config["source_run"]
    out = PROJECT/config["output"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/"capture_manifest.json").read_text())
    cases = json.loads((source/"cases.json").read_text())
    step = cases["step_deg"]
    pitch = 360/config["tooth_count"]
    half_label = config["damaged_half_width_deg"]
    index, count = angle_bins(manifest["trace_config"], config["half_range_deg"], config["bin_deg"])
    inside = index >= 0
    defects = {case["case"]: case["defects"] for case in cases["cases"]}
    mi.set_variant("scalar_rgb")
    frames = sorted(manifest["frames"], key=lambda f: (f["case"], f["index"]))
    binned = np.zeros((len(frames), count*count), np.float32)
    meta = {key: [] for key in ("case", "role", "index", "angle_deg", "segment", "label", "defect_id",
                                "kind", "diameter_mm", "total", "outside_window")}
    for row, frame in enumerate(frames):
        image = np.asarray(mi.Bitmap(str(source/frame["raw"]))).astype(np.float64).ravel()
        total = image.sum()
        binned[row] = np.bincount(index[inside], weights=image[inside], minlength=count*count)
        angle = frame["angle_deg"]
        hit = [d for d in defects[frame["case"]] if abs((angle-d["crossing_angle_deg"]+180) % 360-180) <= half_label+1e-9]
        meta["case"].append(frame["case"])
        meta["role"].append(frame["role"])
        meta["index"].append(frame["index"])
        meta["angle_deg"].append(angle)
        meta["segment"].append(int(angle//pitch))
        meta["label"].append(1 if hit else 0)
        meta["defect_id"].append(hit[0]["id"] if hit else "")
        meta["kind"].append(hit[0]["kind"] if hit else "")
        meta["diameter_mm"].append(hit[0]["diameter_mm"] if hit else 0.0)
        meta["total"].append(total)
        meta["outside_window"].append(1-image[inside].sum()/total)
        if (row+1) % 500 == 0:
            print(f"Vectors {row+1}/{len(frames)}", flush=True)
    tag = config.get("tag", "")
    np.savez_compressed(out/f"vectors{tag}.npz", binned=binned, bin_count=count,
                        **{key: np.asarray(value) for key, value in meta.items()})
    (out/f"config{tag}.json").write_text(json.dumps(config, indent=2)+"\n")
    print(f"Saved {len(frames)} vectors of {count}×{count} bins to {(out/f'vectors{tag}.npz').relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("knn_pca.json"))
    args = parser.parse_args()
    main(args.config.resolve())
