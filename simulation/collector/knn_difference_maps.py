"""Difference maps for selected tuning cavities: damage against render noise at two bin sizes."""

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".tools/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from knn_vectors import angle_bins

CAVITIES = [("d010_gear_2", "G2_D2"), ("d010_gear_2", "G2_D5"), ("d010_gear_3", "G3_D3"),
            ("d010_gear_2", "G2_D3"), ("d010_gear_3", "G3_D4")]
STATUS = {"G2_D2": "missed", "G2_D5": "missed", "G3_D3": "missed", "G2_D3": "caught", "G3_D4": "caught"}


def main():
    config = json.loads((PROJECT/"collector/knn_pca.json").read_text())
    source, out = PROJECT/config["source_run"], PROJECT/config["output"]
    manifest = json.loads((source/"capture_manifest.json").read_text())
    cases = json.loads((source/"cases.json").read_text())
    raw = {(f["case"], f["index"]): source/f["raw"] for f in manifest["frames"]}
    step = cases["step_deg"]
    mi.set_variant("scalar_rgb")

    def share(path):
        image = np.asarray(mi.Bitmap(str(path))).astype(np.float64).ravel()
        return image/image.sum()

    grids = {size: angle_bins(manifest["trace_config"], config["half_range_deg"], size) for size in (2.0, 0.5)}
    summary, maps = [], []
    for case, name in CAVITIES:
        defect = next(d for c in cases["cases"] if c["case"] == case for d in c["defects"] if d["id"] == name)
        index = min(defect["window_indices"], key=lambda i: abs((i*step-defect["crossing_angle_deg"]+180) % 360-180))
        damaged, healthy_a, healthy_b = (share(raw[(c, index)]) for c in (case, "healthy_a", "healthy_b"))
        row = {"cavity": name, "status": STATUS[name], "kind": defect["kind"], "diameter_mm": defect["diameter_mm"],
               "frame_index": int(index), "light_change_pct": None}
        for size, (flat, count) in grids.items():
            inside = flat >= 0

            def binned(values):
                return np.sqrt(np.bincount(flat[inside], weights=values[inside], minlength=count*count)).reshape(count, count)

            signal, noise = binned(damaged)-binned(healthy_a), binned(healthy_b)-binned(healthy_a)
            top = np.sort((signal**2).ravel())[::-1]
            row[f"{size:g}deg"] = {"max_abs_signal_over_max_abs_noise": round(float(abs(signal).max()/abs(noise).max()), 2),
                                   "norm_signal_over_norm_noise": round(float(np.linalg.norm(signal)/np.linalg.norm(noise)), 2),
                                   "share_of_signal_energy_in_top_10_bins": round(float(top[:10].sum()/top.sum()), 3)}
            if size == 2.0:
                maps.append((name, signal, noise))
        summary.append(row)
        print(json.dumps(row), flush=True)
    (out/"difference_maps.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(2, len(maps), figsize=(3*len(maps), 6.2))
    limit = max(abs(s).max() for _, s, _ in maps)
    extent = [-config["half_range_deg"], config["half_range_deg"]]*2
    for column, (name, signal, noise) in enumerate(maps):
        for row_index, (values, title) in enumerate(((signal, f"{name} · {STATUS[name]}\ndamaged − healthy A"),
                                                      (noise, "healthy B − healthy A"))):
            ax = axes[row_index, column]
            shown = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower", extent=extent)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("left-right angle (°)", fontsize=8)
            ax.set_ylabel("up-down angle (°)", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.colorbar(shown, ax=axes, shrink=0.6, label="difference in √(light share)")
    fig.savefig(out/"images/difference_maps.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
