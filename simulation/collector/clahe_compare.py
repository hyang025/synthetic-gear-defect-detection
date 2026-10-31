"""Draw matched-input and score comparisons for CLAHE and histogram equalization."""

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".tools/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(config_path):
    config = json.loads(config_path.read_text())
    root = PROJECT/config["output"]
    if json.loads((root/"comparison.json").read_text())["status"] != "complete":
        raise RuntimeError("Complete the two frozen-model evaluations first.")
    images = root/"images"
    images.mkdir(exist_ok=True)
    methods = [("histogram", "Histogram equalized", "#247ba0", PROJECT/config["histogram_baseline"]),
               ("clahe", "CLAHE · log input", "#e09f3e", root/"clahe")]
    gears = json.loads((root/"defects.json").read_text())["gears"]
    captures = json.loads((root/"capture_manifest.json").read_text())["frames"]
    fig, axs = plt.subplots(4, 5, figsize=(14, 11), layout="constrained")
    for gear_index, gear in enumerate(gears):
        frames = [row for row in captures if row["case"] == gear["id"]]
        for col, defect in enumerate(gear["defects"]):
            center = min(frames, key=lambda row: abs((row["angle_deg"]-defect["crossing_angle_deg"]+180) % 360-180))
            for method_index, (method, label, _, _) in enumerate(methods):
                ax = axs[2*gear_index+method_index, col]
                ax.imshow(Image.open(root/method/"frames"/gear["id"]/f"frame_{center['index']:03d}.png"),
                          cmap="gray", vmin=0, vmax=255)
                if method_index == 0:
                    ax.set_title(f"{defect['id']} · {center['angle_deg']:.2f}°", fontsize=10)
                if col == 0:
                    ax.text(-0.08, 0.5, f"{gear['id']} · {label}", rotation=90, va="center", ha="center",
                            transform=ax.transAxes)
                ax.axis("off")
    fig.suptitle("Two new test gears · matched defect-center captures")
    fig.savefig(images/"input_comparison.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    for method_index, (method, label, color, _) in enumerate(methods):
        run = root/method/"extended"
        metrics = json.loads((run/"metrics.json").read_text())
        defects = read_rows(run/"defect_predictions.csv")
        scores = np.asarray([float(row["max_score"]) for row in defects])
        positions = np.arange(len(defects))
        limits = [metrics["threshold"], metrics["healthy_percentile_on_same_geometry"]["threshold"]]
        for ax, limit, title in zip(axs, limits, ("Validation-F1 threshold", "Healthy 99th-percentile threshold")):
            ax.bar(positions+(method_index-0.5)*0.34, scores/limit, width=0.34, color=color, label=label)
            ax.set(title=title, ylabel="Peak score / threshold", xticks=positions,
                   xticklabels=[row["defect"] for row in defects])
    for ax in axs:
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Three test gears · fifteen spherical cavities")
    fig.savefig(images/"defect_scores.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, (_, label, _, reference) in zip(axs, methods):
        metrics = json.loads((reference/"metrics.json").read_text())
        limit = metrics["healthy_percentile_on_same_geometry"]["threshold"]
        rows = read_rows(reference/"validation_labeled_scores.csv")
        healthy = [float(row["score"])/limit for row in rows if row["label"] == "0"]
        damaged = [float(row["score"])/limit for row in rows if row["label"] == "1"]
        bins = np.linspace(min(healthy+damaged), max(healthy+damaged), 30)
        ax.hist(healthy, bins=bins, density=True, alpha=0.55, color="#247ba0", label="Healthy calibration")
        ax.hist(damaged, bins=bins, density=True, alpha=0.55, color="#d1495b", label="Damage passages")
        ax.axvline(1, color="black", ls="--", label="Healthy threshold")
        ax.axvline(metrics["threshold"]/limit, color="#6a4c93", ls=":", label="F1 threshold")
        ax.set(title=label, xlabel="Score / healthy threshold", ylabel="Density")
        ax.legend(fontsize=8)
    fig.suptitle("Validation score distributions · V_D1–V_D4")
    fig.savefig(images/"validation_scores.png", dpi=160)
    plt.close(fig)
    print("Saved matched inputs, defect scores, and validation distributions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("clahe.json"))
    main(parser.parse_args().config.resolve())
