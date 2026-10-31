"""Compare linear, histogram, gamma, and shadow-boost PatchCore results."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(config_path=None):
    config = json.loads(config_path.read_text()) if config_path else None
    out = PROJECT/config["output"] if config else PROJECT/"outputs/patchcore_hemisphere_enhancement_25mm"
    if (out/"comparison.json").exists():
        raise FileExistsError("Preserve the completed enhancement comparison before a new run.")
    (out/"images").mkdir(parents=True, exist_ok=True)
    specifications = [(method["label"], out/method["id"]) for method in config["methods"]] if config else [
                      ("Histogram equalized", PROJECT/"outputs/patchcore_hemisphere_0p1_25mm"),
                      ("Shared linear gain", PROJECT/"outputs/patchcore_hemisphere_linear_25mm"),
                      ("Gamma 0.25", PROJECT/"outputs/patchcore_hemisphere_gamma_25mm"),
                      ("Asinh shadow boost", PROJECT/"outputs/patchcore_hemisphere_shadowboost_25mm")]
    methods, table, defect_table, sources = [], [], [], [Path(__file__)]
    for label, run in specifications:
        metrics = json.loads((run/"metrics.json").read_text())
        defects = [row for row in rows(run/"defect_predictions.csv") if row["gear"] == "gear_1"]
        legacy = metrics["healthy_percentile_on_same_geometry"]
        healthy_flag = "healthy_percentile_detected"
        for threshold_label, threshold, flag, fp in (("Anomalib F1", metrics["threshold"], "detected", metrics["fp"]),
                ("Healthy 99th percentile", legacy["threshold"], healthy_flag, legacy["fp"])):
            tp = sum(row[flag] == "True" for row in defects)
            table.append({"method": label, "threshold_selection": threshold_label, "threshold": threshold,
                "defects": len(defects), "tp": tp, "fn": len(defects)-tp, "healthy_test_frames": metrics["healthy_test_frames"],
                "fp": fp, "false_positive_rate_percent": 100*fp/metrics["healthy_test_frames"]})
        for defect in defects:
            defect_table.append({"method": label, "defect": defect["defect"], "kind": defect["kind"],
                "max_score": float(defect["max_score"]), "healthy_percentile_threshold": legacy["threshold"],
                "score_over_healthy_threshold": float(defect["max_score"])/legacy["threshold"],
                "detected_F1": defect["detected"], "detected_healthy_percentile": defect[healthy_flag]})
        methods.append({"label": label, "run": run, "metrics": metrics, "defects": defects,
                        "validation": rows(run/"validation_labeled_scores.csv")})
        sources.extend([run/"metrics.json", run/"defect_predictions.csv", run/"validation_labeled_scores.csv"])

    reference = PROJECT/config["spherical_run"] if config else methods[0]["run"]
    geometry = json.loads((reference/"defects.json").read_text())
    gear = next(gear for gear in geometry["gears"] if gear["id"] == "gear_1")
    captures = [row for row in json.loads((reference/"capture_manifest.json").read_text())["frames"] if row["case"] == "gear_1"]
    centers = [min(captures, key=lambda row: abs((row["angle_deg"]-defect["crossing_angle_deg"]+180) % 360-180))
               for defect in gear["defects"]]
    fig, axs = plt.subplots(4, 5, figsize=(14, 11), layout="constrained")
    for row_index, method in enumerate(methods):
        for col, capture in enumerate(centers):
            ax = axs[row_index, col]
            path = method["run"]/"frames/gear_1"/f"frame_{capture['index']:03d}.png"
            ax.imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
            if row_index == 0:
                ax.set_title(f"{gear['defects'][col]['id']} · {capture['angle_deg']:.2f}°", fontsize=10)
            if col == 0:
                ax.text(-0.08, 0.5, method["label"], rotation=90, va="center", ha="center", transform=ax.transAxes)
            ax.axis("off")
    fig.suptitle("Same defect-center captures · four display mappings")
    fig.savefig(out/"images/input_comparison.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, method in zip(axs.ravel(), methods):
        reference_threshold = method["metrics"]["healthy_percentile_on_same_geometry"]["threshold"]
        healthy = [float(row["score"])/reference_threshold for row in method["validation"] if row["label"] == "0"]
        damaged = [float(row["score"])/reference_threshold for row in method["validation"] if row["label"] == "1"]
        edges = np.linspace(min(healthy+damaged), max(healthy+damaged), 30)
        ax.hist(healthy, bins=edges, density=True, alpha=0.55, color="#247ba0", label="Healthy validation")
        ax.hist(damaged, bins=edges, density=True, alpha=0.55, color="#d1495b", label="Damage passages")
        ax.axvline(1, color="black", ls="--", label="Healthy threshold")
        ax.axvline(method["metrics"]["threshold"]/reference_threshold, color="#6a4c93", ls=":", label="F1 threshold")
        ax.set(title=method["label"], xlabel="Score / healthy 99th-percentile threshold", ylabel="Density")
        ax.legend(fontsize=8)
    fig.suptitle("Validation score distributions · matched preprocessing")
    fig.savefig(out/"images/validation_scores.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    positions = np.arange(5)
    colors = ("#247ba0", "#6a4c93", "#e09f3e", "#d1495b")
    for index, (method, color) in enumerate(zip(methods, colors)):
        threshold = method["metrics"]["healthy_percentile_on_same_geometry"]["threshold"]
        ax.bar(positions+(index-1.5)*0.18, [float(row["max_score"])/threshold for row in method["defects"]],
               width=0.18, color=color, label=method["label"])
    ax.axhline(1, color="black", ls="--", lw=1)
    ax.set(title="Five test defects · healthy-calibrated thresholds", ylabel="Peak score / healthy threshold",
           xlabel="Defect", xticks=positions, xticklabels=[defect["id"] for defect in gear["defects"]])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out/"images/defect_scores.png", dpi=160)
    plt.close(fig)
    write_csv(out/"comparison.csv", table)
    write_csv(out/"defect_comparison.csv", defect_table)
    (out/"comparison.json").write_text(json.dumps({"test_gear": "gear_1", "diameter_mm": 0.10, "depth_um": 50,
        "shared_raw_captures": True, "pretrained_CNN_shared": True, "memory_and_threshold_matched_to_preprocessing": True,
        "healthy_training_frames": methods[0]["metrics"].get("healthy_training_frames", 440),
        "exclude_validation_defects": config.get("exclude_validation_defects", []) if config else [],
        "results": table, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}},
        indent=2, allow_nan=False)+"\n")
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    args = parser.parse_args()
    main(args.config.resolve() if args.config else None)
