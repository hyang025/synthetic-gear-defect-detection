"""Compare two spherical-cavity sizes using a frozen histogram PatchCore model."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time

from prototype_patchcore import PROJECT, figures, write_csv, write_json
from validation_threshold import digest, load_rows
from hemisphere_threshold import score_images
import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def comparison_figure(out, rows, threshold):
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    ax.bar(positions-0.18, [row["small_max_score"]/threshold for row in rows],
           width=0.36, color="#247ba0", label="Ø0.10 mm · depth 50 µm")
    ax.bar(positions+0.18, [row["large_max_score"]/threshold for row in rows],
           width=0.36, color="#d1495b", label="Ø0.20 mm · depth 100 µm")
    ax.axhline(1, color="black", ls="--", lw=1, label="Fixed F1 threshold")
    ax.set(title="Histogram equalization · matched defect positions", xlabel="Defect",
           ylabel="Peak passage score / threshold", xticks=positions,
           xticklabels=[row["defect"] for row in rows])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(out/"images/size_comparison.png", dpi=160)
    plt.close(fig)


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    out = PROJECT/config["output"]
    reference = PROJECT/config["histogram_baseline"]
    small = PROJECT/config["small_defect_baseline"]
    if (out/"evaluation_manifest.json").exists() or (out/"metrics.json").exists():
        raise FileExistsError("Preserve the completed size comparison.")
    capture_path = out/"capture_manifest.json"
    while True:
        try:
            captures = json.loads(capture_path.read_text())
            if captures["status"] == "complete":
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if not wait:
            raise RuntimeError("Complete the three configured gear revolutions first.")
        time.sleep(5)
    (out/"images").mkdir(parents=True, exist_ok=True)
    baseline = json.loads((reference/"metrics.json").read_text())
    saved_threshold = json.loads((reference/"threshold.json").read_text())
    threshold = saved_threshold["threshold"]
    write_json(out/"threshold.json", {
        "threshold": threshold, "policy": config["threshold_policy"],
        "validation": saved_threshold["validation"],
        "selection_curve_inclusive": saved_threshold["selection_curve_inclusive"],
        "comparison": "score > threshold", "frozen_before_test_evaluation": True})
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use the RTX GPU for this evaluation.")
    started = time.perf_counter()
    checkpoint = torch.load(reference/"patchcore_model.pt", map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    model = PatchcoreModel(layers=settings["layers"], backbone=settings["backbone"],
        pre_trained=False, num_neighbors=settings["num_neighbors"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)
    bank_shape = list(model.memory_bank.shape)
    predictions = score_images(model, out, load_rows(out/"test_dataset.csv"), settings, device)
    del model
    torch.cuda.empty_cache()
    healthy = [row for row in load_rows(reference/"predictions.csv") if row["case"] == "healthy_test"]
    fields = ("case", "role", "index", "angle_deg", "raw", "image", "defect_ids", "first_seed", "raw_source", "score")
    for source in healthy:
        row = {key: source[key] for key in fields}
        row.update(raw=os.path.relpath(reference/source["raw"], out),
                   image=os.path.relpath(reference/source["image"], out))
        predictions.append(row)
    for row in predictions:
        row.update(threshold=threshold, predicted_anomaly=row["score"] > threshold)
    healthy = [row for row in predictions if row["case"] == "healthy_test"]
    defects = copy.deepcopy(json.loads((out/"defects.json").read_text()))
    old_defects = {d["id"]: d for gear in json.loads((small/"defect_predictions.json").read_text())["gears"]
                   for d in gear["defects"]}
    old_predictions = {(row["case"], row["index"]): row for row in load_rows(small/"predictions.csv")
                       if row["case"] != "healthy_test"}
    if any(abs(row["angle_deg"]-old_predictions[(row["case"], row["index"])]["angle_deg"]) > 1e-8
           or int(row["first_seed"]) != int(old_predictions[(row["case"], row["index"])]["first_seed"])
           for row in predictions if row["case"] != "healthy_test"):
        raise RuntimeError("The size comparison requires matched encoder grids and ray seeds.")
    defect_rows, comparison, per_gear = [], [], []
    for gear in defects["gears"]:
        outcomes = []
        for defect in gear["defects"]:
            old = old_defects[defect["id"]]
            if np.linalg.norm(np.asarray(defect["surface_point_mm"])-old["surface_point_mm"]) > 1e-8:
                raise RuntimeError("Keep the saved spherical-cavity center fixed for the size comparison.")
            selected = [row for row in predictions if row["case"] == gear["id"]
                        and defect["id"] in row["defect_ids"].split(";")]
            peak = max(selected, key=lambda row: row["score"])
            detected = peak["score"] > threshold
            defect.update(max_score=peak["score"], peak_frame_index=peak["index"],
                          peak_angle_deg=peak["angle_deg"], passage_frames=len(selected), detected=detected)
            record = {"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"], "side": defect["side"],
                "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                "passage_frames": len(selected), "max_score": peak["score"], "threshold": threshold,
                "detected": detected, "peak_angle_deg": peak["angle_deg"]}
            defect_rows.append(record)
            outcomes.append(record)
            comparison.append({"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"],
                "small_diameter_mm": old["diameter_mm"], "small_depth_um": old["depth_mm"]*1000,
                "large_diameter_mm": defect["diameter_mm"], "large_depth_um": defect["depth_mm"]*1000,
                "small_max_score": old["max_score"], "large_max_score": peak["score"],
                "score_change": peak["score"]-old["max_score"], "threshold": threshold,
                "small_detected": old["max_score"] > threshold, "large_detected": detected})
        tp = sum(row["detected"] for row in outcomes)
        per_gear.append({"gear": gear["id"], "defects": len(outcomes), "tp": tp, "fn": len(outcomes)-tp})
    tp = sum(row["detected"] for row in defect_rows)
    small_tp = sum(row["small_detected"] for row in comparison)
    fp = sum(row["predicted_anomaly"] for row in healthy)
    metrics = {"method": "Histogram equalized", "threshold_selection": "Frozen validation F1",
        "threshold": threshold, "healthy_training_frames": baseline["healthy_training_frames"],
        "bank_shape": bank_shape, "defects": len(defect_rows), "tp": tp, "fn": len(defect_rows)-tp,
        "detection_rate_percent": 100*tp/len(defect_rows), "healthy_test_frames": len(healthy), "fp": fp,
        "false_positive_rate_percent": 100*fp/len(healthy), "per_gear": per_gear,
        "small_defect_baseline": {"diameter_mm": 0.10, "depth_um": 50, "tp": small_tp,
            "fn": len(comparison)-small_tp, "detection_rate_percent": 100*small_tp/len(comparison)},
        "evaluation_unit": "Detection/FN per physical defect; FP per unique healthy test frame",
        "frozen_reference": config["histogram_baseline"], "test_gears_used_for_fitting": False,
        "paired_ray_seeds": True, "fixed_defect_centers": True, "fixed_passage_half_width_deg": 0.4}
    write_json(out/"metrics.json", metrics)
    write_json(out/"defect_predictions.json", defects)
    write_csv(out/"defect_predictions.csv", defect_rows)
    write_csv(out/"predictions.csv", predictions)
    write_csv(out/"size_comparison.csv", comparison)
    figures(out, predictions, defects, threshold)
    comparison_figure(out, comparison, threshold)
    sources = [config_path, Path(__file__), PROJECT/"collector/hemisphere_threshold.py", capture_path,
               out/"defects.json", out/"test_dataset.csv", reference/"patchcore_model.pt",
               reference/"threshold.json", reference/"predictions.csv", small/"defect_predictions.json",
               small/"predictions.csv", PROJECT/config["source_run"]/"histogram_mapping.npz"]
    write_json(out/"evaluation_manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "cnn_and_memory_fixed": True, "threshold_fixed": True,
        "healthy_test_scores_reused": True, "angular_weighting": False, "gray_band": False,
        "elapsed_seconds": time.perf_counter()-started,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "anomalib", "timm")},
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("hemisphere_size.json"))
    parser.add_argument("--wait-for-captures", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_captures)
