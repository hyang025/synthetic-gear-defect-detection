"""Recalibrate the original PatchCore bank with Anomalib's F1 threshold."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace

from prototype_patchcore import PROJECT, chunks, figures, images_tensor, write_csv, write_json
import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.metrics import F1AdaptiveThreshold
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["index"], row["angle_deg"] = int(row["index"]), float(row["angle_deg"])
        if "score" in row:
            row["score"] = float(row["score"])
    return rows


def comparison_figures(out, healthy, positives, defect_rows, old_threshold, threshold):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    values = [row["score"] for row in healthy+positives]
    edges = np.linspace(min(values), max(values), 35)
    ax.hist([row["score"] for row in healthy], bins=edges, density=True, alpha=0.6,
            color="#247ba0", label=f"Healthy validation · {len(healthy)} frames")
    ax.hist([row["score"] for row in positives], bins=edges, density=True, alpha=0.6,
            color="#d1495b", label=f"Damage passages · {len(positives)} frames")
    ax.axvline(old_threshold, color="black", ls=":", lw=1.4, label="Healthy 99th-percentile threshold")
    ax.axvline(threshold, color="#6a4c93", ls="--", lw=1.4, label="Anomalib F1 threshold")
    ax.set(title="Separate validation data · original PatchCore scores", xlabel="Raw image anomaly score",
           ylabel="Score density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(out/"images/validation_scores.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    positions = np.arange(len(defect_rows))
    values = np.asarray([row["max_score"] for row in defect_rows])
    ax.bar(positions-0.18, values/old_threshold, width=0.36, color="#247ba0", label="Healthy percentile")
    ax.bar(positions+0.18, values/threshold, width=0.36, color="#d1495b", label="Anomalib F1")
    ax.axhline(1, color="black", ls="--", lw=0.8)
    ax.set(title="Original bank · fifteen test defects", xlabel="Defect", ylabel="Score / threshold",
           xticks=positions, xticklabels=[row["defect"] for row in defect_rows])
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out/"images/threshold_comparison.png", dpi=150)
    plt.close(fig)


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    source, out = PROJECT/config["source_run"], PROJECT/config["output"]
    if (out/"recalibration_manifest.json").exists() or (out/"threshold.json").exists():
        raise FileExistsError("Preserve the saved threshold evaluation before a new run.")
    capture_path = out/"capture_manifest.json"
    if wait:
        print("Waiting for separate validation captures...", flush=True)
        while True:
            try:
                if json.loads(capture_path.read_text())["status"] == "complete":
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(5)
    captures = json.loads(capture_path.read_text())
    if captures["status"] != "complete":
        raise RuntimeError("Complete the separate validation captures first.")
    settings = json.loads((source/"config.json").read_text())["patchcore"]
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use the available CUDA device for the requested evaluation.")
    started = time.perf_counter()
    model = PatchcoreModel(layers=settings["layers"], backbone=settings["backbone"],
                          pre_trained=False, num_neighbors=settings["num_neighbors"])
    checkpoint = torch.load(source/"patchcore_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)
    validation = load_rows(out/"validation_dataset.csv")
    with torch.inference_mode():
        for batch in chunks(validation, settings["batch_size"]):
            result = model(images_tensor(out, batch, settings["image_size"], device))
            for row, score in zip(batch, result.pred_score.cpu().numpy().ravel()):
                row["score"] = float(score)
    positives = [row for row in validation if row["defect_ids"]]
    # Read healthy calibration scores for selection; test scores are processed
    # after the threshold has been written and frozen.
    healthy = []
    with (source/"predictions.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["case"] == "healthy_calibration":
                healthy.append({**row, "score": float(row["score"]),
                                "index": int(row["index"]), "angle_deg": float(row["angle_deg"])})
    labeled = [{"case": row["case"], "index": row["index"], "angle_deg": row["angle_deg"],
                "score": row["score"], "label": label, "defect_ids": row["defect_ids"]}
               for label, records in ((0, healthy), (1, positives)) for row in records]
    if not healthy or not positives:
        raise ValueError("Healthy and labeled damaged validation frames are required.")
    metric = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    # The installed metric extracts these two fields from its batch argument.
    metric.update(SimpleNamespace(pred_score=torch.tensor([row["score"] for row in labeled], dtype=torch.float32),
                                  gt_label=torch.tensor([row["label"] for row in labeled], dtype=torch.int64)))
    threshold = float(metric.compute())
    tp = sum(row["score"] > threshold for row in positives)
    fp = sum(row["score"] > threshold for row in healthy)
    fn, tn = len(positives)-tp, len(healthy)-fp
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/len(positives)
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    calibration_metrics = {"positive_passage_frames": len(positives), "healthy_frames": len(healthy),
        "excluded_validation_gear_frames": len(validation)-len(positives), "tp_frames": tp, "fn_frames": fn,
        "fp_frames": fp, "tn_frames": tn, "precision": precision, "recall": recall, "f1": f1,
        "healthy_false_positive_rate_percent": 100*fp/len(healthy)}
    write_json(out/"threshold.json", {"threshold": threshold, "implementation": "anomalib.metrics.F1AdaptiveThreshold",
        "policy": config["threshold_policy"], "comparison": "score > threshold",
        "validation": calibration_metrics, "frozen_before_test_evaluation": True,
        "healthy_source": str((source/"predictions.csv").relative_to(PROJECT)),
        "damaged_source": str((out/"validation_dataset.csv").relative_to(PROJECT))})
    write_csv(out/"validation_labeled_scores.csv", labeled)
    for row in validation:
        row.update(threshold=threshold, predicted_anomaly=row["score"] > threshold,
                   used_for_threshold=bool(row["defect_ids"]))
    write_csv(out/"validation_predictions.csv", validation)
    validation_defects = json.loads((out/"defects.json").read_text())
    validation_rows = []
    for defect in validation_defects["gears"][0]["defects"]:
        selected = [row for row in positives if defect["id"] in row["defect_ids"].split(";")]
        peak = max(selected, key=lambda row: row["score"])
        validation_rows.append({"defect": defect["id"], "kind": defect["kind"],
            "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
            "passage_frames": len(selected), "max_score": peak["score"], "threshold": threshold,
            "detected": peak["score"] > threshold, "peak_angle_deg": peak["angle_deg"]})
    write_csv(out/"validation_defects.csv", validation_rows)
    print(f"Anomalib threshold frozen: {threshold:.8f}; validation image F1 {f1:.4f}", flush=True)

    old_threshold = json.loads((source/"threshold.json").read_text())["threshold"]
    predictions = []
    for row in load_rows(source/"predictions.csv"):
        if row["case"] == "healthy_calibration":
            continue
        row.update(threshold=threshold, predicted_anomaly=row["score"] > threshold,
                   baseline_predicted_anomaly=row["score"] > old_threshold)
        row["image"] = str(Path("..")/source.name/row["image"])
        row["raw"] = str(Path("..")/source.name/row["raw"])
        predictions.append(row)
    defects = copy.deepcopy(json.loads((source/"defects.json").read_text()))
    defect_rows, gear_results = [], []
    for gear in defects["gears"]:
        local_tp = 0
        for defect in gear["defects"]:
            selected = [row for row in predictions if row["case"] == gear["id"] and defect["id"] in row["defect_ids"].split(";")]
            peak = max(selected, key=lambda row: row["score"])
            detected = peak["score"] > threshold
            local_tp += detected
            defect.update(detected=detected, max_score=peak["score"], peak_frame_index=peak["index"],
                          peak_angle_deg=peak["angle_deg"], passage_frames=len(selected))
            defect_rows.append({"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"],
                "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                "passage_frames": len(selected), "max_score": peak["score"], "threshold": threshold,
                "detected": detected, "baseline_detected": peak["score"] > old_threshold,
                "peak_angle_deg": peak["angle_deg"]})
        gear_results.append({"gear": gear["id"], "defects": len(gear["defects"]), "tp": local_tp,
                             "fn": len(gear["defects"])-local_tp, "detection_rate_percent": 100*local_tp/len(gear["defects"])})
    healthy_test = [row for row in predictions if row["case"] == "healthy_test"]
    test_tp, total = sum(row["tp"] for row in gear_results), len(defect_rows)
    test_fp = sum(row["predicted_anomaly"] for row in healthy_test)
    metrics = {"threshold": threshold, "threshold_implementation": "anomalib.metrics.F1AdaptiveThreshold",
        "defects": total, "tp": test_tp, "fn": total-test_tp, "detection_rate_percent": 100*test_tp/total,
        "healthy_test_frames": len(healthy_test), "fp": test_fp, "tn": len(healthy_test)-test_fp,
        "false_positive_rate_percent": 100*test_fp/len(healthy_test), "per_gear": gear_results,
        "validation": calibration_metrics, "validation_defects_detected": sum(row["detected"] for row in validation_rows),
        "validation_defects": len(validation_rows), "baseline": json.loads((source/"metrics.json").read_text()),
        "newly_detected": [row["defect"] for row in defect_rows if row["detected"] and not row["baseline_detected"]],
        "newly_missed": [row["defect"] for row in defect_rows if row["baseline_detected"] and not row["detected"]],
        "evaluation_unit": "Detection/FN per physical defect; FP per held-out healthy frame"}
    write_json(out/"metrics.json", metrics)
    write_json(out/"defect_predictions.json", defects)
    write_csv(out/"defect_predictions.csv", defect_rows)
    write_csv(out/"test_predictions.csv", predictions)
    figures(out, predictions, defects, threshold)
    comparison_figures(out, healthy, positives, defect_rows, old_threshold, threshold)
    sources = [config_path, Path(__file__), PROJECT/"collector/prototype_patchcore.py", capture_path,
               out/"defects.json", out/"validation_dataset.csv", source/"patchcore_model.pt",
               source/"predictions.csv", source/"threshold.json", source/"histogram_mapping.npz"]
    write_json(out/"recalibration_manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "original_bank_shape": list(model.memory_bank.shape),
        "cnn_and_memory_fixed": True, "angular_weighting": False, "phase_dispatch": False,
        "threshold_selector": "anomalib.metrics.F1AdaptiveThreshold", "false_alarm_cap": None,
        "versions": {package: importlib.metadata.version(package) for package in ("torch", "anomalib", "timm")},
        "elapsed_seconds": time.perf_counter()-started,
        "evaluation_context": "Development comparison on the previously evaluated fifteen test defects",
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("validation_patchcore.json"))
    parser.add_argument("--wait-for-captures", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_captures)
