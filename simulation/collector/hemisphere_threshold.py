"""Evaluate 0.10 mm spherical cavities with the original Anomalib PatchCore bank."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace

from prototype_patchcore import PROJECT, chunks, figures, images_tensor, write_csv, write_json
from validation_threshold import comparison_figures, digest, load_rows
import torch
from anomalib.metrics import F1AdaptiveThreshold
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def score_images(model, out, rows, settings, device):
    with torch.inference_mode():
        for batch in chunks(rows, settings["batch_size"]):
            result = model(images_tensor(out, batch, settings["image_size"], device))
            for row, score in zip(batch, result.pred_score.cpu().numpy().ravel()):
                row["score"] = float(score)
    return rows


def frame_metrics(healthy, positives, threshold, inclusive=False):
    alarm = lambda row: row["score"] >= threshold if inclusive else row["score"] > threshold
    tp, fp = sum(alarm(row) for row in positives), sum(alarm(row) for row in healthy)
    fn, tn = len(positives)-tp, len(healthy)-fp
    return {"positive_passage_frames": len(positives), "healthy_frames": len(healthy),
        "tp_frames": tp, "fn_frames": fn, "fp_frames": fp, "tn_frames": tn,
        "precision": tp/(tp+fp) if tp+fp else 0.0,
        "recall": tp/len(positives), "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        "healthy_false_positive_rate_percent": 100*fp/len(healthy)}


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    source, out = PROJECT/config["source_run"], PROJECT/config["output"]
    if (out/"evaluation_manifest.json").exists() or (out/"threshold.json").exists():
        raise FileExistsError("Preserve the saved spherical-cavity evaluation before a new run.")
    capture_path = out/"capture_manifest.json"
    if wait:
        print("Waiting for spherical-cavity captures...", flush=True)
        while True:
            try:
                if json.loads(capture_path.read_text())["status"] == "complete":
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(5)
    captures = json.loads(capture_path.read_text())
    if captures["status"] != "complete":
        raise RuntimeError("Complete the validation and test captures first.")
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
    validation = score_images(model, out, load_rows(out/"validation_dataset.csv"), settings, device)
    positives = [row for row in validation if row["defect_ids"]]
    healthy = [row for row in load_rows(source/"predictions.csv") if row["case"] == "healthy_calibration"]
    if not positives or not healthy:
        raise ValueError("Healthy and damaged validation labels are required.")
    labeled = [{"case": row["case"], "index": row["index"], "angle_deg": row["angle_deg"],
                "score": row["score"], "label": label, "defect_ids": row["defect_ids"]}
               for label, records in ((0, healthy), (1, positives)) for row in records]
    metric = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    metric.update(SimpleNamespace(pred_score=torch.tensor([row["score"] for row in labeled], dtype=torch.float32),
                                  gt_label=torch.tensor([row["label"] for row in labeled], dtype=torch.int64)))
    threshold = float(metric.compute())
    calibration = frame_metrics(healthy, positives, threshold)
    calibration["excluded_validation_gear_frames"] = len(validation)-len(positives)
    curve_metrics = frame_metrics(healthy, positives, threshold, inclusive=True)
    write_json(out/"threshold.json", {"threshold": threshold, "implementation": "anomalib.metrics.F1AdaptiveThreshold",
        "policy": config["threshold_policy"], "comparison": "score > threshold", "validation": calibration,
        "selection_curve_inclusive": curve_metrics, "frozen_before_test_evaluation": True,
        "healthy_source": str((source/"predictions.csv").relative_to(PROJECT)),
        "damaged_source": str((out/"validation_dataset.csv").relative_to(PROJECT))})
    write_csv(out/"validation_labeled_scores.csv", labeled)
    for row in validation:
        row.update(threshold=threshold, predicted_anomaly=row["score"] > threshold,
                   used_for_threshold=bool(row["defect_ids"]))
    write_csv(out/"validation_predictions.csv", validation)
    all_defects = json.loads((out/"defects.json").read_text())
    validation_rows = []
    for gear in all_defects["gears"]:
        if gear["role"] != "validation":
            continue
        for defect in gear["defects"]:
            selected = [row for row in positives if defect["id"] in row["defect_ids"].split(";")]
            peak = max(selected, key=lambda row: row["score"])
            validation_rows.append({"defect": defect["id"], "kind": defect["kind"],
                "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                "passage_frames": len(selected), "max_score": peak["score"], "threshold": threshold,
                "detected": peak["score"] > threshold, "peak_angle_deg": peak["angle_deg"]})
    write_csv(out/"validation_defects.csv", validation_rows)
    print(f"Anomalib threshold frozen: {threshold:.8f}; applied validation F1 {calibration['f1']:.4f}", flush=True)

    # Apply the frozen validation threshold to the newly captured test gears.
    test = score_images(model, out, load_rows(out/"test_dataset.csv"), settings, device)
    healthy_test = [row for row in load_rows(source/"predictions.csv") if row["case"] == "healthy_test"]
    for row in healthy_test:
        row["role"] = "healthy_test"
        for key in ("image", "raw"):
            row[key] = str(Path("..")/source.name/row[key])
    old_threshold = json.loads((source/"threshold.json").read_text())["threshold"]
    fields = ("case", "role", "index", "angle_deg", "raw", "image", "defect_ids", "first_seed", "raw_source", "score")
    predictions = [{key: row[key] for key in fields} for row in healthy_test+test]
    for row in predictions:
        row.update(threshold=threshold, predicted_anomaly=row["score"] > threshold,
                   healthy_percentile_alarm=row["score"] > old_threshold)
    defects = {**copy.deepcopy(all_defects), "gears": [copy.deepcopy(gear) for gear in all_defects["gears"] if gear["role"] == "test"]}
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
            defect_rows.append({"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"], "side": defect["side"],
                "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                "passage_frames": len(selected), "max_score": peak["score"], "threshold": threshold,
                "detected": detected, "healthy_percentile_detected": peak["score"] > old_threshold,
                "peak_angle_deg": peak["angle_deg"]})
        gear_results.append({"gear": gear["id"], "defects": len(gear["defects"]), "tp": local_tp,
            "fn": len(gear["defects"])-local_tp, "detection_rate_percent": 100*local_tp/len(gear["defects"])})
    test_tp, total = sum(row["tp"] for row in gear_results), len(defect_rows)
    test_fp = sum(row["score"] > threshold for row in healthy_test)
    legacy_tp = sum(row["healthy_percentile_detected"] for row in defect_rows)
    legacy_fp = sum(row["score"] > old_threshold for row in healthy_test)
    kind_counts = []
    for kind in sorted({row["kind"] for row in defect_rows}):
        selected = [row for row in defect_rows if row["kind"] == kind]
        tp = sum(row["detected"] for row in selected)
        kind_counts.append({"kind": kind, "defects": len(selected), "tp": tp, "fn": len(selected)-tp})
    metrics = {"threshold": threshold, "threshold_implementation": "anomalib.metrics.F1AdaptiveThreshold",
        "diameter_mm": 0.10, "depth_um": 50, "defects": total, "tp": test_tp, "fn": total-test_tp,
        "detection_rate_percent": 100*test_tp/total, "healthy_test_frames": len(healthy_test),
        "fp": test_fp, "tn": len(healthy_test)-test_fp, "false_positive_rate_percent": 100*test_fp/len(healthy_test),
        "per_gear": gear_results, "per_location": kind_counts, "validation": calibration,
        "validation_defects_detected": sum(row["detected"] for row in validation_rows),
        "validation_defects": len(validation_rows),
        "healthy_percentile_on_same_geometry": {"threshold": old_threshold, "tp": legacy_tp, "fn": total-legacy_tp,
            "detection_rate_percent": 100*legacy_tp/total, "fp": legacy_fp,
            "false_positive_rate_percent": 100*legacy_fp/len(healthy_test)},
        "evaluation_unit": "Detection/FN per physical defect; FP per held-out healthy frame"}
    write_json(out/"metrics.json", metrics)
    write_json(out/"defect_predictions.json", defects)
    write_csv(out/"defect_predictions.csv", defect_rows)
    write_csv(out/"test_predictions.csv", predictions)
    figures(out, predictions, defects, threshold)
    comparison_figures(out, healthy, positives, defect_rows, old_threshold, threshold)
    sources = [config_path, Path(__file__), PROJECT/"collector/prototype_patchcore.py", PROJECT/"collector/validation_threshold.py",
        capture_path, out/"defects.json", out/"validation_dataset.csv", out/"test_dataset.csv",
        source/"patchcore_model.pt", source/"predictions.csv", source/"threshold.json", source/"histogram_mapping.npz"]
    write_json(out/"evaluation_manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "original_bank_shape": list(model.memory_bank.shape), "cnn_and_memory_fixed": True,
        "angular_weighting": False, "phase_dispatch": False, "threshold_selector": "anomalib.metrics.F1AdaptiveThreshold",
        "false_alarm_cap": None, "elapsed_seconds": time.perf_counter()-started,
        "versions": {package: importlib.metadata.version(package) for package in ("torch", "anomalib", "timm")},
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("hemisphere.json"))
    parser.add_argument("--wait-for-captures", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_captures)
