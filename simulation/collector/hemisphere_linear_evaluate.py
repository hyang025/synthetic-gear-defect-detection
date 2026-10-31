"""Fit matched healthy display-input memory and evaluate one spherical-defect gear."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace

from prototype_patchcore import PROJECT, chunks, figures, images_tensor, write_csv, write_json
from validation_threshold import digest, load_rows
from hemisphere_threshold import frame_metrics, score_images
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from anomalib.metrics import F1AdaptiveThreshold
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def display_comparison(out, spherical, predictions, defects, display_label):
    gear = defects["gears"][0]
    fig, axs = plt.subplots(2, 5, figsize=(14, 6), layout="constrained")
    for col, defect in enumerate(gear["defects"]):
        row = next(row for row in predictions if row["case"] == gear["id"] and row["index"] == defect["peak_frame_index"])
        for image_row, (base, label) in enumerate(((spherical, "Histogram equalized"), (out, display_label))):
            path = base/"frames"/gear["id"]/f"frame_{row['index']:03d}.png"
            axs[image_row, col].imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
            axs[image_row, col].set_title(f"{defect['id']} · {label}\n{row['angle_deg']:.2f}°", fontsize=9)
            axs[image_row, col].axis("off")
    fig.suptitle("Same captures · Ø0.10 mm spherical cavities")
    fig.savefig(out/"images/input_comparison.png", dpi=160)
    plt.close(fig)


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    source, spherical, out = (PROJECT/config[key] for key in ("source_run", "spherical_run", "output"))
    display_label = config.get("display_label", "Shared linear gain")
    mapping_name = "display_mapping.json" if "transform" in config else "linear_mapping.json"
    if (out/"evaluation_manifest.json").exists() or (out/"threshold.json").exists():
        raise FileExistsError("Preserve the completed display-input evaluation before a new run.")
    prepare_path = out/"preprocessing_manifest.json"
    if wait:
        print(f"Waiting for {display_label} preprocessing...", flush=True)
        while True:
            try:
                if json.loads(prepare_path.read_text())["status"] == "complete":
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(5)
    if json.loads(prepare_path.read_text())["status"] != "complete":
        raise RuntimeError("Complete the configured display preprocessing first.")
    (out/"images").mkdir(exist_ok=True)
    settings = json.loads((source/"config.json").read_text())["patchcore"]
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use the available GPU for the requested comparison.")
    started = time.perf_counter()
    model = PatchcoreModel(layers=settings["layers"], backbone=settings["backbone"],
        pre_trained=False, num_neighbors=settings["num_neighbors"])
    checkpoint = torch.load(source/"patchcore_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = model.to(device).requires_grad_(False)
    model.memory_bank = torch.empty(0, device=device)
    rows = load_rows(out/"dataset.csv")
    training = [row for row in rows if row["case"] == "healthy_train"]
    if any(row["defect_ids"] for row in training):
        raise RuntimeError("Fit the memory bank using healthy training captures.")
    model.train()
    model.feature_extractor.eval()
    with torch.no_grad():
        for batch in chunks(training, settings["batch_size"]):
            model(images_tensor(out, batch, settings["image_size"], device))
    model.subsample_embedding(settings["coreset_ratio"])
    model.eval()
    fit_seconds = time.perf_counter()-started
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "settings": settings}, out/"patchcore_model.pt")
    print(f"{display_label} healthy bank: {list(model.memory_bank.shape)} · {fit_seconds:.2f} s", flush=True)
    healthy = score_images(model, out, [row for row in rows if row["case"] == "healthy_calibration"], settings, device)
    validation = score_images(model, out, [row for row in rows if row["case"] == config["validation_gear"]], settings, device)
    excluded_ids = set(config.get("exclude_validation_defects", []))
    excluded = [row for row in validation if excluded_ids.intersection(row["defect_ids"].split(";"))]
    positives = [row for row in validation if row["defect_ids"]
                 and not excluded_ids.intersection(row["defect_ids"].split(";"))]
    labeled = [{"case": row["case"], "index": row["index"], "angle_deg": row["angle_deg"], "score": row["score"],
                "label": label, "defect_ids": row["defect_ids"]}
               for label, records in ((0, healthy), (1, positives)) for row in records]
    metric = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    metric.update(SimpleNamespace(pred_score=torch.tensor([row["score"] for row in labeled], dtype=torch.float32),
        gt_label=torch.tensor([row["label"] for row in labeled], dtype=torch.int64)))
    threshold = float(metric.compute())
    percentile = float(np.quantile([row["score"] for row in healthy], settings["healthy_threshold_quantile"]))
    calibration = frame_metrics(healthy, positives, threshold)
    calibration["excluded_validation_gear_frames"] = len(validation)-len(positives)
    calibration["excluded_defect_ids"] = sorted(excluded_ids)
    calibration["excluded_defect_passage_frames"] = len(excluded)
    write_json(out/"threshold.json", {"threshold": threshold, "implementation": "anomalib.metrics.F1AdaptiveThreshold",
        "policy": config["threshold_policy"], "comparison": "score > threshold", "validation": calibration,
        "selection_curve_inclusive": frame_metrics(healthy, positives, threshold, inclusive=True),
        "healthy_percentile_threshold": percentile, "frozen_before_test_evaluation": True})
    write_csv(out/"validation_labeled_scores.csv", labeled)
    if excluded:
        write_csv(out/"validation_excluded_scores.csv", [{"case": row["case"], "index": row["index"],
            "angle_deg": row["angle_deg"], "score": row["score"], "defect_ids": row["defect_ids"],
            "reason": "Excluded validation defect passage"} for row in excluded])
    print(f"{display_label} threshold frozen: {threshold:.8f} · validation F1 {calibration['f1']:.4f}", flush=True)

    healthy_test = score_images(model, out, [row for row in rows if row["case"] == "healthy_test"], settings, device)
    test = score_images(model, out, [row for row in rows if row["case"] == config["test_gear"]], settings, device)
    predictions = healthy+validation+healthy_test+test
    for row in predictions:
        row.update(threshold=threshold, predicted_anomaly=row["score"] > threshold)
    all_defects = json.loads((out/"defects.json").read_text())
    defects = {**copy.deepcopy(all_defects), "gears": [copy.deepcopy(gear) for gear in all_defects["gears"] if gear["id"] == config["test_gear"]]}
    defect_rows = []
    for defect in defects["gears"][0]["defects"]:
        selected = [row for row in test if defect["id"] in row["defect_ids"].split(";")]
        peak = max(selected, key=lambda row: row["score"])
        detected = peak["score"] > threshold
        defect.update(detected=detected, max_score=peak["score"], peak_frame_index=peak["index"],
            peak_angle_deg=peak["angle_deg"], passage_frames=len(selected))
        defect_rows.append({"gear": config["test_gear"], "defect": defect["id"], "kind": defect["kind"],
            "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
            "passage_frames": len(selected), "max_score": peak["score"], "threshold": threshold,
            "detected": detected, "healthy_percentile_detected": peak["score"] > percentile,
            "peak_angle_deg": peak["angle_deg"]})
    tp, total = sum(row["detected"] for row in defect_rows), len(defect_rows)
    fp = sum(row["score"] > threshold for row in healthy_test)
    legacy_tp = sum(row["healthy_percentile_detected"] for row in defect_rows)
    legacy_fp = sum(row["score"] > percentile for row in healthy_test)
    metrics = {"test_gear": config["test_gear"], "threshold": threshold, "threshold_implementation": "anomalib.metrics.F1AdaptiveThreshold",
        "preprocessing": f"{display_label} with shared healthy-derived exposure, rounded to 8-bit",
        "healthy_training_frames": len(training), "healthy_calibration_frames": len(healthy),
        "exclude_validation_defects": sorted(excluded_ids),
        "defects": total, "tp": tp, "fn": total-tp, "detection_rate_percent": 100*tp/total,
        "healthy_test_frames": len(healthy_test), "fp": fp, "tn": len(healthy_test)-fp,
        "false_positive_rate_percent": 100*fp/len(healthy_test), "validation": calibration,
        "healthy_percentile_on_same_geometry": {"threshold": percentile, "tp": legacy_tp, "fn": total-legacy_tp,
            "detection_rate_percent": 100*legacy_tp/total, "fp": legacy_fp,
            "false_positive_rate_percent": 100*legacy_fp/len(healthy_test)},
        "evaluation_unit": "Detection/FN per physical defect; FP per held-out healthy frame"}
    write_json(out/"metrics.json", metrics)
    write_json(out/"defect_predictions.json", defects)
    write_csv(out/"defect_predictions.csv", defect_rows)
    write_csv(out/"predictions.csv", predictions)
    figures(out, predictions, defects, threshold, display_label=display_label)
    display_comparison(out, spherical, predictions, defects, display_label)
    sources = [config_path, Path(__file__), PROJECT/"collector/prototype_patchcore.py", PROJECT/"collector/hemisphere_threshold.py",
        out/mapping_name, prepare_path, out/"dataset.csv", out/"defects.json", source/"patchcore_model.pt"]
    write_json(out/"evaluation_manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "cnn_source": str((source/"patchcore_model.pt").relative_to(PROJECT)),
        "memory_bank_shape": list(model.memory_bank.shape), "healthy_memory_refitted_for_display_inputs": True,
        "display_label": display_label,
        "healthy_training_run": config.get("healthy_training_run"), "exclude_validation_defects": sorted(excluded_ids),
        "angular_weighting": False, "phase_dispatch": False, "false_alarm_cap": None,
        "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter()-started,
        "versions": {package: importlib.metadata.version(package) for package in ("torch", "anomalib", "timm")},
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("hemisphere_linear.json"))
    parser.add_argument("--wait-for-preprocessing", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_preprocessing)
