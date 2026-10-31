"""Score two new gears with frozen CLAHE and histogram memories and thresholds."""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from prototype_patchcore import PROJECT, figures, write_csv, write_json
from validation_threshold import digest, load_rows
from hemisphere_threshold import score_images
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def main(config_path, wait=False):
    config = json.loads(config_path.read_text())
    root = PROJECT/config["output"]
    if (root/"comparison.json").exists():
        raise FileExistsError("Preserve the completed CLAHE comparison.")
    ready_paths = [root/"clahe/evaluation_manifest.json",
                   root/"clahe/additional_preprocessing_manifest.json",
                   root/"histogram/additional_preprocessing_manifest.json"]
    while True:
        try:
            ready = all(json.loads(path.read_text())["status"] == "complete" for path in ready_paths)
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
        if ready:
            break
        if not wait:
            raise RuntimeError("Complete the CLAHE reference fit and both new test-image datasets.")
        time.sleep(5)
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use the RTX GPU for the requested evaluation.")
    additional_defects = json.loads((root/"defects.json").read_text())
    new_cases = {gear["id"] for gear in additional_defects["gears"]}
    methods = [("histogram", "Histogram equalized", PROJECT/config["histogram_baseline"]),
               ("clahe", "CLAHE · log input", root/"clahe")]
    comparison, defect_comparison, sources = [], [], [config_path, Path(__file__), root/"capture_manifest.json"]
    started = time.perf_counter()
    for method, label, reference in methods:
        folder = root/method
        out = folder/"extended"
        if (out/"metrics.json").exists():
            raise FileExistsError("Preserve the saved extended evaluation.")
        (out/"images").mkdir(parents=True, exist_ok=True)
        baseline = json.loads((reference/"metrics.json").read_text())
        threshold = baseline["threshold"]
        healthy_threshold = baseline["healthy_percentile_on_same_geometry"]["threshold"]
        checkpoint = torch.load(reference/"patchcore_model.pt", map_location="cpu", weights_only=True)
        settings = checkpoint["settings"]
        model = PatchcoreModel(layers=settings["layers"], backbone=settings["backbone"],
            pre_trained=False, num_neighbors=settings["num_neighbors"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        model = model.to(device).eval().requires_grad_(False)
        new_rows = load_rows(folder/"additional_dataset.csv")
        if {row["case"] for row in new_rows} != new_cases:
            raise RuntimeError("Evaluate the two configured new test gears.")
        new_predictions = score_images(model, folder, new_rows, settings, device)
        del model
        torch.cuda.empty_cache()
        old_predictions = [row for row in load_rows(reference/"predictions.csv")
                           if row["case"] in ("healthy_test", config["test_gear"])]
        fields = ("case", "role", "index", "angle_deg", "raw", "image", "defect_ids", "first_seed", "raw_source",
                  "score", "threshold", "predicted_anomaly", "healthy_percentile_alarm")
        predictions = []
        for base, records in ((reference, old_predictions), (folder, new_predictions)):
            for record in records:
                row = {key: record.get(key, "") for key in fields}
                row.update(raw=os.path.relpath(base/record["raw"], out), image=os.path.relpath(base/record["image"], out),
                    threshold=threshold, predicted_anomaly=record["score"] > threshold,
                    healthy_percentile_alarm=record["score"] > healthy_threshold)
                predictions.append(row)
        original_defects = json.loads((reference/"defect_predictions.json").read_text())
        defects = {**copy.deepcopy(original_defects),
                   "gears": copy.deepcopy(original_defects["gears"])+copy.deepcopy(additional_defects["gears"])}
        defect_rows, per_gear = [], []
        for gear in defects["gears"]:
            gear_rows = []
            for defect in gear["defects"]:
                selected = [row for row in predictions if row["case"] == gear["id"]
                            and defect["id"] in row["defect_ids"].split(";")]
                peak = max(selected, key=lambda row: row["score"])
                defect.update(max_score=peak["score"], peak_frame_index=peak["index"], peak_angle_deg=peak["angle_deg"],
                              passage_frames=len(selected), detected=peak["score"] > threshold)
                row = {"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"],
                    "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                    "passage_frames": len(selected), "max_score": peak["score"], "threshold": threshold,
                    "detected": peak["score"] > threshold, "healthy_percentile_detected": peak["score"] > healthy_threshold,
                    "peak_angle_deg": peak["angle_deg"]}
                gear_rows.append(row)
                defect_rows.append(row)
                defect_comparison.append({"method": label, **row})
            tp = sum(row["detected"] for row in gear_rows)
            ht = sum(row["healthy_percentile_detected"] for row in gear_rows)
            per_gear.append({"gear": gear["id"], "defects": len(gear_rows), "tp": tp, "fn": len(gear_rows)-tp,
                             "healthy_percentile_tp": ht, "healthy_percentile_fn": len(gear_rows)-ht})
        healthy = [row for row in predictions if row["case"] == "healthy_test"]
        total = len(defect_rows)
        tp = sum(row["detected"] for row in defect_rows)
        ht = sum(row["healthy_percentile_detected"] for row in defect_rows)
        fp = sum(row["predicted_anomaly"] for row in healthy)
        hfp = sum(row["healthy_percentile_alarm"] for row in healthy)
        metrics = {"method": label, "healthy_training_frames": baseline["healthy_training_frames"],
            "threshold": threshold, "defects": total, "tp": tp, "fn": total-tp,
            "detection_rate_percent": 100*tp/total, "healthy_test_frames": len(healthy), "fp": fp,
            "false_positive_rate_percent": 100*fp/len(healthy), "per_gear": per_gear,
            "healthy_percentile_on_same_geometry": {"threshold": healthy_threshold, "tp": ht, "fn": total-ht,
                "fp": hfp, "false_positive_rate_percent": 100*hfp/len(healthy)},
            "frozen_reference": str(reference.relative_to(PROJECT)), "new_test_gears_used_for_fitting": False,
            "evaluation_unit": "Detection/FN per physical defect; FP per unique healthy test frame"}
        write_json(out/"metrics.json", metrics)
        write_json(out/"defect_predictions.json", defects)
        write_csv(out/"defect_predictions.csv", defect_rows)
        write_csv(out/"predictions.csv", predictions)
        figures(out, predictions, defects, threshold, display_label=label)
        for policy, limit, detected, false_alarms in (("Anomalib F1", threshold, tp, fp),
                ("Healthy 99th percentile", healthy_threshold, ht, hfp)):
            comparison.append({"method": label, "threshold_selection": policy, "threshold": limit,
                "defects": total, "tp": detected, "fn": total-detected,
                "healthy_test_frames": len(healthy), "fp": false_alarms,
                "false_positive_rate_percent": 100*false_alarms/len(healthy)})
        sources.extend([reference/"patchcore_model.pt", reference/"threshold.json", reference/"metrics.json",
                        folder/"additional_dataset.csv", folder/"additional_preprocessing_manifest.json", out/"metrics.json"])
        print(json.dumps(metrics, indent=2), flush=True)
    write_csv(root/"comparison.csv", comparison)
    write_csv(root/"defect_comparison.csv", defect_comparison)
    write_json(root/"comparison.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": comparison, "new_test_gears": sorted(new_cases), "new_defects": 10, "total_test_defects": 15,
        "unique_healthy_test_frames": 440, "shared_raw_captures": True,
        "thresholds_frozen_before_new_test_evaluation": True, "elapsed_seconds": time.perf_counter()-started,
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("clahe.json"))
    parser.add_argument("--wait-for-preprocessing", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_preprocessing)
