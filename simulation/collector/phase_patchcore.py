"""Fit and evaluate six encoder-dispatched PatchCore banks on saved captures."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from prototype_patchcore import PROJECT, chunks, images_tensor, write_csv, write_json
import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_fraction(angle, tooth_count, origin):
    phase = ((np.asarray(angle)-origin)*tooth_count/360) % 1
    return np.where(abs(phase-1) < 1e-10, 0, phase)


def dispatch(phase, count):
    return (np.floor(np.asarray(phase)*count+1e-10).astype(int) % count)


def figures(out, predictions, baseline, defects, bank_rows, threshold_rows, pitch):
    def selected(data, case):
        return sorted([row for row in data if row["case"] == case], key=lambda row: row["angle_deg"])

    baseline_healthy = selected(baseline, "healthy_test")
    healthy = selected(predictions, "healthy_test")
    fig, axs = plt.subplots(3, 2, figsize=(15, 10), layout="constrained")
    for row, gear in enumerate(defects["gears"]):
        for col, data in enumerate((baseline, predictions)):
            reference = baseline_healthy if col == 0 else healthy
            damaged = selected(data, gear["id"])
            ax = axs[row, col]
            ax.plot([p["angle_deg"] for p in reference], [p["score_ratio"] for p in reference],
                    color="#247ba0", lw=0.8, label="Healthy test")
            ax.plot([p["angle_deg"] for p in damaged], [p["score_ratio"] for p in damaged],
                    color="#d1495b", lw=0.9, label="Damaged")
            ax.axhline(1, color="black", ls="--", lw=0.8)
            for defect in gear["defects"]:
                ax.axvline(defect["crossing_angle_deg"], color="gray", ls=":", lw=0.7)
            method = "Whole-revolution bank" if col == 0 else "Six phase banks"
            ax.set(title=f"{gear['id']} · {method}", xlabel="Encoder angle (°)",
                   ylabel="Score / dispatched threshold", xlim=(0, 360))
            ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=9)
    fig.suptitle("PatchCore · full rotation · dotted defect crossings")
    fig.savefig(out/"images/rotation_comparison.png", dpi=150)
    plt.close(fig)

    rows = [defect for gear in defects["gears"] for defect in gear["defects"]]
    locations = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    ax.bar(locations-0.18, [row["baseline_peak_score_ratio"] for row in rows],
           width=0.36, color="#247ba0", label="Whole-revolution bank")
    ax.bar(locations+0.18, [row["peak_score_ratio"] for row in rows],
           width=0.36, color="#d1495b", label="Six phase banks")
    ax.axhline(1, color="black", ls="--", lw=0.8)
    ax.set(title="Fifteen defects · peak passage score", ylabel="Score / threshold",
           xticks=locations, xticklabels=[row["id"] for row in rows], xlabel="Defect")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out/"images/defect_comparison.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    for bank in bank_rows:
        b = bank["bank_id"]
        lower, upper = bank["dispatch_start_deg"], bank["dispatch_stop_deg"]
        axs[0].broken_barh([(lower, upper-lower)], (b-0.3, 0.6),
                          facecolors=plt.colormaps["tab10"](b))
    axs[0].set(title="Encoder dispatch · one tooth-and-groove cycle", xlabel="Phase angle within cycle (°)",
               ylabel="Bank ID", yticks=np.arange(len(bank_rows)), xlim=(0, pitch))
    for bank in bank_rows:
        b = bank["bank_id"]
        healthy = [p for p in predictions if p["case"] == "healthy_test" and p["bank_id"] == b]
        axs[1].scatter(np.full(len(healthy), b), [p["score_ratio"] for p in healthy],
                       s=12, alpha=0.65, color=plt.colormaps["tab10"](b))
    axs[1].axhline(1, color="black", ls="--", lw=0.8)
    axs[1].set(title="Healthy test · dispatched scores", xlabel="Bank ID",
               ylabel="Score / bank threshold", xticks=np.arange(len(bank_rows)))
    for ax in axs:
        ax.grid(alpha=0.2)
    fig.savefig(out/"images/bank_dispatch.png", dpi=150)
    plt.close(fig)


def main(config_path):
    config = json.loads(config_path.read_text())
    source, out = PROJECT/config["source_run"], PROJECT/config["output"]
    if out.exists():
        raise FileExistsError(f"Preserve the saved run at {out} before a new evaluation.")
    source_manifest = json.loads((source/"patchcore_manifest.json").read_text())
    capture_manifest = json.loads((source/"manifest.json").read_text())
    if source_manifest["status"] != "complete" or capture_manifest["status"] != "captures_complete":
        raise RuntimeError("Complete the baseline captures and evaluation first.")
    settings = json.loads((source/"config.json").read_text())["patchcore"]
    gear_parameters = json.loads((PROJECT/config["gear_parameters"]).read_text())
    tooth_count = gear_parameters["supplied"]["tooth_count"]
    pitch = 360/tooth_count
    count = config["bank_count"]
    width = pitch/count
    overlap = config["training_overlap_deg"]
    if count < 2 or overlap < 0 or overlap >= width/2:
        raise ValueError("Use positive bank widths with a smaller nonnegative overlap.")
    with (source/"dataset.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["index"], row["angle_deg"] = int(row["index"]), float(row["angle_deg"])
        phase = float(phase_fraction(row["angle_deg"], tooth_count, config["phase_origin_deg"]))
        row.update(phase_fraction=phase, phase_angle_deg=phase*pitch,
                   bank_id=int(dispatch(phase, count)))
    training = [row for row in rows if row["case"] == "healthy_train"]
    if any(row["defect_ids"] for row in training):
        raise ValueError("Healthy training captures are required for the phase banks.")
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use the available CUDA device for this requested evaluation.")
    out.mkdir()
    (out/"images").mkdir()
    effective = {**config, "tooth_count": tooth_count, "tooth_pitch_deg": pitch,
                 "bank_width_deg": width, "patchcore": settings,
                 "feature_weights_source": str((source/"patchcore_model.pt").relative_to(PROJECT))}
    write_json(out/"config.json", effective)
    started = time.perf_counter()
    # Restore the exact frozen backbone and buffers from the baseline checkpoint.
    # Constructor initialization is replaced by the saved pretrained state.
    model = PatchcoreModel(layers=settings["layers"], backbone=settings["backbone"],
                          pre_trained=False, num_neighbors=settings["num_neighbors"])
    checkpoint = torch.load(source/"patchcore_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = model.to(device)
    model.requires_grad_(False)
    model.train()
    model.feature_extractor.eval()
    embeddings = []
    with torch.no_grad():
        for index, batch in enumerate(chunks(training, settings["batch_size"])):
            extracted = model(images_tensor(source, batch, settings["image_size"], device))
            embeddings.append(extracted.reshape(len(batch), -1, extracted.shape[-1]))
            model.embedding_store.clear()
            if index % 10 == 0:
                print(f"Shared healthy embedding batches: {index+1}", flush=True)
    training_features = torch.cat(embeddings)
    del embeddings
    phases = np.asarray([row["phase_fraction"] for row in training])
    banks, bank_rows = {}, []
    for b in range(count):
        center = (b+0.5)/count
        distance = abs((phases-center+0.5) % 1-0.5)
        chosen = np.flatnonzero(distance <= (width/2+overlap)/pitch+1e-10)
        indices = torch.as_tensor(chosen, device=device)
        candidates = training_features[indices].reshape(-1, training_features.shape[-1])
        model.embedding_store = [candidates]
        torch.manual_seed(settings["seed"])
        model.subsample_embedding(settings["coreset_ratio"])
        banks[b] = model.memory_bank
        bank_rows.append({"bank_id": b, "dispatch_start_deg": b*width, "dispatch_stop_deg": (b+1)*width,
                          "training_overlap_deg": overlap, "healthy_training_frames": len(chosen),
                          "candidate_patch_features": len(candidates), "memory_patch_features": len(banks[b]),
                          "feature_dimension": banks[b].shape[1],
                          "training_frame_indices": [training[i]["index"] for i in chosen]})
        print(f"Bank {b}: {len(chosen)} healthy frames → {len(banks[b])} memory features", flush=True)
    del training_features, candidates, indices
    model.embedding_store.clear()
    model.eval()
    fit_seconds = time.perf_counter()-started
    shared_state = {key: value.detach().cpu() for key, value in model.state_dict().items() if key != "memory_bank"}
    torch.save({"shared_state_dict": shared_state, "banks": {b: bank.detach().cpu() for b, bank in banks.items()},
                "config": effective}, out/"phase_banks.pt")

    def infer(case):
        selected = sorted([row for row in rows if row["case"] == case], key=lambda row: row["index"])
        case_predictions = []
        maps = np.empty((len(selected), 1, settings["image_size"], settings["image_size"]), dtype=np.float32)
        positions = {row["index"]: position for position, row in enumerate(selected)}
        for b in range(count):
            model.memory_bank = banks[b]
            group = [row for row in selected if row["bank_id"] == b]
            with torch.inference_mode():
                for batch in chunks(group, settings["batch_size"]):
                    result = model(images_tensor(source, batch, settings["image_size"], device))
                    scores = result.pred_score.cpu().numpy().ravel()
                    heatmaps = result.anomaly_map.cpu().numpy()
                    for row, score, heatmap in zip(batch, scores, heatmaps):
                        prediction = {**row, "score": float(score)}
                        prediction["image"] = os.path.relpath(source/row["image"], out)
                        prediction["raw"] = os.path.relpath(source/row["raw"], out)
                        case_predictions.append(prediction)
                        maps[positions[row["index"]]] = heatmap
        print(f"Evaluated {case}: {len(selected)} frames", flush=True)
        return sorted(case_predictions, key=lambda row: row["index"]), maps

    predictions, all_maps = [], {}
    calibration, all_maps["healthy_calibration"] = infer("healthy_calibration")
    thresholds, threshold_rows = {}, []
    for b in range(count):
        scores = [row["score"] for row in calibration if row["bank_id"] == b]
        if len(scores) < 2:
            raise ValueError(f"Healthy calibration coverage is required for bank {b}.")
        thresholds[b] = float(np.quantile(scores, settings["healthy_threshold_quantile"]))
        threshold_rows.append({"bank_id": b, "threshold": thresholds[b], "healthy_calibration_frames": len(scores),
                               "quantile": settings["healthy_threshold_quantile"]})
    write_json(out/"thresholds.json", {"banks": threshold_rows, "source": "healthy_calibration",
                                       "frozen_before_test_evaluation": True})
    predictions.extend(calibration)
    for case in capture_manifest["cases"]:
        if case in ("healthy_train", "healthy_calibration"):
            continue
        case_predictions, all_maps[case] = infer(case)
        predictions.extend(case_predictions)
    for row in predictions:
        row.update(threshold=thresholds[row["bank_id"]],
                   score_ratio=row["score"]/thresholds[row["bank_id"]],
                   predicted_anomaly=row["score"] > thresholds[row["bank_id"]])
    with (source/"predictions.csv").open() as stream:
        baseline = list(csv.DictReader(stream))
    baseline_threshold = json.loads((source/"threshold.json").read_text())["threshold"]
    for row in baseline:
        row["index"], row["angle_deg"], row["score"] = int(row["index"]), float(row["angle_deg"]), float(row["score"])
        row["score_ratio"] = row["score"]/baseline_threshold
    baseline_lookup = {(row["case"], row["index"]): row for row in baseline}
    for row in predictions:
        previous = baseline_lookup[row["case"], row["index"]]
        row.update(baseline_score=previous["score"], baseline_score_ratio=previous["score_ratio"],
                   baseline_predicted_anomaly=previous["score_ratio"] > 1)
    baseline_defects = json.loads((source/"defect_predictions.json").read_text())
    baseline_defect_lookup = {d["id"]: d for g in baseline_defects["gears"] for d in g["defects"]}
    defects = copy.deepcopy(json.loads((source/"defects.json").read_text()))
    defect_rows, gear_results = [], []
    for gear in defects["gears"]:
        tp = 0
        for defect in gear["defects"]:
            selected = [row for row in predictions if row["case"] == gear["id"] and defect["id"] in row["defect_ids"].split(";")]
            if not selected:
                raise ValueError(f"Passage captures are required for {defect['id']}.")
            peak = max(selected, key=lambda row: row["score_ratio"])
            previous = baseline_defect_lookup[defect["id"]]
            detected = peak["predicted_anomaly"]
            tp += detected
            defect.update(detected=detected, max_score=peak["score"], peak_score_ratio=peak["score_ratio"],
                          peak_frame_index=peak["index"], peak_angle_deg=peak["angle_deg"],
                          peak_bank_id=peak["bank_id"], threshold=peak["threshold"], passage_frames=len(selected),
                          baseline_detected=previous["detected"],
                          baseline_peak_score_ratio=previous["max_score"]/baseline_threshold)
            defect_rows.append({"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"],
                "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                "crossing_angle_deg": defect["crossing_angle_deg"], "passage_frames": len(selected),
                "peak_angle_deg": peak["angle_deg"], "peak_bank_id": peak["bank_id"], "max_score": peak["score"],
                "threshold": peak["threshold"], "peak_score_ratio": peak["score_ratio"], "detected": detected,
                "baseline_detected": previous["detected"], "baseline_peak_score_ratio": defect["baseline_peak_score_ratio"]})
        gear_results.append({"gear": gear["id"], "defects": len(gear["defects"]), "tp": tp,
                             "fn": len(gear["defects"])-tp, "detection_rate_percent": 100*tp/len(gear["defects"])})
    healthy = [row for row in predictions if row["case"] == "healthy_test"]
    fp = sum(row["predicted_anomaly"] for row in healthy)
    per_bank = []
    for b in range(count):
        selected = [row for row in healthy if row["bank_id"] == b]
        local_fp = sum(row["predicted_anomaly"] for row in selected)
        per_bank.append({"bank_id": b, "healthy_test_frames": len(selected), "fp": local_fp,
                         "tn": len(selected)-local_fp, "false_positive_rate_percent": 100*local_fp/len(selected)})
    tp, total = sum(row["tp"] for row in gear_results), len(defect_rows)
    metrics = {"healthy_training_frames": len(training), "healthy_calibration_frames": len(calibration),
               "defects": total, "tp": tp, "fn": total-tp, "detection_rate_percent": 100*tp/total,
               "healthy_test_frames": len(healthy), "fp": fp, "tn": len(healthy)-fp,
               "false_positive_rate_percent": 100*fp/len(healthy), "per_gear": gear_results, "per_bank": per_bank,
               "threshold_selection": config["threshold_selection"], "baseline": json.loads((source/"metrics.json").read_text()),
               "newly_detected": [row["defect"] for row in defect_rows if row["detected"] and not row["baseline_detected"]],
               "newly_missed": [row["defect"] for row in defect_rows if row["baseline_detected"] and not row["detected"]],
               "evaluation_unit": "Detection/FN per physical defect; FP per held-out healthy frame"}
    write_json(out/"metrics.json", metrics)
    write_json(out/"defect_predictions.json", defects)
    write_json(out/"banks.json", {"banks": bank_rows, "memory_features_total": sum(len(bank) for bank in banks.values())})
    write_csv(out/"predictions.csv", predictions)
    write_csv(out/"defect_predictions.csv", defect_rows)
    write_csv(out/"bank_metrics.csv", per_bank)
    np.savez_compressed(out/"anomaly_maps.npz", **all_maps)
    figures(out, predictions, baseline, defects, bank_rows, threshold_rows, pitch)
    sources = [config_path, Path(__file__), Path(__file__).with_name("prototype_patchcore.py"),
               PROJECT/config["gear_parameters"], source/"config.json", source/"manifest.json",
               source/"dataset.csv", source/"histogram_mapping.npz", source/"patchcore_model.pt",
               source/"predictions.csv", source/"metrics.json", source/"defects.json"]
    write_json(out/"manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": effective, "device": str(device), "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter()-started, "new_ray_tracing": False,
        "angular_weighting": False, "frozen_backbone_restored_from_baseline": True,
        "evaluation_context": "Development comparison on the previously evaluated capture sets",
        "versions": {package: importlib.metadata.version(package) for package in ("torch", "anomalib", "timm")},
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources}})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    main(args.config.resolve())
