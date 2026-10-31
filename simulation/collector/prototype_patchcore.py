"""Healthy-only PatchCore fitting and fixed-threshold defect evaluation."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".tools/matplotlib"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def images_tensor(out, rows, size, device):
    images = np.stack([np.asarray(Image.open(out/row["image"]).convert("L"), dtype=np.float32)/255 for row in rows])
    tensor = torch.from_numpy(images[:, None]).to(device)
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    tensor = tensor.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    return (tensor-mean)/std


def chunks(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start+size]


def figures(out, predictions, defects, threshold, display_label="Histogram-equalized grayscale"):
    (out/"images").mkdir(exist_ok=True)
    healthy = sorted([row for row in predictions if row["case"] == "healthy_test"], key=lambda row: row["angle_deg"])
    count = len(defects["gears"])
    fig, axs = plt.subplots(count, 1, figsize=(13, max(4, 3.3*count)), layout="constrained")
    axs = np.atleast_1d(axs)
    for ax, gear in zip(axs, defects["gears"]):
        rows = sorted([row for row in predictions if row["case"] == gear["id"]], key=lambda row: row["angle_deg"])
        ax.plot([row["angle_deg"] for row in healthy], [row["score"]/threshold for row in healthy],
                color="#247ba0", lw=0.8, label="Healthy test")
        ax.plot([row["angle_deg"] for row in rows], [row["score"]/threshold for row in rows],
                color="#d1495b", lw=0.9, label=gear["id"])
        ax.axhline(1, color="black", ls="--", lw=0.8, label="Threshold")
        for defect in gear["defects"]:
            ax.axvline(defect["crossing_angle_deg"], color="gray", ls=":", lw=0.7)
        ax.set(title=gear["id"], xlabel="Encoder angle (°)", ylabel="Score / threshold", xlim=(0, 360))
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=9)
    fig.suptitle("PatchCore · full rotation · dotted defect crossings")
    fig.savefig(out/"images/rotation_scores.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    scores = [defect["max_score"]/threshold for gear in defects["gears"] for defect in gear["defects"]]
    labels = [defect["id"] for gear in defects["gears"] for defect in gear["defects"]]
    ax.bar(labels, scores, color=["#247ba0" if value > 1 else "#d1495b" for value in scores])
    ax.axhline(1, color="black", ls="--", lw=0.8)
    ax.set(title=f"{len(labels)} defects · peak passage score", ylabel="Score / threshold", xlabel="Defect")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out/"images/defect_detection.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(count, 5, figsize=(14, max(3, 2.7*count)), squeeze=False, layout="constrained")
    for row, gear in enumerate(defects["gears"]):
        for col, defect in enumerate(gear["defects"]):
            prediction = next(p for p in predictions if p["case"] == gear["id"] and p["index"] == defect["peak_frame_index"])
            axs[row, col].imshow(Image.open(out/prediction["image"]), cmap="gray", vmin=0, vmax=255)
            result = "Detected" if defect["detected"] else "Missed"
            axs[row, col].set_title(f"{defect['id']} · {result}\nØ{defect['diameter_mm']:.2f} mm · {defect['depth_mm']*1000:.0f} µm", fontsize=10)
            axs[row, col].axis("off")
    fig.suptitle(f"{display_label} · peak passage captures")
    fig.savefig(out/"images/test_captures.png", dpi=150)
    plt.close(fig)


def main(config_path, wait_for_captures=False):
    config = json.loads(config_path.read_text())
    out = PROJECT/config["output"]
    if (out/"patchcore_manifest.json").exists():
        raise RuntimeError("Preserve the existing PatchCore evaluation before fitting a new model.")
    if wait_for_captures:
        print("Waiting for capture and grayscale preprocessing completion...", flush=True)
        while True:
            try:
                if json.loads((out/"manifest.json").read_text())["status"] == "captures_complete":
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(5)
    manifest = json.loads((out/"manifest.json").read_text())
    if manifest["status"] != "captures_complete":
        raise RuntimeError("Complete the requested captures and grayscale preprocessing first.")
    with (out/"dataset.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["index"], row["angle_deg"] = int(row["index"]), float(row["angle_deg"])
    training = [row for row in rows if row["case"] == "healthy_train"]
    if any(row["defect_ids"] for row in training):
        raise RuntimeError("The memory bank requires healthy training captures.")
    settings = config["patchcore"]
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model = PatchcoreModel(layers=settings["layers"], backbone=settings["backbone"],
                          pre_trained=True, num_neighbors=settings["num_neighbors"]).to(device)
    model.train()
    model.feature_extractor.eval()
    with torch.no_grad():
        for index, batch in enumerate(chunks(training, settings["batch_size"])):
            model(images_tensor(out, batch, settings["image_size"], device))
            if index % 10 == 0:
                print(f"Healthy embedding batches: {index+1}", flush=True)
    model.subsample_embedding(settings["coreset_ratio"])
    model.eval()
    fit_seconds = time.perf_counter()-started
    print(f"Healthy memory bank: {list(model.memory_bank.shape)} | {fit_seconds:.1f} s", flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "settings": settings}, out/"patchcore_model.pt")
    predictions, maps, threshold = [], {}, None
    for case in manifest["cases"]:
        if case == "healthy_train":
            continue
        selected = [row for row in rows if row["case"] == case]
        maps[case] = []
        with torch.inference_mode():
            for batch in chunks(selected, settings["batch_size"]):
                result = model(images_tensor(out, batch, settings["image_size"], device))
                scores = result.pred_score.cpu().numpy().ravel()
                maps[case].extend(result.anomaly_map.cpu().numpy().astype(np.float32))
                for row, score in zip(batch, scores):
                    predictions.append({**row, "score": float(score)})
        print(f"Evaluated {case}: {len(selected)} frames", flush=True)
        if case == "healthy_calibration":
            threshold = float(np.quantile([row["score"] for row in predictions], settings["healthy_threshold_quantile"]))
            write_json(out/"threshold.json", {"threshold": threshold, "quantile": settings["healthy_threshold_quantile"],
                                              "source": "healthy_calibration", "frozen_before_test_evaluation": True})
            print(f"Healthy-calibrated threshold fixed: {threshold:.8f}", flush=True)
        elif threshold is None:
            raise RuntimeError("Calibrate the healthy threshold before test evaluation.")
    calibration = [row["score"] for row in predictions if row["case"] == "healthy_calibration"]
    for prediction in predictions:
        prediction["predicted_anomaly"] = prediction["score"] > threshold
    healthy_test = [row for row in predictions if row["case"] == "healthy_test"]
    fp = sum(row["predicted_anomaly"] for row in healthy_test)
    defects = json.loads((out/"defects.json").read_text())
    defect_rows, gear_results = [], []
    for gear in defects["gears"]:
        tp = 0
        for defect in gear["defects"]:
            selected = [row for row in predictions if row["case"] == gear["id"] and defect["id"] in row["defect_ids"].split(";")]
            if not selected:
                raise RuntimeError(f"Missing passage captures for {defect['id']}.")
            peak = max(selected, key=lambda row: row["score"])
            detected = peak["score"] > threshold
            tp += detected
            defect.update(detected=detected, max_score=peak["score"], peak_frame_index=peak["index"],
                          peak_angle_deg=peak["angle_deg"], passage_frames=len(selected))
            defect_rows.append({"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"], "side": defect["side"],
                                "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                                "crossing_angle_deg": defect["crossing_angle_deg"], "passage_frames": len(selected),
                                "peak_angle_deg": peak["angle_deg"], "max_score": peak["score"], "threshold": threshold,
                                "detected": detected})
        gear_results.append({"gear": gear["id"], "defects": len(gear["defects"]), "tp": tp,
                             "fn": len(gear["defects"])-tp, "detection_rate_percent": 100*tp/len(gear["defects"])})
    tp, total = sum(row["tp"] for row in gear_results), len(defect_rows)
    metrics = {"threshold": threshold, "threshold_selection": "99th percentile of healthy_calibration scores",
               "healthy_training_frames": len(training), "healthy_calibration_frames": len(calibration),
               "defects": total, "tp": tp, "fn": total-tp, "detection_rate_percent": 100*tp/total,
               "healthy_test_frames": len(healthy_test), "fp": fp, "tn": len(healthy_test)-fp,
               "false_positive_rate_percent": 100*fp/len(healthy_test), "per_gear": gear_results,
               "evaluation_unit": "Detection/FN per physical defect; FP per held-out healthy frame"}
    write_json(out/"metrics.json", metrics)
    write_json(out/"defect_predictions.json", defects)
    write_csv(out/"predictions.csv", predictions)
    write_csv(out/"defect_predictions.csv", defect_rows)
    np.savez_compressed(out/"anomaly_maps.npz", **{case: np.stack(value) for case, value in maps.items()})
    figures(out, predictions, defects, threshold)
    versions = {package: importlib.metadata.version(package) for package in ("torch", "anomalib", "timm")}
    write_json(out/"patchcore_manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                                              "versions": versions, "device": str(device), "settings": settings,
                                              "memory_bank_shape": list(model.memory_bank.shape), "fit_seconds": fit_seconds,
                                              "total_seconds": time.perf_counter()-started,
                                              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("prototype.json"))
    parser.add_argument("--wait-for-captures", action="store_true")
    args = parser.parse_args()
    main(args.config.resolve(), args.wait_for_captures)
