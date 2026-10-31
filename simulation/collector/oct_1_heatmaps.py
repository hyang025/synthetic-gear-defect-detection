"""Show mean PatchCore passage maps with exactly angle-matched healthy examples."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from prototype_patchcore import PROJECT, chunks, images_tensor, write_json
from validation_threshold import load_rows
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

DEMO = PROJECT/"outputs/oct_1_demo"
RUN = PROJECT/"outputs/patchcore_hemisphere_0p2_25mm"
REFERENCE = PROJECT/"outputs/patchcore_hemisphere_dense_25mm/histogram"


def main():
    target = DEMO/"heatmaps/passage_manifest.json"
    if target.exists():
        raise FileExistsError("Preserve the saved October 1 passage maps.")
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use the RTX GPU for the requested frozen-model visualization.")
    started = time.perf_counter()
    checkpoint_path = REFERENCE/"patchcore_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    model = PatchcoreModel(layers=settings["layers"], backbone=settings["backbone"],
        pre_trained=False, num_neighbors=settings["num_neighbors"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)
    predictions = load_rows(RUN/"predictions.csv")
    healthy = [row for row in predictions if row["case"] == "healthy_test"]
    gear = next(gear for gear in json.loads((RUN/"defect_predictions.json").read_text())["gears"]
                if gear["id"] == "gear_1")
    arrays, passages, panels = {}, [], []
    for defect in gear["defects"]:
        damaged = sorted([row for row in predictions if row["case"] == "gear_1"
                          and defect["id"] in row["defect_ids"].split(";")], key=lambda row: row["angle_deg"])
        matched = []
        for row in damaged:
            matches = [item for item in healthy if abs(item["angle_deg"]-row["angle_deg"]) < 1e-8]
            if len(matches) != 1:
                raise RuntimeError("Each passage image requires exactly one healthy image at its encoder angle.")
            matched.append(matches[0])
        pair = {"defect": defect["id"], "kind": defect["kind"], "detected": defect["detected"],
                "diameter_mm": defect["diameter_mm"], "depth_mm": defect["depth_mm"],
                "angles_deg": [row["angle_deg"] for row in damaged], "frames": []}
        display = {}
        for case, rows in (("healthy", matched), ("damaged", damaged)):
            maps, images = [], []
            folder = DEMO/"frames"/f"heatmaps_{case}"
            folder.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                for batch in chunks(rows, settings["batch_size"]):
                    result = model(images_tensor(RUN, batch, settings["image_size"], device))
                    maps.extend(result.anomaly_map.cpu().numpy()[:, 0].astype(np.float32))
            for row in rows:
                image_path = (RUN/row["image"]).resolve()
                images.append(np.asarray(Image.open(image_path).convert("L"), dtype=np.float32))
                copy_path = folder/f"{defect['id']}_frame_{row['index']:03d}.png"
                shutil.copy2(image_path, copy_path)
                pair["frames"].append({"case": case, "index": row["index"], "angle_deg": row["angle_deg"],
                    "source_image": str(image_path.relative_to(PROJECT)), "first_seed": row["first_seed"],
                    "demo_image": str(copy_path.relative_to(DEMO))})
            maps, images = np.stack(maps), np.stack(images)
            mean_map, mean_image = maps.mean(axis=0), images.mean(axis=0)
            prefix = f"{defect['id']}_{case}"
            arrays[prefix+"_individual_maps"] = maps
            arrays[prefix+"_mean_map"] = mean_map
            arrays[prefix+"_mean_input"] = mean_image
            display[case] = (mean_image, mean_map)
        arrays[defect["id"]+"_angles_deg"] = np.asarray(pair["angles_deg"])
        panels.append((defect, display))
        passages.append(pair)
    del model
    torch.cuda.empty_cache()
    vmax = max(float(display[case][1].max()) for _, display in panels for case in ("healthy", "damaged"))
    norm = Normalize(vmin=0, vmax=vmax)
    fig, axes = plt.subplots(4, 5, figsize=(15, 11), layout="constrained")
    heat = None
    for column, (defect, display) in enumerate(panels):
        for row, (case, field) in enumerate((("healthy", 0), ("damaged", 0), ("healthy", 1), ("damaged", 1))):
            ax = axes[row, column]
            if field == 0:
                ax.imshow(display[case][field], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            else:
                heat = ax.imshow(display[case][field], cmap="inferno", norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0, column].set_title(f"{defect['id']} · {defect['kind'].replace('_', ' ')}", fontsize=11)
        axes[1, column].set_title("Detected" if defect["detected"] else "Missed", fontsize=10)
    for row, label in enumerate(("Healthy input", "Damaged input", "Healthy map", "Damaged map")):
        axes[row, 0].set_ylabel(label, fontsize=11)
    fig.colorbar(heat, ax=axes[2:, :].ravel().tolist(), fraction=0.018, pad=0.02, label="Mean PatchCore response")
    fig.suptitle("Gear 1 · mean passage maps · matched encoder angles · Ø0.20 mm × depth 100 µm", fontsize=14)
    figure = DEMO/"images/06_combined_patchcore_heatmaps.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    np.savez_compressed(DEMO/"heatmaps/combined_passage_maps.npz", **arrays)
    sources = [Path(__file__), checkpoint_path, RUN/"predictions.csv", RUN/"defect_predictions.json"]
    write_json(target, {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "gear": "gear_1", "defects": 5, "method": "Arithmetic mean of pixel anomaly maps within each saved ±0.4 degree defect passage",
        "input_display": "Arithmetic mean of the corresponding histogram-equalized grayscale inputs",
        "healthy_matching": "Exact encoder-angle equality within 1e-8 degrees; held-out healthy_test frames",
        "screen_grid": "Fixed 128 × 128 screen-pixel grid", "color_scale": {"minimum": 0, "maximum": vmax, "shared": True},
        "model_and_threshold": "Frozen dense-training histogram PatchCore model and saved F1 threshold",
        "reported_detection_counts": "Preserved from the original single-frame passage rule",
        "scope": "Five gear-1 example passages including edge, crown, flank, detected, and missed cases",
        "elapsed_seconds": time.perf_counter()-started, "passages": passages,
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}})
    print(f"Saved combined maps for {len(passages)} angle-matched healthy/damaged passages.", flush=True)


if __name__ == "__main__":
    main()
