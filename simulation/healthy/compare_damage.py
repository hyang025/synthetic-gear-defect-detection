"""Compare the saved matched captures; figures use shared display scales."""

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".tools" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from PIL import Image, ImageDraw


def analyze(output, presentation_only=False):
    output = Path(output)
    for folder in ("images", "frames", "animations"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    dents = json.loads((output / "dents.json").read_text())["dents"]
    frames = manifest["frames"]
    images = {case: np.stack([np.asarray(mi.Bitmap(str(output/f["files"][case])))[..., 0].copy() for f in frames]) for case in ("healthy", "damaged")}
    difference = images["damaged"]-images["healthy"]
    vmax = float(np.quantile(np.stack(list(images.values())), 0.999)) or 1
    dmax = float(np.quantile(np.abs(difference), 0.999)) or 1
    h, w = images["healthy"].shape[1:]
    camera = config["camera"]
    angle = np.degrees(np.arctan((np.arange(w)-(w-1)/2)*(camera["sensor_width_mm"]/w)/camera["physical_focal_length_mm"]))
    profiles = {case: values.sum(axis=1, dtype=np.float64) for case, values in images.items()}
    energies = {case: values.sum(axis=1) for case, values in profiles.items()}
    means = {case: (values*angle).sum(axis=1)/energies[case] for case, values in profiles.items()}
    spreads = {case: np.sqrt((values*(angle-means[case][:, None])**2).sum(axis=1)/energies[case]) for case, values in profiles.items()}
    records = []
    for i, f in enumerate(frames):
        records.append({k: f[k] for k in ("index", "dent_id", "diameter_mm", "depth_mm", "gear_angle_deg", "offset_deg", "healthy_energy", "damaged_energy", "delta_energy_percent", "delta_energy_percent_mc_se")})
        records[-1].update({"healthy_mean_angle_deg": means["healthy"][i], "damaged_mean_angle_deg": means["damaged"][i],
                           "delta_mean_angle_deg": means["damaged"][i]-means["healthy"][i],
                           "healthy_spread_deg": spreads["healthy"][i], "damaged_spread_deg": spreads["damaged"][i]})
    if not presentation_only:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        np.savez_compressed(output / "comparison_profiles.npz", reflected_angle_deg=angle,
                            healthy_profile=profiles["healthy"], damaged_profile=profiles["damaged"],
                            gear_angle_deg=[f["gear_angle_deg"] for f in frames],
                            healthy_mean_angle_deg=means["healthy"], damaged_mean_angle_deg=means["damaged"],
                            healthy_spread_deg=spreads["healthy"], damaged_spread_deg=spreads["damaged"])
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), layout="constrained")
    center_indices, summaries = [], []
    for row, dent in enumerate(dents):
        indices = np.array([i for i, f in enumerate(frames) if f["dent_id"] == dent["id"]])
        offsets = np.array([frames[i]["offset_deg"] for i in indices])
        delta = np.array([frames[i]["delta_energy_percent"] for i in indices])
        error = np.array([frames[i]["delta_energy_percent_mc_se"] for i in indices])
        center = int(indices[np.argmin(abs(offsets))])
        center_indices.append(center)
        healthy_ref = energies["healthy"][center]
        axes[row, 0].plot(offsets, energies["healthy"][indices]/healthy_ref, label="Healthy", color="#176dad")
        axes[row, 0].plot(offsets, energies["damaged"][indices]/healthy_ref, label="Damaged", color="#bb3940")
        axes[row, 0].set(title=f"Ø{dent['diameter_mm']:g} × {dent['depth_mm']:g} mm deep", ylabel="Relative collected intensity")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(offsets, delta, color="#bb3940", lw=1.5)
        axes[row, 1].fill_between(offsets, delta-2*error, delta+2*error, color="#bb3940", alpha=0.2, label="±2 Monte Carlo SE")
        axes[row, 1].axhline(0, color="gray", lw=0.8)
        axes[row, 1].set(title="Intensity change", ylabel="Change from healthy (%)")
        axes[row, 1].legend(fontsize=8)
        axes[row, 2].plot(angle, profiles["healthy"][center], label="Healthy", color="#176dad")
        axes[row, 2].plot(angle, profiles["damaged"][center], label="Damaged", color="#bb3940")
        axes[row, 2].set(title="Dent-center profile", xlabel="Horizontal reflected angle (deg)", ylabel="Linear intensity summed over rows")
        for ax in axes[row, :2]:
            ax.set_xlabel("Rotation offset from dent crossing (deg)")
        for ax in axes[row]:
            ax.grid(alpha=0.2)
        peak = int(indices[np.argmax(abs(delta))])
        summaries.append({"dent_id": dent["id"], "diameter_mm": dent["diameter_mm"], "depth_mm": dent["depth_mm"],
                          "nominal_crossing_deg": dent["nominal_laser_crossing_deg"],
                          "center_delta_energy_percent": frames[center]["delta_energy_percent"],
                          "center_mc_se_percent": frames[center]["delta_energy_percent_mc_se"],
                          "largest_abs_energy_change_percent": frames[peak]["delta_energy_percent"],
                          "peak_offset_deg": frames[peak]["offset_deg"],
                          "peak_mc_se_percent": frames[peak]["delta_energy_percent_mc_se"]})
    fig.suptitle("Dent comparison", fontsize=16)
    fig.savefig(output / "images" / "damage_summary.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    for row, index in enumerate(center_indices):
        for col, case in enumerate(("healthy", "damaged")):
            im = axes[row, col].imshow(images[case][index], cmap="inferno", vmin=0, vmax=vmax)
            axes[row, col].set_title(f"Ø{frames[index]['diameter_mm']:g} mm · {case.capitalize()}")
        di = axes[row, 2].imshow(difference[index], cmap="coolwarm", vmin=-dmax, vmax=dmax)
        axes[row, 2].set_title("Difference · Separate scale")
        fig.colorbar(di, ax=axes[row, 2], label="Signed linear-intensity difference")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Dent-center captures", fontsize=16)
    fig.savefig(output / "images" / "damage_captures.png", dpi=150)
    plt.close(fig)
    animation = []
    for index, f in enumerate(frames):
        canvas = Image.new("RGB", (960, 288), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"Dent {f['diameter_mm']:g} x {f['depth_mm']:g} mm (diameter x depth) | gear {f['gear_angle_deg']:.2f} deg | offset {f['offset_deg']:+.1f} deg", fill="black")
        arrays = [images["healthy"][index], images["damaged"][index], difference[index]]
        for col, values in enumerate(arrays):
            norm = np.clip(values/vmax, 0, 1) if col < 2 else np.clip(0.5+values/(2*dmax), 0, 1)
            rgb = (plt.get_cmap("inferno" if col < 2 else "coolwarm")(norm)[..., :3]*255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb).resize((320, 200), Image.Resampling.BILINEAR), (col*320, 44))
            draw.text((col*320+8, 27), ("HEALTHY", "DAMAGED", "DIFFERENCE (SEPARATE SCALE)")[col], fill="black")
        draw.text((8, 252), f"Intensity change: {f['delta_energy_percent']:+.3f}% | Healthy / damaged: shared scale", fill="black")
        draw.text((8, 272), "Three dent passages | Matched angles", fill="black")
        canvas.save(output / "frames" / f"frame_{index:03d}.png")
        animation.append(canvas.resize((480, 144), Image.Resampling.BILINEAR).quantize(colors=128, dither=Image.Dither.NONE))
    animation[0].save(output / "animations" / "damage_comparison.gif", save_all=True, append_images=animation[1:], duration=100, loop=0)
    result = {"matched_poses": len(frames), "case_captures": 2*len(frames), "dents": summaries,
              "display_max": vmax, "difference_display_abs_max": dmax,
              "display_note": "Healthy/damaged use a shared scale; signed difference has a separate scale. Both scales are fixed across frames.",
              "interpretation": "Differences use the recorded geometry/material/optics assumptions. Monte Carlo SE describes numerical sampling variability. CAD, material, and hardware uncertainties require separate characterization."}
    if not presentation_only:
        (output / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
    else:
        print(f"Updated presentation from {len(frames)} saved capture pairs in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--presentation-only", action="store_true", help="Refresh figures and animation frames while preserving numerical result files.")
    args = parser.parse_args()
    analyze(args.output, presentation_only=args.presentation_only)
