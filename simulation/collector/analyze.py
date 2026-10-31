"""Plot saved focal-plane captures and their angular distributions."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent/".tools/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mitsuba as mi
import numpy as np

CASES = ("healthy", "dented", "tilted")
LABELS = {"healthy": "Healthy", "dented": "Dent · Ø0.10 mm × 25 µm deep", "tilted": "Healthy · 0.05° tilt"}
COLORS = {"healthy": "#176dad", "dented": "#bf373b", "tilted": "#258158"}


def moments(profile, coordinate):
    energy = profile.sum()
    if not np.isfinite(energy) or energy <= 0:
        raise RuntimeError("A profile has insufficient signal for its moments.")
    mean = float(profile@coordinate/energy)
    variance = float(profile@(coordinate-mean)**2/energy)
    return float(energy), mean, float(np.sqrt(variance))


def analyze(output):
    out = Path(output)
    config = json.loads((out/"config.json").read_text())
    manifest = json.loads((out/"manifest.json").read_text())
    data = np.load(out/"profiles.npz")
    settings = config["analysis"]
    edges = np.linspace(-settings["angle_half_range_deg"], settings["angle_half_range_deg"], settings["angle_bins"]+1)
    angle = (edges[1:]+edges[:-1])/2
    angle_width = edges[1]-edges[0]
    profiles = {case: data[f"{case}_v_batches"].mean(axis=0) for case in CASES}
    count = len(data["healthy_v_batches"])
    images = {case: np.asarray(mi.Bitmap(str(out/"raw"/f"{case}.exr"))).squeeze() for case in CASES}
    screen = config["screen"]
    bin_area = screen["width_mm"]*screen["height_mm"]/(screen["width_px"]*screen["height_px"])
    screen_v = ((np.arange(screen["height_px"])+0.5)/screen["height_px"]-0.5)*screen["height_mm"]
    records = []
    base_energy, base_mean, base_std = moments(profiles["healthy"], angle)
    base_normalized = profiles["healthy"]/base_energy
    for case in CASES:
        energy, mean, std = moments(profiles[case], angle)
        _, screen_mean, screen_std = moments(images[case].sum(axis=1), screen_v)
        batch = data[f"{case}_v_batches"]
        healthy_batch = data["healthy_v_batches"]
        delta_batch = batch.sum(axis=1)-healthy_batch.sum(axis=1)
        row = {"case": case, "screen_power_relative": energy,
               "energy_change_percent": 100*(energy/base_energy-1),
               "energy_change_mc_se_percentage_points": float(100*delta_batch.std(ddof=1)/np.sqrt(count)/base_energy),
               "angular_centroid_deg": mean, "centroid_shift_deg": mean-base_mean,
               "angular_spread_deg": std, "spread_change_deg": std-base_std,
               "screen_centroid_mm": screen_mean, "screen_spread_mm": screen_std,
               "profile_l1_relative": float(abs(profiles[case]-profiles["healthy"]).sum()),
               "profile_l1_percent_of_healthy": float(100*abs(profiles[case]-profiles["healthy"]).sum()/base_energy),
               "normalized_profile_l1": float(abs(profiles[case]/energy-base_normalized).sum())}
        for field in ("collector_power", "screen_clipped_power", "depth_truncated_power", "surface_incident_power", "sampled_source_power"):
            row[field] = float(np.mean([b["cases"][case][field] for b in manifest["batches"]]))
        records.append(row)
    with (out/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    with (out/"angular_profiles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["angle_center_deg", *[f"{case}_power_per_degree" for case in CASES],
                         "dent_minus_healthy_per_degree", "tilt_minus_healthy_per_degree"])
        writer.writerows(zip(angle, *[profiles[case]/angle_width for case in CASES],
                             (profiles["dented"]-profiles["healthy"])/angle_width,
                             (profiles["tilted"]-profiles["healthy"])/angle_width))
    result = {"analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": records, "axis": "v: incidence-plane angle relative to the nominal collector axis",
              "angle_bin_width_deg": float(angle_width), "screen_bin_area_mm2": bin_area,
              "angular_profile_scope": "Rays admitted by the collector and falling within the screen",
              "radiometry": "Unit source power; raw EXR values are collected power per screen bin",
              "uncertainty": "Paired-batch standard errors describe numerical sampling variability",
              "interpretation_scope": "Tooth-sized planar surface with uniform assumed GGX roughness and ideal angular mapping"}
    (out/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10})
    irradiance = {case: images[case]/bin_area for case in CASES}
    vmax = max(float(image.max()) for image in irradiance.values())
    color_map = plt.get_cmap("inferno").copy()
    color_map.set_bad(color_map(0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout="constrained")
    extent = [-screen["width_mm"]/2, screen["width_mm"]/2, -screen["height_mm"]/2, screen["height_mm"]/2]
    for ax, case in zip(axes, CASES):
        shown = ax.imshow(irradiance[case], origin="lower", extent=extent, cmap=color_map,
                          norm=LogNorm(vmin=vmax*1e-5, vmax=vmax))
        ax.set(title=LABELS[case], xlabel="Screen u (mm)", ylabel="Screen v (mm)")
    fig.colorbar(shown, ax=list(axes), shrink=0.8, label="Relative power / mm² · Log scale")
    fig.suptitle("Ideal collector · Focal-plane patterns at 25 mm", fontsize=17)
    fig.savefig(out/"images/focal_plane.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for case in CASES:
        density = profiles[case]/angle_width
        for ax in axes[0]:
            ax.plot(angle, density, color=COLORS[case], label=LABELS[case], lw=1.4,
                    ls="--" if case == "tilted" else "-")
        axes[1, 1].plot(angle, density/profiles[case].sum(), color=COLORS[case], label=LABELS[case],
                       lw=1.4, ls="--" if case == "tilted" else "-")
    axes[0, 0].set(title="Collected angular distribution", xlim=(-55, 55), ylabel="Relative power / degree")
    axes[0, 1].set(title="Central lobe", xlim=(-8, 8), ylabel="Relative power / degree")
    for case in ("dented", "tilted"):
        differences = (data[f"{case}_v_batches"]-data["healthy_v_batches"])/angle_width
        delta, se = differences.mean(axis=0), differences.std(axis=0, ddof=1)/np.sqrt(count)
        axes[1, 0].plot(angle, delta, color=COLORS[case], label=f"{case.capitalize()} − healthy", lw=1.2)
        axes[1, 0].fill_between(angle, delta-2*se, delta+2*se, color=COLORS[case], alpha=0.18)
    axes[1, 0].axhline(0, color="gray", lw=0.8)
    axes[1, 0].set(title="Profile change · ±2 MC SE", xlim=(-45, 45), ylabel="Power change / degree")
    axes[1, 1].set(title="Normalized profiles", xlim=(-12, 12), ylabel="Normalized signal / degree")
    for ax in axes.flat:
        ax.set_xlabel("Reflected angle v (deg)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Angular reflection · 25 mm collector distance", fontsize=17)
    fig.savefig(out/"images/angular_distribution.png", dpi=170)
    plt.close(fig)
    # Resolve the positive and negative redistribution for the dent at full scale.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, case in zip(axes, ("dented", "tilted")):
        differences = (data[f"{case}_v_batches"]-data["healthy_v_batches"])/angle_width
        delta, se = differences.mean(axis=0), differences.std(axis=0, ddof=1)/np.sqrt(count)
        ax.plot(angle, delta, color=COLORS[case], lw=1.3)
        ax.fill_between(angle, delta-2*se, delta+2*se, color=COLORS[case], alpha=0.2)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(title=f"{LABELS[case]} · ±2 MC SE", xlabel="Reflected angle v (deg)", ylabel="Power change / degree",
               xlim=(-50, 50) if case == "dented" else (-10, 10))
        ax.grid(alpha=0.2)
    fig.suptitle("Change from healthy · Separate axis scales", fontsize=16)
    fig.savefig(out/"images/profile_changes.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
