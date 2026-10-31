"""Matched full-revolution comparison for the three existing geometric dents."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from run import ROOT, PROJECT
from compare_rotation import block_sum, moments
import mitsuba as mi
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


def analyze(out):
    started = time.perf_counter()
    config = json.loads((out/"config.json").read_text())
    manifest = json.loads((out/"manifest.json").read_text())
    metadata = json.loads((ROOT/config["dent_metadata"]).read_text())
    dents = metadata["dents"]
    h = np.load(out/"healthy_profiles.npz")
    d = np.load(out/"damaged_profiles.npz")
    angles = d["angle_deg"]
    if not np.array_equal(angles, h["angle_deg"]):
        raise RuntimeError("The paired encoder positions differ.")
    bins, half = config["analysis"]["angle_bins"], config["analysis"]["angle_half_range_deg"]
    delta = 2*half/bins
    theta = np.linspace(-half+delta/2, half-delta/2, bins)
    hp = {a: h[f"{a}_batches"].mean(axis=1) for a in ("u", "v")}
    dp = {a: d[f"{a}_batches"].mean(axis=1) for a in ("u", "v")}
    eh, ed = hp["v"].sum(axis=1), dp["v"].sum(axis=1)
    ehb, edb = h["v_batches"].sum(axis=2), d["v_batches"].sum(axis=2)
    change = 100*(ed-eh)/eh
    energy_se = 100*np.std(edb-(ed/eh)[:, None]*ehb, axis=1, ddof=1)/np.sqrt(ehb.shape[1])/eh
    diff = {a: dp[a]-hp[a] for a in hp}
    diff_se = {a: np.std(d[f"{a}_batches"]-h[f"{a}_batches"], axis=1, ddof=1)/np.sqrt(ehb.shape[1]) for a in hp}
    stats = {}
    for axis in hp:
        _, mh, sh = moments(hp[axis], theta)
        _, md, sd = moments(dp[axis], theta)
        stats[axis] = {"centroid_shift_deg": md-mh, "spread_change_deg": sd-sh,
                       "raw_profile_l1_percent": 100*np.abs(diff[axis]).sum(axis=1)/eh,
                       "normalized_profile_l1_percent": 100*np.abs(dp[axis]/ed[:, None]-hp[axis]/eh[:, None]).sum(axis=1)}
    rows = [{"angle_deg": float(angle), "healthy_screen_power": float(eh[i]),
             "damaged_screen_power": float(ed[i]), "energy_change_percent": float(change[i]),
             "paired_energy_change_se_percentage_points": float(energy_se[i]),
             **{f"{axis}_{key}": float(value[i]) for axis in stats for key, value in stats[axis].items()}}
            for i, angle in enumerate(angles)]
    with (out/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    crossings = [int(np.argmin(abs(angles-dent["nominal_laser_crossing_deg"]))) for dent in dents]
    summary = {"frames": len(angles), "dent_crossings": [
        {"id": dent["id"], "diameter_mm": dent["diameter_mm"], "depth_mm": dent["depth_mm"],
         "tooth_index": dent["tooth_index"], "frame_index": idx, "metrics": rows[idx]} for dent, idx in zip(dents, crossings)],
        "largest_absolute_energy_change": rows[int(np.argmax(abs(change)))],
        "screen_clipped_power_max": max(f["damaged"]["power"]["screen_clipped_power"] for f in manifest["frames"]),
        "depth_truncated_power_max": max(f["damaged"]["power"]["depth_truncated_power"] for f in manifest["frames"])}
    (out/"comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(out/"comparison_profiles.npz", gear_angle_deg=angles, reflection_angle_deg=theta,
                        **{f"{case}_{axis}": p[axis] for case, p in (("healthy", hp), ("damaged", dp)) for axis in hp},
                        **{f"difference_{axis}_se": value for axis, value in diff_se.items()})
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    colors = ("#247ba0", "#d1495b")
    labels = [f"Ø{dent['diameter_mm']:.1f} × {dent['depth_mm']:.1f} mm" for dent in dents]
    for axis in ("u", "v"):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for col, (index, label) in enumerate(zip(crossings, labels)):
            ax = axs[0, col]
            ax.plot(theta, hp[axis][index]/delta, label="Healthy", color=colors[0])
            ax.plot(theta, dp[axis][index]/delta, label="Dented", color=colors[1], ls="--")
            ax.set(title=f"{label}\n{angles[index]:.3f}°", ylabel="Beam fraction / degree")
            ax.legend()
            ax = axs[1, col]
            residual, se = diff[axis][index]/delta, diff_se[axis][index]/delta
            ax.plot(theta, residual, color=colors[1])
            ax.fill_between(theta, residual-2*se, residual+2*se, color=colors[1], alpha=0.2)
            ax.axhline(0, color="gray", lw=0.7)
            ax.set(ylabel="Dented − healthy / degree")
            for ax in axs[:, col]:
                ax.set(xlabel=f"Reflection angle {axis} (°)", xlim=(-half, half))
                ax.grid(alpha=0.2)
        fig.suptitle(f"Three dent crossings · angular profiles {axis}")
        fig.savefig(out/"images"/f"angular_comparison_{axis}.png")
        plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), layout="constrained")
    for col, (index, label) in enumerate(zip(crossings, labels)):
        ax = axs[0, col]
        ax.plot(angles, change, color=colors[1], marker=".")
        ax.fill_between(angles, change-2*energy_se, change+2*energy_se, color=colors[1], alpha=0.2)
        ax.axvline(angles[index], color="gray", ls=":", lw=0.7)
        ax.set(title=label, xlim=(angles[index]-2, angles[index]+2), xlabel="Gear angle (°)", ylabel="Captured-signal change (%)")
        ax.grid(alpha=0.2)
    axs[1, 0].plot(angles, change, color=colors[1])
    axs[1, 0].set(title="Full revolution", xlim=(0, 360), ylabel="Captured-signal change (%)")
    for axis, color in zip(("u", "v"), colors):
        axs[1, 1].plot(angles, stats[axis]["normalized_profile_l1_percent"], label=axis, color=color)
        axs[1, 2].plot(angles, stats[axis]["spread_change_deg"], label=axis, color=color)
    axs[1, 1].set(title="Angular shape change", ylabel="Normalized profile distance (%)")
    axs[1, 2].set(title="Angular broadening", ylabel="Spread change (°)")
    for ax in axs[1]:
        ax.set(xlim=(0, 360), xlabel="Gear angle (°)")
        ax.grid(alpha=0.2)
    axs[1, 1].legend()
    axs[1, 2].legend()
    fig.suptitle("Existing three-dent gear · ideal collector at 25 mm")
    fig.savefig(out/"images/damage_signal.png")
    plt.close(fig)

    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")
    images = {case: [np.asarray(mi.Bitmap(str(out/"raw"/case/f"frame_{i:03d}.exr"))).squeeze()
                    for i in range(len(angles))] for case in ("healthy", "damaged")}
    residuals = [block_sum(di-hi) for hi, di in zip(images["healthy"], images["damaged"])]
    maximum = max(float(im.max()) for case in images.values() for im in case)
    norm = LogNorm(vmin=maximum/3000, vmax=maximum)
    dmax = max(float(abs(im).max()) for im in residuals)
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
    cmap = plt.colormaps["inferno"].copy()
    cmap.set_bad(cmap(0))
    extent = [-75, 75, -75, 75]
    fig, axs = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
    for row, (index, label) in enumerate(zip(crossings, labels)):
        for col, case in enumerate(("healthy", "damaged")):
            art = axs[row, col].imshow(images[case][index], origin="lower", extent=extent, cmap=cmap, norm=norm)
            axs[row, col].set(title=f"{case.capitalize()} · {label}")
        fig.colorbar(art, ax=axs[row, :2], shrink=0.6, label="Beam fraction / bin")
        art = axs[row, 2].imshow(residuals[index], origin="lower", extent=extent, cmap="RdBu_r", norm=dnorm)
        axs[row, 2].set(title=f"Dented − healthy · {angles[index]:.3f}°")
        fig.colorbar(art, ax=axs[row, 2], shrink=0.6, label="Beam fraction / 8 × 8 bins")
    for ax in axs.flat:
        ax.set(xlabel="Screen u (mm)", ylabel="Screen v (mm)")
    fig.suptitle("Three dent crossings · shared screen scales")
    fig.savefig(out/"images/screen_comparison.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    fig = plt.figure(figsize=(14.4, 8), dpi=100, layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.3, 0.8))
    artists = []
    for col, label in enumerate(("Healthy", "Three dents", "Dented − healthy")):
        ax = fig.add_subplot(grid[0, col])
        art = ax.imshow(images["healthy"][0] if col < 2 else residuals[0], origin="lower", extent=extent,
                        cmap=cmap if col < 2 else "RdBu_r", norm=norm if col < 2 else dnorm)
        artists.append(art)
        ax.set(title=label, xlabel="Screen u (mm)", ylabel="Screen v (mm)")
        fig.colorbar(art, ax=ax, shrink=0.7, label="Beam fraction / bin" if col < 2 else "Beam fraction / 8 × 8 bins")
    ax = fig.add_subplot(grid[1, :2])
    hc, = ax.plot(theta, hp["v"][0]/delta, label="Healthy", color=colors[0])
    dc, = ax.plot(theta, dp["v"][0]/delta, label="Three dents", color=colors[1], ls="--")
    ax.set(xlim=(-half, half), ylim=(0, max(hp["v"].max(), dp["v"].max())/delta*1.06),
           xlabel="Reflection angle v (°)", ylabel="Beam fraction / degree")
    ax.legend()
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(grid[1, 2])
    ax.plot(angles, change, color=colors[1])
    cursor = ax.axvline(0, color=colors[0], lw=1)
    ax.set(xlim=(0, 360), xlabel="Gear angle (°)", ylabel="Captured-signal change (%)")
    ax.grid(alpha=0.2)
    title = fig.suptitle("")
    center_labels = dict(zip(crossings, labels))
    for i, angle in enumerate(angles):
        for art, im in zip(artists, (images["healthy"][i], images["damaged"][i], residuals[i])):
            art.set_data(im)
        hc.set_ydata(hp["v"][i]/delta)
        dc.set_ydata(dp["v"][i]/delta)
        cursor.set_xdata([angle, angle])
        dent_label = f" · {center_labels[i]}" if i in center_labels else ""
        title.set_text(f"Three dents · {angle:.3f}°{dent_label} · signal change {change[i]:+.3f}%")
        fig.savefig(out/"frames"/f"frame_{i:03d}.png", dpi=100)
        if i == 0 or (i+1) % 60 == 0:
            print(f"Comparison frames: {i+1}/{len(angles)}", flush=True)
    plt.close(fig)
    manifest["analysis_seconds"] = time.perf_counter()-started
    manifest["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["render_and_analysis_seconds"] = manifest["render_seconds"]+manifest["analysis_seconds"]
    manifest["status"] = "frames_complete"
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=PROJECT/"outputs/collector_three_dents_25mm")
    analyze(parser.parse_args().output.resolve())
