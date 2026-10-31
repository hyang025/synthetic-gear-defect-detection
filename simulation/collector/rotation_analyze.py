"""Screen animation and encoder-angle profiles for the ideal collector."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from run import PROJECT
import mitsuba as mi
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def analyze(out, make_frames=True):
    started = time.perf_counter()
    config = json.loads((out/"config.json").read_text())
    manifest = json.loads((out/"manifest.json").read_text())
    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")
    data = np.load(out/"profiles.npz")
    angles = data["angle_deg"]
    half = config["analysis"]["angle_half_range_deg"]
    bins = config["analysis"]["angle_bins"]
    delta = 2*half/bins
    theta = np.linspace(-half+delta/2, half-delta/2, bins)
    profiles = {axis: data[f"{axis}_batches"].mean(axis=1) for axis in ("u", "v")}
    energy = profiles["v"].sum(axis=1)
    moments = {}
    for axis, p in profiles.items():
        mu = (p*theta).sum(axis=1)/np.maximum(energy, 1e-30)
        sigma = np.sqrt((p*(theta[None, :]-mu[:, None])**2).sum(axis=1)/np.maximum(energy, 1e-30))
        moments[axis] = (mu, sigma)
    rows = []
    for i, frame in enumerate(manifest["frames"]):
        rows.append({"angle_deg": float(angles[i]), **frame["power"],
                     "screen_power_standard_error": frame["screen_power_standard_error"],
                     **{f"{axis}_{name}_deg": float(moments[axis][j][i])
                        for axis in ("u", "v") for j, name in enumerate(("centroid", "spread"))}})
    with (out/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Preserve all angle-resolved profiles as compact numeric arrays.
    np.savez_compressed(out/"angular_profiles.npz", gear_angle_deg=angles,
                        reflection_angle_deg=theta, u_power_per_bin=profiles["u"],
                        v_power_per_bin=profiles["v"])
    low, high = int(np.argmin(energy)), int(np.argmax(energy))
    summary = {
        "frames": len(angles), "reflection_angle_bin_deg": delta,
        "screen_energy_min": rows[low], "screen_energy_max": rows[high],
        "screen_energy_mean": float(energy.mean()),
        "screen_clipped_power_max": max(row["screen_clipped_power"] for row in rows),
        "depth_truncated_power_max": max(row["depth_truncated_power"] for row in rows),
        "centroid_range_deg": {axis: [float(moments[axis][0].min()), float(moments[axis][0].max())] for axis in moments},
        "spread_range_deg": {axis: [float(moments[axis][1].min()), float(moments[axis][1].max())] for axis in moments},
    }
    (out/"analysis.json").write_text(json.dumps(summary, indent=2)+"\n")
    images = [np.asarray(mi.Bitmap(str(out/"raw"/f"frame_{i:03d}.exr"))).squeeze() for i in range(len(angles))]
    maximum = max(float(im.max()) for im in images)
    minimum = maximum/3000
    norm = LogNorm(vmin=minimum, vmax=maximum)
    cmap = plt.colormaps["inferno"].copy()
    cmap.set_bad(cmap(0))
    extent = [-config["screen"]["width_mm"]/2, config["screen"]["width_mm"]/2,
              -config["screen"]["height_mm"]/2, config["screen"]["height_mm"]/2]
    selected = list(dict.fromkeys((0, low, high)))
    colors = ("#247ba0", "#d1495b", "#328750")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, axis in enumerate(("u", "v")):
        density = profiles[axis]/delta
        heat = axs[0, col].imshow(density.T, origin="lower", aspect="auto",
                                  extent=[angles[0]-0.5, angles[-1]+0.5, -half, half],
                                  cmap=cmap, norm=LogNorm(vmin=density.max()/3000, vmax=density.max()))
        axs[0, col].set(title=f"Angular distribution: {axis}", xlabel="Gear angle (°)", ylabel=f"Reflection angle {axis} (°)")
        fig.colorbar(heat, ax=axs[0, col], label="Beam fraction / degree")
        for index, color in zip(selected, colors):
            axs[1, col].plot(theta, density[index], color=color, label=f"{angles[index]:.0f}°")
        axs[1, col].set(xlabel=f"Reflection angle {axis} (°)", ylabel="Beam fraction / degree", xlim=(-half, half))
        axs[1, col].legend(title="Gear angle")
        axs[1, col].grid(alpha=0.2)
    fig.suptitle("Healthy gear · full revolution · ideal collector at 25 mm")
    fig.savefig(out/"images/angular_distribution.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axs[0].plot(angles, 100*energy, lw=1)
    axs[0].set(ylabel="Captured beam (%)", title="Healthy reflection over one revolution")
    for axis, color in zip(("u", "v"), colors):
        axs[1].plot(angles, moments[axis][0], label=axis, color=color, lw=1)
        axs[2].plot(angles, moments[axis][1], label=axis, color=color, lw=1)
    axs[1].set(ylabel="Centroid (°)")
    axs[2].set(ylabel="Angular spread (°)", xlabel="Gear angle (°)", xlim=(0, 360))
    for ax in axs:
        ax.grid(alpha=0.2)
    axs[1].legend()
    axs[2].legend()
    fig.savefig(out/"images/rotation_summary.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, len(selected), figsize=(5*len(selected), 5.4), constrained_layout=True, squeeze=False)
    for ax, index in zip(axs[0], selected):
        im = ax.imshow(images[index], origin="lower", extent=extent, cmap=cmap, norm=norm)
        ax.set(title=f"Gear angle {angles[index]:.0f}°", xlabel="Screen u (mm)", ylabel="Screen v (mm)")
    fig.colorbar(im, ax=axs.ravel().tolist(), label="Beam fraction / screen bin", shrink=0.8)
    fig.suptitle("Full focal-plane patterns · shared intensity scale")
    fig.savefig(out/"images/focal_plane.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    if not make_frames:
        print("Summary figures saved.", flush=True)
        return

    # Stable color and vertical scales preserve brightness changes in playback.
    fig = plt.figure(figsize=(12, 6), dpi=100, layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.1, 1))
    screen_ax = fig.add_subplot(grid[:, 0])
    image_artist = screen_ax.imshow(images[0], origin="lower", extent=extent, cmap=cmap, norm=norm)
    screen_ax.set(xlabel="Screen u (mm)", ylabel="Screen v (mm)", title="Focal-plane screen")
    fig.colorbar(image_artist, ax=screen_ax, label="Beam fraction / bin", shrink=0.7)
    curves = {}
    for row, axis in enumerate(("u", "v")):
        ax = fig.add_subplot(grid[row, 1])
        curves[axis], = ax.plot(theta, profiles[axis][0]/delta, lw=1.4)
        ax.set(xlim=(-half, half), ylim=(0, profiles[axis].max()/delta*1.06),
               xlabel=f"Reflection angle {axis} (°)", ylabel="Beam fraction / degree")
        ax.grid(alpha=0.2)
    title = fig.suptitle("")
    for i, angle in enumerate(angles):
        image_artist.set_data(images[i])
        for axis in curves:
            curves[axis].set_ydata(profiles[axis][i]/delta)
        title.set_text(f"Healthy gear · {angle:03.0f}° · captured beam {100*energy[i]:.2f}%")
        fig.savefig(out/"frames"/f"frame_{i:03d}.png", dpi=100)
        if i == 0 or (i+1) % 60 == 0:
            print(f"Display frames: {i+1}/{len(angles)}", flush=True)
    plt.close(fig)
    manifest["analysis_seconds"] = time.perf_counter()-started
    manifest["render_and_analysis_seconds"] = manifest["render_seconds"]+manifest["analysis_seconds"]
    manifest["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["status"] = "frames_complete"
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=PROJECT/"outputs/collector_rotation_25mm")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    analyze(args.output.resolve(), make_frames=not args.figures_only)
