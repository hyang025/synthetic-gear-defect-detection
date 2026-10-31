"""Matched healthy/dented encoder profiles, screen captures, and animation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from run import ROOT, PROJECT
import mitsuba as mi
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


def block_sum(image, factor=8):
    h, w = image.shape
    return image.reshape(h//factor, factor, w//factor, factor).sum(axis=(1, 3))


def moments(profile, theta):
    energy = profile.sum(axis=-1)
    mu = (profile*theta).sum(axis=-1)/energy
    spread = np.sqrt((profile*(theta-mu[..., None])**2).sum(axis=-1)/energy)
    return energy, mu, spread


def geometry_picture(out, metadata_path):
    mesh = np.load(metadata_path.parent/"deformation.npz")
    local = (mesh["vertices_mm"]-mesh["center_mm"])@mesh["basis"]
    base = (mesh["base_vertices_mm"]-mesh["center_mm"])@mesh["basis"]
    faces = mesh["faces"]
    use = (np.max(abs(base[faces, :2]), axis=(1, 2)) < 0.065) & (np.max(abs(base[faces, 2]), axis=1) < 0.01)
    triangles = faces[use]
    ids, inverse = np.unique(triangles, return_inverse=True)
    vertices = local[ids]*1000
    fig = plt.figure(figsize=(11, 4.7))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.86, wspace=0.3)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=inverse.reshape(-1, 3), cmap="cividis", linewidth=0, antialiased=False)
    ax.set(xlabel="Along tooth (µm)", ylabel="Across tooth (µm)", zlabel="Height (µm)")
    ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect(np.ptp(vertices, axis=0))
    ax = fig.add_subplot(1, 2, 2)
    x = np.linspace(-65, 65, 501)
    depth = -25*np.maximum(0, 1-(x/50)**2)**2
    ax.plot(x, depth, color="#246b8f")
    ax.axhline(0, color="gray", lw=0.7)
    ax.annotate("", xy=(-50, 5), xytext=(50, 5), arrowprops={"arrowstyle": "<->"})
    ax.text(0, 6.5, "0.10 mm", ha="center")
    ax.annotate("", xy=(58, 0), xytext=(58, -25), arrowprops={"arrowstyle": "<->"})
    ax.text(60, -12.5, "25 µm", rotation=90, va="center")
    ax.set(xlabel="Distance from dent center (µm)", ylabel="Normal displacement (µm)", ylim=(-29, 12))
    ax.grid(alpha=0.2)
    fig.suptitle("Tooth 7 · Ø0.10 mm · 25 µm deep")
    fig.savefig(out/"images/dent_geometry.png", dpi=140)
    plt.close(fig)


def analyze(out):
    started = time.perf_counter()
    config = json.loads((out/"config.json").read_text())
    manifest = json.loads((out/"manifest.json").read_text())
    reference = (ROOT/config["healthy_reference"]).resolve()
    healthy_manifest = json.loads((reference/"manifest.json").read_text())
    if manifest["geometry"] != healthy_manifest["geometry"]:
        raise RuntimeError("The paired optical layouts differ.")
    metadata_path = (ROOT/config["dent_metadata"]).resolve()
    dent = json.loads(metadata_path.read_text())
    healthy = np.load(reference/"profiles.npz")
    damaged = np.load(out/"profiles.npz")
    angles = damaged["angle_deg"]
    if not np.array_equal(angles, healthy["angle_deg"]):
        raise RuntimeError("The paired encoder positions differ.")
    bins = config["analysis"]["angle_bins"]
    half = config["analysis"]["angle_half_range_deg"]
    delta = 2*half/bins
    theta = np.linspace(-half+delta/2, half-delta/2, bins)
    hp = {axis: healthy[f"{axis}_batches"].mean(axis=1) for axis in ("u", "v")}
    dp = {axis: damaged[f"{axis}_batches"].mean(axis=1) for axis in ("u", "v")}
    eh, ed = hp["v"].sum(axis=1), dp["v"].sum(axis=1)
    eh_batch = healthy["v_batches"].sum(axis=2)
    ed_batch = damaged["v_batches"].sum(axis=2)
    change = 100*(ed-eh)/eh
    ratio = ed/eh
    energy_se = 100*np.std(ed_batch-ratio[:, None]*eh_batch, axis=1, ddof=1)/np.sqrt(eh_batch.shape[1])/eh
    diff = {axis: dp[axis]-hp[axis] for axis in hp}
    diff_se = {axis: np.std(damaged[f"{axis}_batches"]-healthy[f"{axis}_batches"], axis=1, ddof=1)/np.sqrt(eh_batch.shape[1]) for axis in hp}
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
    crossing = int(np.argmin(abs(angles-dent["nominal_laser_crossing_deg"])))
    strongest = int(np.argmax(abs(change)))
    summary = {"dent": dent, "frames": len(angles), "nominal_crossing": rows[crossing],
               "largest_absolute_energy_change": rows[strongest],
               "largest_v_profile_distance": rows[int(np.argmax(stats["v"]["raw_profile_l1_percent"]))],
               "screen_clipped_power_max": max(f["power"]["screen_clipped_power"] for f in manifest["frames"]),
               "depth_truncated_power_max": max(f["power"]["depth_truncated_power"] for f in manifest["frames"]),
               "healthy_reference": str(reference.relative_to(PROJECT))}
    (out/"comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(out/"comparison_profiles.npz", gear_angle_deg=angles, reflection_angle_deg=theta,
                        **{f"{case}_{axis}": p[axis] for case, p in (("healthy", hp), ("damaged", dp)) for axis in hp},
                        **{f"difference_{axis}_se": value for axis, value in diff_se.items()})
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    geometry_picture(out, metadata_path)
    colors = ("#247ba0", "#d1495b")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, axis in enumerate(("u", "v")):
        ax = axs[0, col]
        ax.plot(theta, hp[axis][crossing]/delta, label="Healthy", color=colors[0])
        ax.plot(theta, dp[axis][crossing]/delta, label="Dented", color=colors[1], ls="--")
        ax.set(title=f"Angular profile {axis}", ylabel="Beam fraction / degree")
        ax.legend()
        ax = axs[1, col]
        d = diff[axis][crossing]/delta
        se = diff_se[axis][crossing]/delta
        ax.plot(theta, d, color=colors[1])
        ax.fill_between(theta, d-2*se, d+2*se, color=colors[1], alpha=0.2, label="±2 sampling SE")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(ylabel="Dented − healthy / degree")
        ax.legend()
        for ax in axs[:, col]:
            ax.set(xlabel=f"Reflection angle {axis} (°)", xlim=(-half, half))
            ax.grid(alpha=0.2)
    fig.suptitle(f"Dent crossing · gear angle {angles[crossing]:.0f}°")
    fig.savefig(out/"images/angular_comparison.png")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax in (axs[0, 0], axs[1, 0]):
        ax.plot(angles, change, color=colors[1], marker=".", ms=3)
        ax.fill_between(angles, change-2*energy_se, change+2*energy_se, color=colors[1], alpha=0.2)
        ax.set(ylabel="Captured-signal change (%)", xlabel="Gear angle (°)")
    axs[0, 0].set(xlim=(0, 360), title="Matched-angle energy change")
    axs[1, 0].set(xlim=(angles[crossing]-3, angles[crossing]+3), title="Dent passage")
    for axis, color in zip(("u", "v"), colors):
        axs[0, 1].plot(angles, stats[axis]["normalized_profile_l1_percent"], color=color, label=axis)
    axs[0, 1].set(xlim=(0, 360), ylabel="Normalized profile distance (%)", xlabel="Gear angle (°)", title="Angular shape change")
    axs[0, 1].legend()
    for key, label, color in (("centroid_shift_deg", "v centroid", colors[0]), ("spread_change_deg", "v spread", colors[1])):
        axs[1, 1].plot(angles, stats["v"][key], color=color, label=label, marker=".")
    axs[1, 1].set(xlim=(angles[crossing]-3, angles[crossing]+3), ylabel="Angular change (°)", xlabel="Gear angle (°)")
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.axvline(angles[crossing], color="gray", ls=":", lw=0.7)
        ax.grid(alpha=0.2)
    fig.suptitle("Healthy / dented · full revolution · collector at 25 mm")
    fig.savefig(out/"images/damage_signal.png")
    plt.close(fig)

    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")
    raw = out/"raw"/config["raw_case"]
    h_images = [np.asarray(mi.Bitmap(str(reference/"raw"/f"frame_{i:03d}.exr"))).squeeze() for i in range(len(angles))]
    d_images = [np.asarray(mi.Bitmap(str(raw/f"frame_{i:03d}.exr"))).squeeze() for i in range(len(angles))]
    residuals = [block_sum(d-h) for h, d in zip(h_images, d_images)]
    maximum = max(float(im.max()) for im in h_images+d_images)
    norm = LogNorm(vmin=maximum/3000, vmax=maximum)
    dmax = max(float(np.abs(im).max()) for im in residuals)
    dnorm = TwoSlopeNorm(vmin=-max(dmax, 1e-20), vcenter=0, vmax=max(dmax, 1e-20))
    cmap = plt.colormaps["inferno"].copy()
    cmap.set_bad(cmap(0))
    extent = [-75, 75, -75, 75]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained")
    for ax, im, label in zip(axs[:2], (h_images[crossing], d_images[crossing]), ("Healthy", "Dented")):
        artist = ax.imshow(im, origin="lower", extent=extent, cmap=cmap, norm=norm)
        ax.set(title=label, xlabel="Screen u (mm)", ylabel="Screen v (mm)")
    fig.colorbar(artist, ax=axs[:2], shrink=0.65, label="Beam fraction / bin")
    artist = axs[2].imshow(residuals[crossing], origin="lower", extent=extent, cmap="RdBu_r", norm=dnorm)
    axs[2].set(title="Dented − healthy", xlabel="Screen u (mm)", ylabel="Screen v (mm)")
    fig.colorbar(artist, ax=axs[2], shrink=0.65, label="Beam fraction / 8 × 8 bins")
    fig.suptitle(f"Full screen comparison · gear angle {angles[crossing]:.0f}°")
    fig.savefig(out/"images/screen_comparison.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    fig = plt.figure(figsize=(14.4, 8), dpi=100, layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.3, 0.8))
    artists = []
    for col, label in enumerate(("Healthy", "Dented", "Dented − healthy")):
        ax = fig.add_subplot(grid[0, col])
        artist = ax.imshow(h_images[0] if col < 2 else residuals[0], origin="lower", extent=extent,
                           cmap=cmap if col < 2 else "RdBu_r", norm=norm if col < 2 else dnorm)
        artists.append(artist)
        ax.set(title=label, xlabel="Screen u (mm)", ylabel="Screen v (mm)")
        fig.colorbar(artist, ax=ax, shrink=0.7, label="Beam fraction / bin" if col < 2 else "Beam fraction / 8 × 8 bins")
    ax = fig.add_subplot(grid[1, :2])
    hc, = ax.plot(theta, hp["v"][0]/delta, label="Healthy", color=colors[0])
    dc, = ax.plot(theta, dp["v"][0]/delta, label="Dented", color=colors[1], ls="--")
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
    for i, angle in enumerate(angles):
        for artist, im in zip(artists, (h_images[i], d_images[i], residuals[i])):
            artist.set_data(im)
        hc.set_ydata(hp["v"][i]/delta)
        dc.set_ydata(dp["v"][i]/delta)
        cursor.set_xdata([angle, angle])
        title.set_text(f"Healthy / dented · {angle:03.0f}° · signal change {change[i]:+.3f}%")
        fig.savefig(out/"frames"/f"frame_{i:03d}.png", dpi=100)
        if i == 0 or (i+1) % 60 == 0:
            print(f"Comparison frames: {i+1}/{len(angles)}", flush=True)
    plt.close(fig)
    manifest["analysis_seconds"] = time.perf_counter()-started
    manifest["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["status"] = "frames_complete"
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"nominal_crossing": rows[crossing], "largest_energy_change": rows[strongest]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=PROJECT/"outputs/collector_damaged_rotation_25mm")
    analyze(parser.parse_args().output.resolve())
