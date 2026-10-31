"""Analyze saved linear captures without rerendering the scene."""

import argparse
import csv
import hashlib
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
from PIL import Image


def draw_setup(output, config, manifest):
    """Draw the recorded optical geometry."""
    geometry = manifest["geometry"]
    patch = np.asarray(geometry["patch_m"])*1000
    normal = np.asarray(geometry["normal"])
    across = np.asarray(geometry["across_tangent"])
    camera = np.asarray(geometry["camera_origin_m"])*1000
    laser = np.asarray(geometry["laser_origin_m"])*1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    ax = axes[0]
    ax.plot([-7, 7], [0, 0], color="gray", lw=4)
    ax.annotate("Tooth tangent plane", (0, -0.7), ha="center", va="top")
    ax.plot([0, 0], [0, 27], "--", color="gray", lw=1)
    ax.text(0.5, 26, "normal", color="gray")
    delta = camera-patch
    cam_x, cam_y = np.dot(delta, across), np.dot(delta, normal)
    ax.annotate("", (cam_x, cam_y), (0, 0), arrowprops={"arrowstyle": "->", "color": "#236ba0", "lw": 2})
    ax.scatter(cam_x, cam_y, color="#236ba0", s=50)
    c = config["camera"]
    ax.text(-16, 32, f"Pupil: {c['patch_to_pupil_mm']:g} mm from patch\n{c['physical_focal_length_mm']:g} mm lens, f/{c['f_number']:g}\nFocus plane: {c['focus_distance_m']:g} m", ha="left", va="top", fontsize=9)
    delta = (laser-patch)/np.linalg.norm(laser-patch)*27
    ax.annotate("", (0, 0), (np.dot(delta, across), np.dot(delta, normal)),
                arrowprops={"arrowstyle": "->", "color": "#b52f30", "lw": 2})
    l = config["laser"]
    ax.text(6, 26, f"Laser: {l['distance_mm']:g} mm\n{l['incidence_deg']:g}° incidence\nShortened path", color="#b52f30", fontsize=9)
    ax.scatter(0, 0, color="black", s=25)
    ax.set(xlim=(-18, 18), ylim=(-4, 36), xlabel="Along incidence-plane tangent (mm)", ylabel="Above tangent plane (mm)", title="Local incidence plane")
    ax.set_aspect("equal")
    ax = axes[1]
    vertices, triangles = [], []
    for line in Path(config["mesh_absolute"]).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            triangles.append([int(v.split("/")[0])-1 for v in line.split()[1:4]])
    v = np.asarray(vertices)
    ax.triplot(v[:, 0], v[:, 1], np.asarray(triangles), lw=0.15, color="#73818a", alpha=0.45)
    ax.scatter(*patch[:2], color="#b52f30", s=45, zorder=4)
    ax.annotate("Target tooth", patch[:2], (-6, -15),
                arrowprops={"arrowstyle": "->"}, color="#b52f30")
    if geometry.get("laser_layout") == "diameter":
        direction = np.asarray(geometry["diameter_radial"])
        radius = json.loads((output / "gear_parameters.json").read_text())["supplied"]["outside_diameter"]/2
        ax.plot([-radius*direction[0], radius*direction[0]], [-radius*direction[1], radius*direction[1]], color="#b52f30", lw=1.5)
        ax.text(0.025, 0.975, f"{l['surface_width_fwhm_mm']:g} mm laser line through bore", transform=ax.transAxes, va="top", color="#b52f30", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    ax.plot([patch[0], camera[0]], [patch[1], camera[1]], "--", color="#236ba0")
    ax.scatter(*camera[:2], color="#236ba0", s=35)
    ax.set(xlabel="Gear X (mm)", ylabel="Gear Y (mm)", title="Gear plan view")
    ax.set_aspect("equal")
    fig.suptitle("Inspection geometry", fontsize=15)
    fig.savefig(output / "images" / "setup.png", dpi=160)
    plt.close(fig)


def analyze(output, presentation_only=False):
    output = Path(output)
    for folder in ("images", "frames", "animations"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    frames = manifest["frames"]
    if len(frames) != len(config["inspection"]["gear_angles_deg"]):
        raise RuntimeError("The requested rotation sequence is incomplete.")
    # Keep only the illuminated channel in RAM for long sequences; original
    # RGB captures remain on disk. Accumulate profiles in double precision.
    intensity = np.stack([np.asarray(mi.Bitmap(str(output / f["file"])))[..., 0].copy() for f in frames])
    angles = np.array([f["gear_angle_deg"] for f in frames])
    nominal = int(np.argmin(np.abs(angles)))
    camera, settings = config["camera"], config["analysis"]
    h, w = intensity.shape[1:]
    pixels = np.arange(w, dtype=float)
    pitch = camera["sensor_width_mm"]/w
    x_angle = np.degrees(np.arctan((pixels-(w-1)/2)*pitch/camera["physical_focal_length_mm"]))
    y_angle = np.degrees(np.arctan((np.arange(h)-(h-1)/2)*camera["sensor_height_mm"]/h/camera["physical_focal_length_mm"]))
    profiles = intensity.sum(axis=1, dtype=np.float64)
    energy = profiles.sum(axis=1)
    threshold = settings["minimum_relative_energy"]*energy.max()
    signal_valid = (energy > 0) & (energy >= threshold) & np.isfinite(energy)
    norm = np.divide(profiles, energy[:, None], out=np.full_like(profiles, np.nan), where=signal_valid[:, None])
    centroid_px = np.sum(norm*pixels, axis=1)
    spread_px = np.sqrt(np.sum(norm*(pixels-centroid_px[:, None])**2, axis=1))
    mean_angle = np.sum(norm*x_angle, axis=1)
    spread_angle = np.sqrt(np.sum(norm*(x_angle-mean_angle[:, None])**2, axis=1))
    border = min(settings["border_pixels"], h//2, w//2)
    inner = intensity[:, border:-border, border:-border].sum(axis=(1, 2), dtype=np.float64)
    border_fraction = np.divide(energy-inner, energy, out=np.full_like(energy, np.nan), where=energy > 0)
    edge_warning = border_fraction > settings["border_fraction_warning"]
    derivative_valid = signal_valid[:-1] & signal_valid[1:] & ~edge_warning[:-1] & ~edge_warning[1:]
    derivative = np.where(derivative_valid, np.diff(mean_angle)/np.diff(angles), np.nan)
    derivative_angles = (angles[:-1]+angles[1:])/2
    # Preserve the original exposure scale in raw images and profiles.
    if not presentation_only:
        np.savez_compressed(output / "profiles.npz", gear_angle_deg=angles,
                            reflected_angle_deg=x_angle, profile_linear=profiles,
                            profile_normalized=norm, integrated_intensity=energy,
                            centroid_px=centroid_px, spread_px=spread_px,
                            mean_angle_deg=mean_angle, spread_angle_deg=spread_angle,
                            signal_valid=signal_valid, border_fraction=border_fraction,
                            edge_warning=edge_warning, derivative_gear_angle_deg=derivative_angles,
                            derivative_deg_per_deg=derivative, derivative_valid=derivative_valid)
        fields = ["gear_angle_deg", "integrated_intensity", "centroid_px", "spread_px",
                  "mean_angle_deg", "spread_angle_deg", "signal_valid", "border_fraction", "edge_warning"]
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(zip(angles, energy, centroid_px, spread_px, mean_angle,
                                 spread_angle, signal_valid, border_fraction, edge_warning))
        with (output / "derivative.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gear_angle_midpoint_deg", "d_reflected_angle_d_gear_angle", "valid"])
            writer.writerows(zip(derivative_angles, derivative, derivative_valid))
    # One display scale for every frame. Percentile clipping affects PNG/GIF only.
    display_max = float(np.quantile(intensity, 0.999))
    if display_max <= 0:
        display_max = float(intensity.max()) or 1.0
    extent = [x_angle[0], x_angle[-1], y_angle[-1], y_angle[0]]
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    ax = axes[0, 0]
    im = ax.imshow(intensity[nominal], cmap="inferno", vmin=0, vmax=display_max, extent=extent, aspect="auto")
    fig.colorbar(im, ax=ax, label="Linear red intensity (relative units)")
    ax.set(title=f"Capture at {angles[nominal]:+.1f}°", xlabel="Horizontal reflected angle (deg)", ylabel="Vertical angle (deg; down positive)")
    ax = axes[0, 1]
    selected = np.unique(np.linspace(0, len(angles)-1, min(9, len(angles)), dtype=int))
    for i in selected:
        ax.plot(x_angle, profiles[i], color=colors[i], label=f"{angles[i]:+.1f}°", lw=1)
    ax.set(title="Captured horizontal profiles", xlabel="Horizontal reflected angle (deg)", ylabel="Sum over image rows (relative units)")
    ax.legend(ncol=3, fontsize=8)
    ax = axes[0, 2]
    ax.plot(angles, energy, "o-", color="#236ba0")
    ax.axhline(threshold, color="gray", ls="--", label="1% of sequence maximum")
    ax.set(title="Collected intensity", xlabel="Gear rotation (deg)", ylabel="Integrated linear red intensity")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(angles, mean_angle, "o-", color="#236ba0")
    ax.fill_between(angles, mean_angle-spread_angle, mean_angle+spread_angle, alpha=0.2,
                    label="Profile spread (±1 SD)")
    ax.set(title="Captured mean angle and spread", xlabel="Gear rotation (deg)", ylabel="Horizontal reflected angle (deg)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(derivative_angles, derivative, "o-", color="#236ba0")
    ax.set(title="Adjacent-angle derivative", xlabel="Gear rotation midpoint (deg)", ylabel="d(mean reflected angle) / d(gear angle)")
    ax = axes[1, 2]
    ax.plot(angles, border_fraction*100, "o-", label=f"Outer {border} pixels")
    ax.axhline(settings["border_fraction_warning"]*100, color="gray", ls="--", label="Warning threshold")
    ax.set(title="Image-edge signal", xlabel="Gear rotation (deg)", ylabel="Fraction of captured intensity (%)")
    ax.legend(fontsize=8)
    for ax in axes.flat[1:]:
        ax.grid(alpha=0.2)
    fig.suptitle("Healthy gear · Rotation summary", fontsize=16)
    fig.savefig(output / "images" / "summary.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    for slot, ax in enumerate(axes.flat):
        if slot >= len(selected):
            ax.axis("off")
            continue
        index = selected[slot]
        ax.imshow(intensity[index], cmap="inferno", vmin=0, vmax=display_max, extent=extent, aspect="auto")
        relative = energy[index]/energy[nominal] if energy[nominal] > 0 else float("nan")
        ax.set_title(f"Gear {angles[index]:+.1f}° | E/E(0°) = {relative:.3f}")
        ax.set(xlabel="Horizontal angle (deg)", ylabel="Vertical angle (deg)")
    fig.suptitle("Healthy gear captures", fontsize=16)
    fig.savefig(output / "images" / "captures.png", dpi=150)
    plt.close(fig)
    from PIL import ImageDraw
    animation = config.get("animation", {})
    fps = animation.get("fps")
    gif_width = animation.get("gif_width_px", 640)
    gif_height = round(gif_width*448/640)
    # A single fixed palette avoids per-frame color shifts and keeps the GIF
    # compact. MP4 source PNGs retain the full display-color precision.
    palette = (plt.cm.inferno(np.linspace(0, 1, 63))[:, :3]*255).astype(np.uint8).tolist()
    palette[0] = [0, 0, 0]
    palette += [[255, 255, 255]]+[[0, 0, 0]]*192
    palette = [value for color in palette for value in color]
    animated = []
    for i, values in enumerate(intensity):
        rgb = (plt.cm.inferno(np.clip(values/display_max, 0, 1))[..., :3]*255).astype(np.uint8)
        frame = Image.new("RGB", (640, 448), "black")
        frame.paste(Image.fromarray(rgb).resize((640, 400)), (0, 24))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 6), f"Healthy gear | {angles[i]:06.1f} deg", fill="white")
        draw.text((8, 431), f"Laser line: {config['laser']['surface_width_fwhm_mm']:g} mm | Pupil distance: {camera['patch_to_pupil_mm']:g} mm | Infinity focus", fill="white")
        frame.save(output / "frames" / f"frame_{i:03d}.png")
        gray = Image.fromarray(np.clip(values/display_max, 0, 1).astype(np.float32)).resize((gif_width, round(gif_width*400/640)), Image.Resampling.BILINEAR)
        indexed = Image.fromarray(np.rint(np.asarray(gray)*62).astype(np.uint8)).convert("P")
        indexed.putpalette(palette)
        compact = Image.new("P", (gif_width, gif_height), 0)
        compact.putpalette(palette)
        compact.paste(indexed, (0, round(gif_width*24/640)))
        label = ImageDraw.Draw(compact)
        label.text((3, 0), f"{angles[i]:05.1f} deg | fixed laser + camera", fill=63)
        label.text((3, gif_height-12), "Infinity focus | one intensity scale", fill=63)
        animated.append(compact)
    animated[0].save(output / "animations" / "healthy_rotation.gif", save_all=True,
                     append_images=animated[1:], duration=round(1000/fps) if fps else 550,
                     loop=0, optimize=False, disposal=1)
    draw_setup(output, config, manifest)
    summary = {
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frames": len(frames), "raw_image_shape": [h, w, 3],
        "animation": {"fps": fps, "gif_width_px": gif_width,
                      "note": animation.get("note", "Playback speed selected for viewing.")},
        "signal_valid_frames": int(signal_valid.sum()), "edge_warning_frames": int(edge_warning.sum()),
        "valid_derivative_intervals": int(derivative_valid.sum()),
        "integrated_intensity_min": float(energy.min()), "integrated_intensity_max": float(energy.max()),
        "nominal_integrated_intensity": float(energy[nominal]),
        "nominal_mean_angle_deg": float(mean_angle[nominal]) if signal_valid[nominal] else None,
        "nominal_profile_std_deg": float(spread_angle[nominal]) if signal_valid[nominal] else None,
        "display_mapping": {"colormap": "inferno", "linear_vmin": 0, "linear_vmax": display_max,
                            "vmax_rule": "99.9th percentile over all pixels and all frames; raw EXRs unaffected",
                            "clipped_display_fraction": float((intensity > display_max).mean())},
        "signal_validity_rule": f"Numerical threshold: finite positive intensity at least {settings['minimum_relative_energy']:.1%} of sequence maximum.",
        "derivative_validity_rule": "Both adjacent signals valid and neither has an image-border warning.",
        "angle_convention": "Intensity-weighted direction relative to the nominal optical axis; positive follows increasing image column.",
        "edge_warning_note": "Sensor-border occupancy is a clipping heuristic. Pupil and off-sensor losses require separate characterization.",
        "intensity_note": "Raw linear red channel in relative units; physical intensity calibration is pending.",
    }
    if not presentation_only:
        (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(f"Updated presentation from {len(frames)} saved captures in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--presentation-only", action="store_true", help="Refresh figures and animation frames while preserving numerical result files.")
    args = parser.parse_args()
    analyze(args.output, presentation_only=args.presentation_only)
