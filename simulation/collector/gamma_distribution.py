"""Measure the angular distributions of existing gamma 0.25 screen displays."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from low_light import PROJECT, block_sum, digest, label, write_json
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_stats(healthy, damaged, theta, keep):
    h = healthy.sum(axis=1)*keep
    d = damaged.sum(axis=1)*keep
    eh, ed = h.sum(), d.sum()
    hn, dn = h/eh, d/ed
    mh, md = np.sum(hn*theta), np.sum(dn*theta)
    sh = np.sqrt(np.sum(hn*(theta-mh)**2))
    sd = np.sqrt(np.sum(dn*(theta-md)**2))
    return {
        "healthy_sum": float(eh), "damaged_sum": float(ed),
        "sum_change_percent": float(100*(ed/eh-1)),
        "raw_profile_distance_percent": float(100*np.abs(d-h).sum()/eh),
        "normalized_profile_distance_percent": float(100*np.abs(dn-hn).sum()),
        "healthy_centroid_deg": float(mh), "damaged_centroid_deg": float(md),
        "centroid_shift_deg": float(md-mh),
        "healthy_spread_deg": float(sh), "damaged_spread_deg": float(sd),
        "spread_change_deg": float(sd-sh),
    }


def mask_geometry(spec, reference, shape, extent, focal_length, resolution):
    edges = np.linspace(extent[2], extent[3], shape[0]+1)
    theta = np.degrees(np.arctan(((edges[:-1]+edges[1:])/2)/focal_length))
    lower_angle, upper_angle = reference["mask_lower_edge_deg"], reference["mask_upper_edge_deg"]
    keep = (theta < lower_angle) | (theta > upper_angle)
    excluded = np.flatnonzero(~keep)
    lower = focal_length*np.tan(np.radians(lower_angle))
    upper = focal_length*np.tan(np.radians(upper_angle))
    actual_lower, actual_upper = edges[excluded[0]], edges[excluded[-1]+1]
    audit = {"case": spec["id"], "display_resolution": resolution,
             "mask_lower_angle_deg": lower_angle, "mask_upper_angle_deg": upper_angle,
             "ideal_lower_screen_mm": float(lower), "ideal_upper_screen_mm": float(upper),
             "gray_lower_screen_mm": float(actual_lower), "gray_upper_screen_mm": float(actual_upper),
             "lower_edge_error_mm": float(actual_lower-lower),
             "upper_edge_error_mm": float(actual_upper-upper),
             "display_bin_height_mm": float(edges[1]-edges[0])}
    return keep, audit, theta, np.diff(np.degrees(np.arctan(edges/focal_length)))


def draw_profiles(out, cases):
    fig, axs = plt.subplots(3, len(cases), figsize=(16, 10), layout="constrained")
    for col, case in enumerate(cases):
        theta, step = case["theta"], case["angular_bin_width_deg"]
        residuals = {}
        for row, space in enumerate(("linear", "gamma")):
            h = case[f"{space}_healthy"].sum(axis=1)
            d = case[f"{space}_damaged"].sum(axis=1)
            h, d = h/h.sum(), d/d.sum()
            ax = axs[row, col]
            ax.plot(theta, h/step, color="#247ba0", label="Healthy")
            ax.plot(theta, d/step, color="#d1495b", ls="--", label="Dented")
            ax.set(ylabel="Normalized profile / degree",
                   title=label(case) if row == 0 else "Gamma 0.25")
            residuals[space] = (d-h)/step
        ax = axs[2, col]
        ax.plot(theta, residuals["linear"], color="#247ba0", label="Linear")
        ax.plot(theta, residuals["gamma"], color="#d1495b", label="Gamma 0.25")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(title="Dented − healthy", ylabel="Normalized difference / degree")
        for ax in axs[:, col]:
            ax.set(xlim=(-55, 55), xlabel="Reflection angle v (°)")
            ax.grid(alpha=0.2)
            ax.axvline(case["mask_lower_angle_deg"], color="gray", ls=":", lw=0.8)
            ax.axvline(case["mask_upper_angle_deg"], color="gray", ls=":", lw=0.8)
    for row in range(3):
        axs[row, 0].legend(fontsize=9)
    fig.suptitle("Angular distribution · linear and gamma 0.25 · dotted mask boundaries")
    fig.savefig(out/"images/angular_distribution.png", dpi=150)
    plt.close(fig)


def draw_comparison(out, rows, cases):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    positions = np.arange(len(cases))
    labels = [f"Ø{c['diameter_mm']:.2f} mm\n{c['depth_um']} µm" for c in cases]
    lookup = {(r["case"], r["region"], r["space"]): r for r in rows}
    for col, region in enumerate(("full", "outer")):
        for space, offset, color in (("linear", -0.18, "#247ba0"), ("gamma", 0.18, "#d1495b")):
            data = [lookup[(c["id"], region, space)] for c in cases]
            axs[0, col].bar(positions+offset, [r["normalized_profile_distance_percent"] for r in data],
                            width=0.36, color=color, label="Linear" if space == "linear" else "Gamma 0.25")
            axs[1, col].bar(positions+offset, [r["sum_change_percent"] for r in data],
                            width=0.36, color=color)
        title = "Full screen" if region == "full" else "Outer region · ±20° excluded"
        axs[0, col].set(title=title, ylabel="Normalized profile distance (%)")
        axs[1, col].set(ylabel="Sum change (%)")
        axs[1, col].axhline(0, color="gray", lw=0.7)
        for ax in axs[:, col]:
            ax.set(xticks=positions, xticklabels=labels)
            ax.grid(axis="y", alpha=0.2)
    axs[0, 0].legend()
    fig.suptitle("Four dents · angular shape and summed response")
    fig.savefig(out/"images/feature_comparison.png", dpi=150)
    plt.close(fig)


def main(path):
    started = time.perf_counter()
    config = json.loads(path.read_text())
    out = PROJECT/config["output"]
    out.mkdir(exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("Choose a new output directory to preserve the saved analysis.")
    (out/"images").mkdir(exist_ok=True)
    display_path = PROJECT/config["display_config"]
    display = json.loads(display_path.read_text())
    summary_path = PROJECT/display["peak_suppression_summary"]
    previous = json.loads(summary_path.read_text())
    boundaries = {r["case"]: r for r in previous["all_crossings"]
                  if r["excluded_half_width_deg"] == config["outer_half_width_deg"]}
    mapping_path = PROJECT/config["display_mapping"]
    with np.load(mapping_path) as mapping:
        reference = float(mapping["peak"])
    gamma = config["gamma"]
    if gamma <= 0 or gamma != display["gamma"]:
        raise ValueError("A positive gamma matching the saved display is required.")
    write_json(out/"config.json", {**config, "intensity_reference": reference})
    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")
    cases, rows, audits, recoveries, saved = [], [], [], [], {}
    sources = [path, Path(__file__), Path(__file__).with_name("low_light.py"), display_path, summary_path, mapping_path]
    for spec in display["cases"]:
        case = dict(spec)
        raw = {}
        for state in ("healthy", "damaged"):
            source = PROJECT/spec[state]
            sources.append(source)
            raw[state] = np.asarray(mi.Bitmap(str(source))).squeeze().astype(np.float64)
            if raw[state].ndim != 2 or np.any(raw[state] < 0) or not np.all(np.isfinite(raw[state])):
                raise ValueError("Finite nonnegative scalar power arrays are required.")
            linear = block_sum(raw[state], display["display_block_size"])
            q = linear/reference
            transformed = q**gamma
            recovered = reference*transformed**(1/gamma)
            case[f"linear_{state}"], case[f"gamma_{state}"] = linear, transformed
            positive = linear > 0
            recoveries.append({"case": spec["id"], "state": state,
                               "max_relative_recovery_error": float(np.max(np.abs(recovered[positive]/linear[positive]-1))),
                               "max_absolute_recovery_error": float(np.abs(recovered-linear).max()),
                               "relative_total_recovery_error": float(abs(recovered.sum()/linear.sum()-1)),
                               "existing_display_above_reference_bins": int(np.count_nonzero(q > 1)),
                               "max_gamma_value": float(transformed.max())})
            saved[f"{spec['id']}_linear_{state}"] = linear
            saved[f"{spec['id']}_gamma_{state}"] = transformed
            saved[f"{spec['id']}_recovered_{state}"] = recovered
        keep, audit, theta, angular_step = mask_geometry(
            spec, boundaries[spec["id"]], case["linear_healthy"].shape,
            display["screen_extent_mm"], display["focal_length_mm"], "128x128")
        audits.append(audit)
        _, diff_audit, _, _ = mask_geometry(
            spec, boundaries[spec["id"]], block_sum(raw["healthy"], display["difference_block_size"]).shape,
            display["screen_extent_mm"], display["focal_length_mm"], "64x64")
        audits.append(diff_audit)
        case.update(theta=theta, angular_bin_width_deg=angular_step, keep=keep,
                    mask_lower_angle_deg=audit["mask_lower_angle_deg"], mask_upper_angle_deg=audit["mask_upper_angle_deg"])
        saved[f"{spec['id']}_keep"] = keep
        for region, select in (("full", np.ones_like(keep)), ("outer", keep)):
            for space in ("linear", "gamma"):
                stats = profile_stats(case[f"{space}_healthy"], case[f"{space}_damaged"], theta, select)
                rows.append({"case": spec["id"], "diameter_mm": spec["diameter_mm"], "depth_um": spec["depth_um"],
                             "angle_deg": spec["angle_deg"], "region": region, "space": space,
                             "sum_units": "incident-beam fraction" if space == "linear" else "summed gamma levels",
                             **stats})
        cases.append(case)
    saved.update(reflection_angle_deg=theta, angular_bin_width_deg=angular_step,
                 gamma=np.array(gamma), intensity_reference=np.array(reference))
    np.savez_compressed(out/"gamma_arrays.npz", **saved)
    write_csv(out/"metrics.csv", rows)
    write_csv(out/"mask_geometry.csv", audits)
    write_csv(out/"recovery.csv", recoveries)
    summary = {"gamma": gamma, "intensity_reference": reference, "metrics": rows,
               "mask_geometry": audits, "recovery": recoveries,
               "maximum_mask_edge_error_mm": max(abs(r[k]) for r in audits for k in ("lower_edge_error_mm", "upper_edge_error_mm")),
               "maximum_relative_recovery_error": max(r["max_relative_recovery_error"] for r in recoveries),
               "existing_display_above_reference_bins": sum(r["existing_display_above_reference_bins"] for r in recoveries)}
    write_json(out/"summary.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    draw_profiles(out, cases)
    draw_comparison(out, rows, cases)
    manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in sources},
                "gamma": gamma, "cases": 4, "source_batches_per_capture": 16,
                "display_shape": [128, 128], "new_ray_tracing": False,
                "elapsed_seconds": time.perf_counter()-started}
    write_json(out/"manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=Path(__file__).with_suffix(".json"))
    main(parser.parse_args().config.resolve())
