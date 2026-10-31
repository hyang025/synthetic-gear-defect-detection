"""Evaluate healthy-calibrated peak exclusion on saved matched gear revolutions."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".tools" / "matplotlib"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values):
    return np.std(values, axis=1, ddof=1) / np.sqrt(values.shape[1])


def load_run(spec, settings):
    directory = PROJECT / spec["directory"]
    source_config = read_json(directory / "config.json")
    source_manifest = read_json(directory / "manifest.json")
    axis = settings["axis"]
    with np.load(PROJECT / spec["healthy_profiles"]) as data:
        angles = data["angle_deg"]
        healthy = data[f"{axis}_batches"]
    with np.load(PROJECT / spec["damaged_profiles"]) as data:
        damaged_angles = data["angle_deg"]
        damaged = data[f"{axis}_batches"]
    if not np.array_equal(angles, damaged_angles) or healthy.shape != damaged.shape:
        raise ValueError("Paired encoder positions and batch shapes must match.")
    if source_manifest["status"] != "complete":
        raise ValueError(f"Source run requires completion: {directory}")
    split = settings["calibration_batches"]
    if not 0 < split < healthy.shape[1] - 1:
        raise ValueError("Calibration and evaluation each require saved batches.")
    half = source_config["analysis"]["angle_half_range_deg"]
    bins = source_config["analysis"]["angle_bins"]
    if bins != healthy.shape[-1]:
        raise ValueError("Angular bin count differs from the source configuration.")
    step = 2 * half / bins
    theta = np.linspace(-half + step/2, half - step/2, bins)
    calibration = healthy[:, :split].mean(axis=1)
    smooth_bins = settings["healthy_peak_smoothing_bins"]
    smoothed = np.array([np.convolve(p, np.ones(smooth_bins)/smooth_bins, mode="same")
                         for p in calibration])
    centers = theta[np.argmax(smoothed, axis=1)]
    hb, db = healthy[:, split:], damaged[:, split:]
    h, d = hb.mean(axis=1), db.mean(axis=1)
    ehb, edb = hb.sum(axis=-1), db.sum(axis=-1)
    eh, ed = ehb.mean(axis=1), edb.mean(axis=1)
    masks, all_rows = [], []
    for width in settings["excluded_half_width_deg"]:
        keep = (np.abs(theta[None, :] - centers[:, None]) > width + 1e-8
                if width else np.ones(h.shape, dtype=bool))
        thb = np.sum(hb * keep[:, None, :], axis=-1)
        tdb = np.sum(db * keep[:, None, :], axis=-1)
        th, td = thb.mean(axis=1), tdb.mean(axis=1)
        if np.any(th <= 0) or np.any(td <= 0):
            raise ValueError("Retained signal is zero; the relative comparison is undefined.")
        delta_b = tdb - thb
        delta = td - th
        delta_se = standard_error(delta_b)
        ratio = td / th
        ratio_se = standard_error(tdb - ratio[:, None]*thb) / th
        normalized_delta = delta / eh
        normalized_delta_se = standard_error(delta_b - normalized_delta[:, None]*ehb) / eh
        fh, fd = th/eh, td/ed
        fraction_se = standard_error((tdb - fd[:, None]*edb)/ed[:, None]
                                     - (thb - fh[:, None]*ehb)/eh[:, None])
        coefficient = np.abs(delta)/np.sqrt(th + td)
        rows = []
        for i, angle in enumerate(angles):
            rows.append({
                "run": spec["id"], "angle_deg": float(angle),
                "excluded_half_width_deg": float(width),
                "healthy_peak_v_deg": float(centers[i]),
                "mask_lower_edge_deg": float(centers[i] - width - step/2) if width else None,
                "mask_upper_edge_deg": float(centers[i] + width + step/2) if width else None,
                "healthy_total_beam_fraction": float(eh[i]),
                "damaged_total_beam_fraction": float(ed[i]),
                "healthy_retained_beam_fraction": float(th[i]),
                "damaged_retained_beam_fraction": float(td[i]),
                "healthy_retained_percent": float(100*fh[i]),
                "damaged_retained_percent": float(100*fd[i]),
                "retained_signal_change_percent": float(100*(ratio[i]-1)),
                "change_se_percentage_points": float(100*ratio_se[i]),
                "absolute_change_beam_fraction": float(delta[i]),
                "absolute_change_beam_fraction_se": float(delta_se[i]),
                "change_percent_healthy_total": float(100*normalized_delta[i]),
                "change_percent_healthy_total_se": float(100*normalized_delta_se[i]),
                "retained_fraction_change_percentage_points": float(100*(fd[i]-fh[i])),
                "retained_fraction_change_se_percentage_points": float(100*fraction_se[i]),
                "poisson_sensitivity_coefficient": float(coefficient[i]),
            })
        masks.append(keep)
        all_rows.append(rows)
    crossings = []
    for crossing in spec["crossings"]:
        index = int(np.argmin(abs(angles - crossing["angle_deg"])))
        if abs(angles[index] - crossing["angle_deg"]) > 1e-6:
            raise ValueError("The exact requested dent crossing is absent from the saved run.")
        baseline = all_rows[0][index]["poisson_sensitivity_coefficient"]
        rows = [{"case": crossing["id"], "diameter_mm": crossing["diameter_mm"],
                 "depth_um": crossing["depth_um"], **width_rows[index],
                 "poisson_sensitivity_relative_to_full": width_rows[index]["poisson_sensitivity_coefficient"]/baseline}
                for width_rows in all_rows]
        crossings.append({**crossing, "index": index, "rows": rows})
    return {"spec": spec, "config": source_config, "angles": angles, "theta": theta,
            "step": step, "centers": centers, "calibration": calibration,
            "healthy": h, "damaged": d, "difference_se": standard_error(db - hb),
            "masks": np.asarray(masks), "rows": all_rows, "crossings": crossings,
            "evaluation_batches": hb.shape[1], "evaluation_rays": hb.shape[1]*source_config["sampling"]["rays_per_batch"]}


def case_label(crossing):
    return f"Ø{crossing['diameter_mm']:.2f} mm · {crossing['depth_um']} µm deep"


def draw_profiles(out, runs, settings):
    display = settings["excluded_half_width_deg"].index(settings["display_half_width_deg"])
    cases = [(run, crossing) for run in runs for crossing in run["crossings"]]
    cases.sort(key=lambda item: (item[1]["diameter_mm"], item[1]["depth_um"]))
    fig, axs = plt.subplots(2, len(cases), figsize=(16, 7), layout="constrained")
    for col, (run, crossing) in enumerate(cases):
        i = crossing["index"]
        x, step = run["theta"], run["step"]
        h, d = run["healthy"][i]/step, run["damaged"][i]/step
        keep = run["masks"][display, i]
        row = crossing["rows"][display]
        ax = axs[0, col]
        ax.semilogy(x, h, color="#247ba0", label="Healthy")
        ax.semilogy(x, d, color="#d1495b", ls="--", label="Dented")
        ax.set(title=case_label(crossing), ylim=(1e-6, 0.05), ylabel="Beam fraction / degree")
        ax = axs[1, col]
        difference = (d-h)
        se = run["difference_se"][i]/step
        ax.plot(x, np.where(keep, difference, np.nan), color="#d1495b")
        ax.fill_between(x, difference-2*se, difference+2*se, where=keep,
                        color="#d1495b", alpha=0.22)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(title=f"Outer light {row['retained_signal_change_percent']:+.2f}%",
               ylabel="Dented − healthy / degree")
        for ax in axs[:, col]:
            ax.axvspan(row["mask_lower_edge_deg"], row["mask_upper_edge_deg"], color="#bbbbbb", alpha=0.35)
            ax.set(xlabel="Reflection angle v (°)", xlim=(-50, 50))
            ax.grid(alpha=0.2)
    axs[0, 0].legend(loc="lower center")
    axs[1, 0].plot([], [], color="#d1495b", alpha=0.4, lw=5, label="±2 paired MC SE")
    axs[1, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Healthy-peak exclusion · ±{settings['display_half_width_deg']}° · gray band excluded")
    fig.savefig(out/"images/angular_wings.png", dpi=150)
    plt.close(fig)


def draw_sweep(out, runs, settings):
    cases = [crossing for run in runs for crossing in run["crossings"]]
    cases.sort(key=lambda c: (c["diameter_mm"], c["depth_um"]))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = ("#1982c4", "#8a5a9e", "#e18a24", "#c43c39")
    widths = settings["excluded_half_width_deg"]
    for crossing, color in zip(cases, colors):
        rows = crossing["rows"]
        label = case_label(crossing)
        values = np.array([r["retained_signal_change_percent"] for r in rows])
        error = 2*np.array([r["change_se_percentage_points"] for r in rows])
        axs[0, 0].errorbar(widths, values, yerr=error, marker="o", color=color, label=label, capsize=3)
        axs[0, 1].plot(widths, [r["healthy_retained_percent"] for r in rows], marker="o", color=color)
        axs[1, 0].plot(widths, [r["change_percent_healthy_total"] for r in rows], marker="o", color=color)
        axs[1, 1].plot(widths, [r["poisson_sensitivity_relative_to_full"] for r in rows], marker="o", color=color)
    axs[0, 0].set(title="Retained-light contrast · ±2 paired MC SE", ylabel="Dented − healthy (% of retained healthy)")
    axs[0, 0].legend(fontsize=9)
    axs[0, 1].set(title="Healthy light retained", ylabel="Fraction of full signal (%)", yscale="log")
    axs[1, 0].set(title="Absolute signal change", ylabel="Dented − healthy (% of full healthy)")
    axs[1, 1].set(title="Equal-exposure photon sensitivity", ylabel="Relative to full-profile energy score")
    axs[1, 1].axhline(1, color="gray", ls=":")
    for ax in axs.flat:
        ax.set(xticks=widths, xticklabels=["Full", "5°", "10°", "15°", "20°"], xlabel="Excluded half-width")
        ax.grid(alpha=0.2)
    fig.suptitle("Peak-suppression sweep · four dent crossings")
    fig.savefig(out/"images/mask_sweep.png", dpi=150)
    plt.close(fig)


def draw_rotation(out, runs, settings):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), layout="constrained", sharex=True)
    colors = ("#333333", "#aaaaaa", "#e18a24", "#1982c4", "#8a5a9e")
    display = settings["excluded_half_width_deg"].index(settings["display_half_width_deg"])
    for row, run in enumerate(runs):
        for width, rows, color in zip(settings["excluded_half_width_deg"], run["rows"], colors):
            label = "Full" if width == 0 else f"±{width}° excluded"
            axs[row, 0].plot(run["angles"], [r["retained_signal_change_percent"] for r in rows],
                             color=color, label=label, lw=1.1)
        for idx in (0, display):
            width = settings["excluded_half_width_deg"][idx]
            rows = run["rows"][idx]
            axs[row, 1].plot(run["angles"], [r["change_percent_healthy_total"] for r in rows],
                            color=colors[idx], label="Full" if width == 0 else f"±{width}° excluded")
        for crossing in run["crossings"]:
            for ax in axs[row]:
                ax.axvline(crossing["angle_deg"], color="gray", lw=0.6, ls=":")
        name = "Original three dents" if run["spec"]["id"] == "three_dents" else "Ø0.10 mm · 25 µm dent"
        axs[row, 0].set(title=f"{name} · relative contrast", ylabel="Change (% of retained healthy)")
        axs[row, 1].set(title=f"{name} · absolute change", ylabel="Change (% of full healthy)")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    axs[0, 1].legend(fontsize=9)
    for ax in axs.flat:
        ax.set(xlim=(0, 360), xlabel="Encoder angle (°)")
        ax.grid(alpha=0.2)
    fig.suptitle("Full-revolution peak suppression · matched encoder angles")
    fig.savefig(out/"images/rotation_signals.png", dpi=150)
    plt.close(fig)


def main(config_path):
    started = time.perf_counter()
    settings = read_json(config_path)
    out = PROJECT/settings["output"]
    out.mkdir(exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("Choose a new output folder to preserve this completed experiment.")
    (out/"images").mkdir(exist_ok=True)
    write_json(out/"config.json", settings)
    sources = [Path(__file__), config_path]
    for spec in settings["runs"]:
        sources.extend([PROJECT/spec["healthy_profiles"], PROJECT/spec["damaged_profiles"],
                        PROJECT/spec["directory"]/"config.json", PROJECT/spec["directory"]/"manifest.json"])
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "method": "Post-capture angular masking of saved Mitsuba ray-weight histograms",
                "source_sha256": {str(p.relative_to(PROJECT)): sha256(p) for p in sources}}
    write_json(out/"manifest.json", manifest)
    runs = [load_run(spec, settings) for spec in settings["runs"]]
    print("Loaded both revolutions; evaluating four dent crossings.", flush=True)
    all_rows = [row for run in runs for rows in run["rows"] for row in rows]
    crossing_rows = [row for run in runs for crossing in run["crossings"] for row in crossing["rows"]]
    write_csv(out/"metrics.csv", all_rows)
    write_csv(out/"crossing_metrics.csv", crossing_rows)
    arrays = {}
    for run in runs:
        name = run["spec"]["id"]
        for key in ("angles", "theta", "centers", "calibration", "healthy", "damaged", "difference_se", "masks"):
            arrays[f"{name}_{key}"] = run[key]
    np.savez_compressed(out/"profiles_and_masks.npz", **arrays)
    selected = [row for row in crossing_rows if row["excluded_half_width_deg"] == settings["display_half_width_deg"]]
    summary = {"axis": settings["axis"], "display_half_width_deg": settings["display_half_width_deg"],
               "runs": [{"id": r["spec"]["id"], "encoder_positions": len(r["angles"]),
                         "evaluation_batches": r["evaluation_batches"], "evaluation_rays_per_angle": r["evaluation_rays"],
                         "healthy_peak_min_deg": float(r["centers"].min()),
                         "healthy_peak_max_deg": float(r["centers"].max())} for r in runs],
               "display_crossings": selected, "all_crossings": crossing_rows}
    write_json(out/"summary.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 120})
    draw_profiles(out, runs, settings)
    draw_sweep(out, runs, settings)
    draw_rotation(out, runs, settings)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter()-started,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    processed_encoder_positions=sum(len(r["angles"]) for r in runs),
                    mask_settings_per_angle=len(settings["excluded_half_width_deg"]),
                    analysis_rows=len(all_rows), crossing_rows=len(crossing_rows),
                    new_ray_tracing=False)
    write_json(out/"manifest.json", manifest)
    print(json.dumps({"display_crossings": selected, "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=Path(__file__).with_suffix(".json"))
    main(parser.parse_args().config.resolve())
