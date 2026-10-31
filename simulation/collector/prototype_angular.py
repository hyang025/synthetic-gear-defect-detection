"""Compare encoder-matched angular profiles using capped healthy weights."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".tools/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth(profile, kernel):
    radius = len(kernel)//2
    padded = np.pad(profile, ((0, 0), (radius, radius)), mode="edge")
    result = sum(value*padded[:, offset:offset+profile.shape[1]]
                 for offset, value in enumerate(kernel))
    # Keep the original total energy after the finite-edge convolution.
    return result*(profile.sum(axis=1)/result.sum(axis=1))[:, None]


def normalize(profile):
    return profile/profile.sum(axis=1, keepdims=True)


def moments(profile, angle):
    normalized = normalize(profile)
    centroid = normalized@angle
    spread = np.sqrt(np.sum(normalized*(angle[None]-centroid[:, None])**2, axis=1))
    return centroid, spread


def draw_profiles(out, gears, angles, beta, widths, linear, weighted, kernel):
    for gear in gears:
        fig, axs = plt.subplots(3, 5, figsize=(16, 10), layout="constrained")
        for column, defect in enumerate(gear["defects"]):
            index = int(np.argmin(abs((angles-defect["crossing_angle_deg"]+180) % 360-180)))
            for row, data in enumerate((linear, weighted)):
                for case, color, style, label in (
                    ("healthy_train", "#247ba0", "-", "Healthy reference"),
                    ("healthy_test", "#777777", ":", "Healthy test"),
                    (gear["id"], "#d1495b", "--", "Damaged"),
                ):
                    axs[row, column].plot(beta, data[case][index]/widths,
                                          color=color, ls=style, lw=1, label=label)
                axs[row, column].set_ylabel("Normalized profile / degree")
            axs[0, column].set_title(
                f"{defect['id']} · Ø{defect['diameter_mm']:.2f} mm\n"
                f"{defect['depth_mm']*1000:.0f} µm · encoder {angles[index]:.3f}°",
                fontsize=10,
            )
            axs[1, column].set_title("Healthy weights · cap 20×", fontsize=10)
            axs[2, column].plot(beta, (linear[gear["id"]][index]-linear["healthy_train"][index])/widths,
                                color="#247ba0", lw=1, label="Linear damage − reference")
            axs[2, column].plot(beta, (weighted[gear["id"]][index]-weighted["healthy_train"][index])/widths,
                                color="#d1495b", lw=1, label="Weighted damage − reference")
            axs[2, column].plot(beta, (weighted["healthy_test"][index]-weighted["healthy_train"][index])/widths,
                                color="#777777", ls=":", lw=1, label="Weighted healthy − reference")
            axs[2, column].axhline(0, color="black", lw=0.6)
            axs[2, column].set_ylabel("Difference / degree")
            for ax in axs[:, column]:
                ax.set(xlim=(beta[0], beta[-1]), xlabel="Reflection angle v (°)")
                ax.grid(alpha=0.2)
        for row in range(3):
            axs[row, 0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{gear['id']} · defect-center angular profiles · three-bin smoothing")
        fig.savefig(out/"images"/f"angular_profiles_{gear['id']}.png", dpi=150)
        plt.close(fig)


def draw_weights(out, gears, angles, beta, weights):
    fig, axs = plt.subplots(3, 5, figsize=(15, 8), layout="constrained")
    for row, gear in enumerate(gears):
        for column, defect in enumerate(gear["defects"]):
            index = int(np.argmin(abs((angles-defect["crossing_angle_deg"]+180) % 360-180)))
            ax = axs[row, column]
            ax.plot(beta, weights[index], color="#247ba0", lw=1)
            ax.set(title=defect["id"], xlabel="Reflection angle v (°)", ylabel="Weight",
                   ylim=(0, 21), xlim=(beta[0], beta[-1]))
            ax.grid(alpha=0.2)
    fig.suptitle("Encoder-matched healthy weights · maximum 20×")
    fig.savefig(out/"images/angular_weights.png", dpi=150)
    plt.close(fig)


def draw_separation(out, rows):
    fig, axs = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    locations = np.arange(len(rows))
    for mode, offset, color in (("linear", -0.18, "#247ba0"), ("weighted", 0.18, "#d1495b")):
        axs[0].bar(locations+offset, [row[f"{mode}_passage_damage_to_healthy_ratio"] for row in rows],
                   width=0.36, color=color, label=mode.capitalize())
    axs[0].axhline(1, color="black", ls="--", lw=0.8)
    axs[0].set(title="Passage-window distance relative to healthy variation",
               ylabel="Damage / healthy profile distance")
    axs[0].legend()
    axs[1].plot(locations, [row["damage_distance_amplification"] for row in rows],
                "o-", color="#d1495b", label="Damage distance")
    axs[1].plot(locations, [row["healthy_distance_amplification"] for row in rows],
                "o-", color="#777777", label="Healthy distance")
    axs[1].set(title="Weighted / linear distance", ylabel="Amplification (×)")
    axs[1].legend()
    for ax in axs:
        ax.set(xticks=locations, xticklabels=[row["defect"] for row in rows])
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Angular weighting · fifteen defects · matched encoder windows")
    fig.savefig(out/"images/angular_separation.png", dpi=150)
    plt.close(fig)


def draw_rotation(out, gears, angles, distances):
    order = np.argsort(angles)
    fig, axs = plt.subplots(3, 2, figsize=(15, 10), layout="constrained")
    for row, gear in enumerate(gears):
        for column, mode in enumerate(("linear", "weighted")):
            ax = axs[row, column]
            healthy = np.maximum(distances[mode]["healthy_calibration"], distances[mode]["healthy_test"])
            ax.plot(angles[order], healthy[order], color="#777777", lw=0.8, label="Healthy deviations")
            ax.plot(angles[order], distances[mode][gear["id"]][order], color="#d1495b", lw=0.9,
                    label="Damaged deviations")
            for defect in gear["defects"]:
                ax.axvline(defect["crossing_angle_deg"], color="gray", ls=":", lw=0.7)
            ax.set(title=f"{gear['id']} · {mode}", xlabel="Encoder angle (°)",
                   ylabel="Normalized profile distance (%)", xlim=(0, 360))
            ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=9)
    fig.suptitle("Full rotation · angular-profile deviations · dotted defect crossings")
    fig.savefig(out/"images/angular_rotation.png", dpi=150)
    plt.close(fig)


def main(config_path):
    started = time.perf_counter()
    settings = json.loads(config_path.read_text())
    out = PROJECT/settings["run"]
    if (out/"angular_manifest.json").exists():
        raise FileExistsError("Preserve the existing angular analysis before a new run.")
    patchcore = json.loads((out/"patchcore_manifest.json").read_text())
    if patchcore["status"] != "complete":
        raise RuntimeError("Finish the requested PatchCore evaluation first.")
    manifest = json.loads((out/"manifest.json").read_text())
    defects = json.loads((out/"defects.json").read_text())
    screen = manifest["trace_config"]["screen"]
    focal_length = manifest["trace_config"]["collector"]["focal_length_mm"]
    angles = np.asarray(manifest["angles_deg"])
    if settings["axis"] != "v":
        raise ValueError("This configuration extracts the v profile by summing across u.")
    screen_edges = np.linspace(-screen["height_mm"]/2, screen["height_mm"]/2, screen["height_px"]+1)
    beta_edges = np.degrees(np.arctan(screen_edges/focal_length))
    beta = np.degrees(np.arctan((screen_edges[:-1]+screen_edges[1:])/2/focal_length))
    widths = np.diff(beta_edges)
    mi.set_variant("scalar_rgb")
    profiles, source_hashes = {}, {}
    for case in manifest["cases"]:
        selected = sorted([row for row in manifest["frames"] if row["case"] == case], key=lambda row: row["index"])
        if len(selected) != len(angles):
            raise ValueError(f"Encoder-grid size mismatch for {case}.")
        case_profiles = []
        for row in selected:
            source = out/row["raw"]
            array = np.asarray(mi.Bitmap(str(source))).squeeze().astype(np.float64)
            if array.shape != (screen["height_px"], screen["width_px"]):
                raise ValueError(f"Unexpected screen shape: {source}")
            if not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError(f"Finite nonnegative linear intensity required: {source}")
            case_profiles.append(array.sum(axis=1))
            source_hashes[row["raw"]] = digest(source)
        profiles[case] = np.stack(case_profiles)
        print(f"Linear v profiles: {case} · {len(selected)} captures", flush=True)
    kernel = np.asarray(settings["smoothing_kernel"])
    smoothed = {case: smooth(profile, kernel) for case, profile in profiles.items()}
    linear = {case: normalize(profile) for case, profile in smoothed.items()}
    reference = settings["reference_case"]
    density = linear[reference]/widths[None]
    peak = density.max(axis=1, keepdims=True)
    cap = settings["weight_cap"]
    weights = np.clip(peak/np.maximum(density, peak/cap), 1, cap)
    weighted_mass = {case: profile*weights for case, profile in smoothed.items()}
    weighted = {case: normalize(profile) for case, profile in weighted_mass.items()}
    distances = {mode: {case: 100*np.abs(profile-data[reference]).sum(axis=1)
                        for case, profile in data.items() if case != reference}
                 for mode, data in (("linear", linear), ("weighted", weighted))}
    raw_moments = {case: moments(profile, beta) for case, profile in profiles.items()}
    weighted_moments = {case: moments(profile, beta) for case, profile in weighted_mass.items()}
    feature_rows, feature_lookup = [], {}
    for case in manifest["cases"]:
        if case == reference:
            continue
        for index, angle in enumerate(angles):
            row = {"case": case, "index": index, "encoder_angle_deg": float(angle),
                   "linear_energy": float(profiles[case][index].sum()),
                   "linear_energy_change_percent": float(100*(profiles[case][index].sum()/profiles[reference][index].sum()-1)),
                   "linear_centroid_deg": float(raw_moments[case][0][index]),
                   "linear_centroid_shift_deg": float(raw_moments[case][0][index]-raw_moments[reference][0][index]),
                   "linear_spread_deg": float(raw_moments[case][1][index]),
                   "linear_spread_change_deg": float(raw_moments[case][1][index]-raw_moments[reference][1][index]),
                   "linear_profile_distance_percent": float(distances["linear"][case][index]),
                   "weighted_sum": float(weighted_mass[case][index].sum()),
                   "weighted_sum_change_percent": float(100*(weighted_mass[case][index].sum()/weighted_mass[reference][index].sum()-1)),
                   "weighted_centroid_deg": float(weighted_moments[case][0][index]),
                   "weighted_centroid_shift_deg": float(weighted_moments[case][0][index]-weighted_moments[reference][0][index]),
                   "weighted_spread_deg": float(weighted_moments[case][1][index]),
                   "weighted_spread_change_deg": float(weighted_moments[case][1][index]-weighted_moments[reference][1][index]),
                   "weighted_profile_distance_percent": float(distances["weighted"][case][index])}
            feature_rows.append(row)
            feature_lookup[case, index] = row
    defect_rows = []
    for gear in defects["gears"]:
        for defect in gear["defects"]:
            distance = abs((angles-defect["crossing_angle_deg"]+180) % 360-180)
            index = int(distance.argmin())
            if distance[index] > 1e-7:
                raise ValueError(f"Missing geometric crossing capture: {defect['id']}")
            passage = distance <= defects["passage_half_width_deg"]+1e-7
            row = {"gear": gear["id"], "defect": defect["id"], "kind": defect["kind"],
                   "diameter_mm": defect["diameter_mm"], "depth_um": defect["depth_mm"]*1000,
                   "crossing_angle_deg": defect["crossing_angle_deg"], "center_frame_index": index,
                   "passage_frames": int(passage.sum())}
            center = feature_lookup[gear["id"], index]
            row.update({f"center_{key}": value for key, value in center.items()
                        if key not in ("case", "index", "encoder_angle_deg")})
            for mode in ("linear", "weighted"):
                damage = float(distances[mode][gear["id"]][passage].max())
                healthy = float(max(distances[mode][case][passage].max()
                                    for case in settings["healthy_comparison_cases"]))
                row[f"{mode}_passage_damage_distance_percent"] = damage
                row[f"{mode}_passage_healthy_distance_percent"] = healthy
                row[f"{mode}_passage_damage_to_healthy_ratio"] = damage/healthy
            row["damage_distance_amplification"] = row["weighted_passage_damage_distance_percent"]/row["linear_passage_damage_distance_percent"]
            row["healthy_distance_amplification"] = row["weighted_passage_healthy_distance_percent"]/row["linear_passage_healthy_distance_percent"]
            row["separation_ratio_change"] = row["weighted_passage_damage_to_healthy_ratio"]/row["linear_passage_damage_to_healthy_ratio"]
            defect_rows.append(row)
    write_json(out/"angular_config.json", settings)
    write_csv(out/"angular_features.csv", feature_rows)
    write_csv(out/"angular_defects.csv", defect_rows)
    saved = {"encoder_angle_deg": angles, "reflection_angle_deg": beta, "angular_bin_width_deg": widths,
             "reflection_angle_edges_deg": beta_edges, "healthy_weights": weights, "smoothing_kernel": kernel}
    for case in manifest["cases"]:
        saved[f"{case}_linear_mass"] = profiles[case]
        saved[f"{case}_smoothed_mass"] = smoothed[case]
        saved[f"{case}_linear_normalized"] = linear[case]
        saved[f"{case}_weighted_normalized"] = weighted[case]
    np.savez_compressed(out/"angular_arrays.npz", **saved)
    summary = {"method": "Encoder-conditioned inverse healthy angular density, capped at 20x",
               "axis": "v", "weight_cap": cap, "angles_preserved": True,
               "healthy_reference": reference, "healthy_comparison_cases": settings["healthy_comparison_cases"],
               "profile_bins": len(beta), "reflection_angle_range_deg": beta_edges[[0, -1]].tolist(),
               "profile_smoothing_kernel": kernel.tolist(), "defects": len(defect_rows),
               "separation_ratio_increased": sum(row["separation_ratio_change"] > 1 for row in defect_rows),
               "median_linear_damage_to_healthy_ratio": float(np.median([row["linear_passage_damage_to_healthy_ratio"] for row in defect_rows])),
               "median_weighted_damage_to_healthy_ratio": float(np.median([row["weighted_passage_damage_to_healthy_ratio"] for row in defect_rows])),
               "median_damage_distance_amplification": float(np.median([row["damage_distance_amplification"] for row in defect_rows])),
               "median_healthy_distance_amplification": float(np.median([row["healthy_distance_amplification"] for row in defect_rows])),
               "median_per_defect_separation_ratio_change": float(np.median([row["separation_ratio_change"] for row in defect_rows])),
               "comparison": "Maximum damage and healthy L1 deviations from healthy_train in the same predeclared passage window"}
    write_json(out/"angular_summary.json", summary)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    draw_profiles(out, defects["gears"], angles, beta, widths, linear, weighted, kernel)
    draw_weights(out, defects["gears"], angles, beta, weights)
    draw_separation(out, defect_rows)
    draw_rotation(out, defects["gears"], angles, distances)
    sources = [config_path, Path(__file__), out/"manifest.json", out/"defects.json", out/"patchcore_manifest.json"]
    write_json(out/"angular_manifest.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in sources},
        "raw_source_sha256": source_hashes, "captures": len(manifest["frames"]),
        "new_ray_tracing": False, "elapsed_seconds": time.perf_counter()-started})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    main(args.config.resolve())
