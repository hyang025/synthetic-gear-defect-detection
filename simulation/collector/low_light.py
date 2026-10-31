"""Create shared-scale low-light displays from saved focal-plane power arrays."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".tools/matplotlib"))

import mitsuba as mi
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_sum(array, size):
    h, w = array.shape
    return array.reshape(h//size, size, w//size, size).sum(axis=(1, 3))


def label(case):
    return f"Ø{case['diameter_mm']:.2f} mm · {case['depth_um']} µm deep"


class DisplayMapping:
    def __init__(self, healthy, config):
        self.config = config
        self.peak = max(float(im.max()) for im in healthy)
        self.floor = self.peak*config["histogram_floor_relative_to_peak"]
        # Train a single histogram from positive healthy intensities in log space.
        # The common map is reused for each healthy/damaged panel.
        edges = np.linspace(np.log10(self.floor), np.log10(self.peak), config["histogram_bins"]+1)
        counts = np.zeros(config["histogram_bins"], dtype=np.int64)
        for im in healthy:
            signal = im[im > self.floor]
            counts += np.histogram(np.log10(signal), bins=edges)[0]
        self.edges = 10**edges
        self.cdf = np.r_[0, np.cumsum(counts)/counts.sum()]
        self.counts = counts

    def transform(self, array, mode):
        q = np.maximum(array, 0)/self.peak
        if mode == "log":
            span = self.config["reference_log_range"]
            return np.clip(np.log(np.maximum(q, 1/span)*span)/np.log(span), 0, 1)
        if mode == "gamma":
            return np.clip(q, 0, 1)**self.config["gamma"]
        if mode == "asinh":
            span = self.config["asinh_dynamic_range"]
            return np.clip(np.arcsinh(span*q)/np.arcsinh(span), 0, 1)
        if mode == "histogram":
            return np.interp(np.log10(np.maximum(array, self.floor)),
                             np.log10(self.edges), self.cdf, left=0, right=1)
        raise ValueError(mode)


def finish_axis(ax):
    ax.set(xlabel="Screen u (mm)", ylabel="Screen v (mm)",
           xticks=(-60, -30, 0, 30, 60), yticks=(-60, -30, 0, 30, 60))


def method_comparisons(out, cases, mapping, config):
    modes = (("log", "Log · 3000:1"), ("gamma", "Gamma · 0.25"),
             ("asinh", "Asinh shadow lift"), ("histogram", "Histogram equalized"))
    for case in cases:
        fig, axs = plt.subplots(2, 4, figsize=(14, 7.4), layout="constrained")
        for row, state in enumerate(("healthy", "damaged")):
            for col, (mode, title) in enumerate(modes):
                art = axs[row, col].imshow(mapping.transform(case[state], mode), origin="lower",
                                           extent=config["screen_extent_mm"], cmap="inferno", vmin=0, vmax=1,
                                           interpolation="nearest")
                axs[row, col].set_title(f"{state.capitalize()} · {title}")
                finish_axis(axs[row, col])
        fig.colorbar(art, ax=axs, shrink=0.65, label="Display level")
        fig.suptitle(f"{label(case)} · shared display mappings · 4 × 4 binning")
        fig.savefig(out/"images"/f"methods_{case['id']}.png", dpi=150)
        plt.close(fig)


def gallery(out, cases, mapping, config, outer=False):
    fig, axs = plt.subplots(len(cases), 3, figsize=(12, 4*len(cases)), layout="constrained")
    differences = []
    for case in cases:
        difference = case["difference"].copy()
        if outer:
            difference[~case["difference_keep"]] = np.nan
        differences.append(difference)
    limit = max(float(np.nanmax(np.abs(d))) for d in differences)
    norm = AsinhNorm(linear_width=limit/config["difference_asinh_dynamic_range"], vmin=-limit, vmax=limit)
    cmap = plt.colormaps["inferno"].copy()
    cmap.set_bad("#cccccc")
    diff_cmap = plt.colormaps["RdBu_r"].copy()
    diff_cmap.set_bad("#cccccc")
    for row, case in enumerate(cases):
        for col, state in enumerate(("healthy", "damaged")):
            display = mapping.transform(case[state], "histogram")
            if outer:
                display[~case["keep"]] = np.nan
            art = axs[row, col].imshow(display, origin="lower", extent=config["screen_extent_mm"],
                                      cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_title(f"{state.capitalize()} · {label(case)}")
        fig.colorbar(art, ax=axs[row, :2], shrink=0.7, label="Equalized display level")
        art = axs[row, 2].imshow(differences[row], origin="lower", extent=config["screen_extent_mm"],
                                 cmap=diff_cmap, norm=norm, interpolation="nearest")
        axs[row, 2].set_title(f"Dented − healthy · {case['angle_deg']:.3f}°")
        fig.colorbar(art, ax=axs[row, 2], shrink=0.7, label="Beam fraction / 8 × 8 bins")
        for ax in axs[row]:
            finish_axis(ax)
    title = "Outer wings · histogram equalized · gray band excluded" if outer else "Four dents · histogram equalized · shared scales"
    fig.suptitle(title)
    fig.savefig(out/"images"/("enhanced_outer_wings.png" if outer else "enhanced_comparison.png"), dpi=150)
    plt.close(fig)
    return limit


def main(path):
    started = time.perf_counter()
    config = json.loads(path.read_text())
    out = PROJECT/config["output"]
    out.mkdir(exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("Choose a new output folder to preserve the existing display set.")
    (out/"images").mkdir(exist_ok=True)
    write_json(out/"config.json", config)
    source_summary = PROJECT/config["peak_suppression_summary"]
    summary = json.loads(source_summary.read_text())
    rows = {r["case"]: r for r in summary["all_crossings"]
            if r["excluded_half_width_deg"] == config["outer_half_width_deg"]}
    sources = [Path(__file__), path, source_summary]
    cases = []
    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")
    for spec in config["cases"]:
        raw = {state: np.asarray(mi.Bitmap(str(PROJECT/spec[state]))).squeeze().astype(np.float64)
               for state in ("healthy", "damaged")}
        if raw["healthy"].shape != raw["damaged"].shape or raw["healthy"].ndim != 2:
            raise ValueError("Matched scalar screen arrays are required.")
        sources.extend(PROJECT/spec[state] for state in raw)
        case = {**spec, **{state: block_sum(raw[state], config["display_block_size"]) for state in raw}}
        case["difference"] = block_sum(raw["damaged"]-raw["healthy"], config["difference_block_size"])
        reference = rows[spec["id"]]
        for key, im in (("keep", case["healthy"]), ("difference_keep", case["difference"])):
            size = im.shape[0]
            step = (config["screen_extent_mm"][3]-config["screen_extent_mm"][2])/size
            y = config["screen_extent_mm"][2] + step*(np.arange(size)+0.5)
            theta = np.degrees(np.arctan(y/config["focal_length_mm"]))
            keep_y = ((theta < reference["mask_lower_edge_deg"])
                      | (theta > reference["mask_upper_edge_deg"]))
            case[key] = np.broadcast_to(keep_y[:, None], im.shape).copy()
        cases.append(case)
    mapping = DisplayMapping([case["healthy"] for case in cases], config)
    np.savez_compressed(out/"display_mapping.npz", intensity_edges=mapping.edges,
                        cumulative_probability=mapping.cdf, healthy_histogram_counts=mapping.counts,
                        peak=np.array(mapping.peak), floor=np.array(mapping.floor))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    method_comparisons(out, cases, mapping, config)
    full_limit = gallery(out, cases, mapping, config)
    outer_limit = gallery(out, cases, mapping, config, outer=True)
    manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in sources},
                "healthy_peak_power_per_display_bin": mapping.peak,
                "histogram_floor_power_per_display_bin": mapping.floor,
                "full_difference_limit_power_per_bin": full_limit,
                "outer_difference_limit_power_per_bin": outer_limit,
                "histogram_training": "Positive intensities from the four healthy center captures; shared globally across all cases",
                "images": sorted(str(p.relative_to(out)) for p in (out/"images").glob("*.png")),
                "elapsed_seconds": time.perf_counter()-started,
                "new_ray_tracing": False}
    write_json(out/"manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=Path(__file__).with_suffix(".json"))
    main(parser.parse_args().config.resolve())
