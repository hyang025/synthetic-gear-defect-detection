"""Existing three-dent gear: full revolution and exact dent-center samples."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from run import ROOT, PROJECT, trace_batch
from rotation import setup, object_geometry, save_json
import drjit as dr
import mitsuba as mi
import numpy as np


def trace_frame(scene, bsdf, config, geometry, angle, first_seed):
    sampling = config["sampling"]
    count = sampling["batches"]
    film = np.zeros((config["screen"]["height_px"], config["screen"]["width_px"]))
    profiles = {"u": [], "v": []}
    powers, bounces = [], np.zeros(sampling["max_surface_bounces"])
    local = object_geometry(geometry, angle)
    for batch in range(count):
        im, pu, pv, power, bounce = trace_batch(scene, bsdf, config, first_seed+batch, local)
        film += im/count
        profiles["u"].append(pu)
        profiles["v"].append(pv)
        powers.append(power)
        bounces[:len(bounce)] += np.asarray(bounce)/count
    record = {"power": {key: float(np.mean([p[key] for p in powers])) for key in powers[0]},
              "collector_power_by_bounce": bounces.tolist(),
              "screen_power_standard_error": float(np.std([p["screen_power"] for p in powers], ddof=1)/np.sqrt(count))}
    return film, profiles, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"three_dents.json")
    parser.add_argument("--output", type=Path, default=PROJECT/"outputs/collector_three_dents_25mm")
    args = parser.parse_args()
    out = args.output.resolve()
    if (out/"manifest.json").exists():
        raise SystemExit(f"An existing run occupies {out}.")
    config = json.loads(args.config.read_text())
    mi.set_variant(config["variant"])
    started = time.perf_counter()
    scene, bsdf, geometry, gear = setup(config)
    healthy_config = copy.deepcopy(config)
    healthy_config["mesh"] = config["alignment_mesh"]
    healthy_scene, healthy_bsdf, _, _ = setup(healthy_config)
    reference = (ROOT/config["healthy_reference"]).resolve()
    reference_manifest = json.loads((reference/"manifest.json").read_text())
    with np.load(reference/"profiles.npz") as data:
        reference_profiles = {key: data[key] for key in data.files}
    metadata = json.loads((ROOT/config["dent_metadata"]).read_text())
    centers = [dent["nominal_laser_crossing_deg"] for dent in metadata["dents"]]
    reference_angles = reference_profiles["angle_deg"]
    angles = np.unique(np.r_[reference_angles, centers])
    config["extra_sample_angles_deg"] = centers
    config["animation"]["extra_hold_frames"] = {
        str(int(np.argmin(abs(angles-a)))): config["animation"]["hold_crossing_frames"] for a in centers}
    for folder in ("raw/healthy", "raw/damaged", "images", "frames", "animations"):
        (out/folder).mkdir(parents=True, exist_ok=True)
    save_json(out/"config.json", config)
    sources = [Path(__file__), args.config.resolve(), ROOT/"run.py", ROOT/"rotation.py",
               ROOT/"compare_three_dents.py", PROJECT/"healthy/scene.py", PROJECT/"gear/machining_marks.py",
               ROOT/config["mesh"], ROOT/config["alignment_mesh"], ROOT/config["dent_metadata"],
               reference/"manifest.json"]
    manifest = {"status": "rendering", "started_utc": datetime.now(timezone.utc).isoformat(),
                "variant": mi.variant(), "mitsuba_version": mi.__version__, "drjit_version": dr.__version__,
                "geometry": geometry, "gear_tooth_count": gear["supplied"]["tooth_count"],
                "rays_per_frame": config["sampling"]["batches"]*config["sampling"]["rays_per_batch"],
                "power_units": "Fraction of unit incident laser-beam power",
                "healthy_reference": str(reference.relative_to(PROJECT)),
                "healthy_reused_positions": len(reference_angles), "healthy_new_positions": len(centers),
                "source_sha256": {str(p.resolve().relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "frames": []}
    save_json(out/"manifest.json", manifest)
    profile_data = {case: {axis: [] for axis in ("u", "v")} for case in ("healthy", "damaged")}
    sampling = config["sampling"]
    for index, value in enumerate(angles):
        angle = float(value)
        frame_started = time.perf_counter()
        match = np.flatnonzero(abs(reference_angles-angle) < 1e-9)
        if len(match):
            source_index = int(match[0])
            first_seed = sampling["seed"]+source_index*sampling["batches"]
            source_path = reference/"raw"/f"frame_{source_index:03d}.exr"
            link = out/"raw/healthy"/f"frame_{index:03d}.exr"
            link.symlink_to(os.path.relpath(source_path, link.parent))
            h_record = reference_manifest["frames"][source_index]
            for axis in ("u", "v"):
                profile_data["healthy"][axis].append(reference_profiles[f"{axis}_batches"][source_index])
            healthy_source = {"kind": "saved_reference", "index": source_index}
        else:
            center_index = int(np.argmin(abs(np.asarray(centers)-angle)))
            first_seed = sampling["seed"]+(len(reference_angles)+center_index)*sampling["batches"]
            h_image, h_profile, h_record = trace_frame(healthy_scene, healthy_bsdf, config, geometry, angle, first_seed)
            mi.Bitmap(h_image.astype(np.float32)).write(str(out/"raw/healthy"/f"frame_{index:03d}.exr"))
            for axis in ("u", "v"):
                profile_data["healthy"][axis].append(h_profile[axis])
            healthy_source = {"kind": "new_paired_crossing", "dent_index": center_index}
        d_image, d_profile, d_record = trace_frame(scene, bsdf, config, geometry, angle, first_seed)
        mi.Bitmap(d_image.astype(np.float32)).write(str(out/"raw/damaged"/f"frame_{index:03d}.exr"))
        for axis in ("u", "v"):
            profile_data["damaged"][axis].append(d_profile[axis])
        record = {"index": index, "angle_deg": angle, "first_seed": int(first_seed),
                  "healthy_source": healthy_source, "healthy": h_record, "damaged": d_record,
                  "elapsed_seconds": time.perf_counter()-frame_started}
        manifest["frames"].append(record)
        if index == 0 or (index+1) % 10 == 0 or not len(match):
            manifest["elapsed_seconds"] = time.perf_counter()-started
            save_json(out/"manifest.json", manifest)
            delta = 100*(d_record["power"]["screen_power"]-h_record["power"]["screen_power"])/h_record["power"]["screen_power"]
            print(f"{index+1}/{len(angles)} | {angle:.6f} deg | {manifest['elapsed_seconds']:.1f} s | change {delta:+.5f}%", flush=True)
    for case in profile_data:
        np.savez_compressed(out/f"{case}_profiles.npz", angle_deg=angles,
                            **{f"{axis}_batches": np.asarray(profile_data[case][axis]) for axis in ("u", "v")})
    manifest["status"] = "rendered"
    manifest["render_seconds"] = time.perf_counter()-started
    save_json(out/"manifest.json", manifest)
    from compare_three_dents import analyze
    analyze(out)
    print("Three-dent comparison frames complete.", flush=True)


if __name__ == "__main__":
    main()
