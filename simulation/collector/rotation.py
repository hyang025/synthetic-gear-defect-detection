"""Full-gear rotation with a fixed line laser and ideal angular collector."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from run import PROJECT, ROOT, trace_batch
import drjit as dr
import mitsuba as mi
import numpy as np


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n")


def setup(config):
    """Reuse the outside-rim alignment and aim the short fan along the crown."""
    sys.path.insert(0, str(PROJECT/"healthy"))
    from scene import resolve_geometry, material, unit
    gear = json.loads((ROOT/config["gear_parameters"]).read_text())
    alignment = json.loads((ROOT/config["alignment_config"]).read_text())
    mesh = (ROOT/config["mesh"]).resolve()
    alignment["mesh_absolute"] = str((ROOT/config.get("alignment_mesh", config["mesh"])).resolve())
    alignment["camera"]["patch_to_pupil_mm"] = config["collector"]["distance_mm"]
    alignment["laser"]["distance_mm"] = config["laser"]["distance_mm"]
    alignment["laser"]["incidence_deg"] = config["laser"]["incidence_deg"]
    reference = resolve_geometry(alignment, gear)
    patch = np.asarray(reference["patch_m"])
    normal = np.asarray(reference["normal"])
    center = np.asarray(reference["camera_origin_m"])
    axis = unit(center-patch)
    s, e = gear["supplied"], gear["estimated"]
    t = alignment["inspection"]["radial_fraction"]
    span = s["outside_diameter"]/2-e["tooth_inner_radius"]
    radius = e["tooth_inner_radius"]+t*span
    sweep = math.radians(e["tooth_sweep_degrees"])
    theta = 2*math.pi*alignment["inspection"]["tooth_index"]/s["tooth_count"]+sweep*(0.65*t+0.35*t*t)
    dtheta = sweep*(0.65+0.70*t)
    tangent = np.array([span*math.cos(theta)-radius*math.sin(theta)*dtheta,
                        span*math.sin(theta)+radius*math.cos(theta)*dtheta,
                        e["outer_root_height"]-(s["total_thickness"]-e["tooth_height"])])
    tangent = unit(tangent-np.dot(tangent, normal)*normal)
    across = unit(np.cross(normal, tangent))
    u = unit(tangent-np.dot(tangent, axis)*axis)
    v = unit(np.cross(axis, u))
    geometry = {"patch_m": patch.tolist(), "normal": normal.tolist(),
                "laser_origin_m": reference["laser_origin_m"],
                "line_tangent": tangent.tolist(), "line_across": across.tolist(),
                "collector_center_m": center.tolist(), "collector_axis": axis.tolist(),
                "collector_u": u.tolist(), "collector_v": v.tolist()}
    if config.get("healthy_reference"):
        reference_dir = (ROOT/config["healthy_reference"]).resolve()
        reference_config = json.loads((reference_dir/"config.json").read_text())
        for key in ("variant", "rotation", "laser", "collector", "screen", "material", "sampling", "analysis"):
            if config[key] != reference_config[key]:
                raise RuntimeError(f"Paired configuration mismatch: {key}")
        geometry = json.loads((reference_dir/"manifest.json").read_text())["geometry"]
    bsdf = mi.load_dict(material(config, gear, 0))
    scene = mi.load_dict({"type": "scene", "gear": {
        "type": "obj", "filename": str(mesh), "to_world": mi.ScalarTransform4f.scale(0.001),
        "bsdf": bsdf}})
    return scene, bsdf, geometry, gear


def object_geometry(geometry, angle):
    # Inverse-rotate the complete optical layout for tracing in the fixed mesh.
    # This is equivalent to rotating the gear with the optics fixed in the lab.
    a = math.radians(-angle)
    rotation = np.array([[math.cos(a), -math.sin(a), 0],
                         [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return {key: (rotation@np.asarray(value)).tolist() for key, value in geometry.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"rotation.json")
    parser.add_argument("--output", type=Path, default=PROJECT/"outputs/collector_rotation_25mm")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.output.resolve()
    if (out/"manifest.json").exists():
        raise SystemExit(f"An existing run occupies {out}; choose a fresh run directory.")
    mi.set_variant(config["variant"])
    started = time.perf_counter()
    scene, bsdf, geometry, gear = setup(config)
    for folder in ("raw", "images", "frames", "animations"):
        (out/folder).mkdir(parents=True, exist_ok=True)
    raw = out/"raw"/config.get("raw_case", "")
    raw.mkdir(exist_ok=True)
    save_json(out/"config.json", config)
    r, sampling = config["rotation"], config["sampling"]
    angles = np.arange(r["start_deg"], r["stop_deg"], r["step_deg"])
    sources = [Path(__file__), ROOT/"run.py", ROOT/"rotation_analyze.py", args.config.resolve(),
               PROJECT/"healthy/scene.py", PROJECT/"gear/machining_marks.py",
               ROOT/config["mesh"], ROOT/config["gear_parameters"], ROOT/config["alignment_config"]]
    if config.get("healthy_reference"):
        sources.extend([ROOT/"compare_rotation.py", ROOT/config["dent_metadata"],
                        ROOT/config["healthy_reference"]/"manifest.json"])
    manifest = {
        "status": "rendering", "started_utc": datetime.now(timezone.utc).isoformat(),
        "variant": mi.variant(), "mitsuba_version": mi.__version__, "drjit_version": dr.__version__,
        "geometry": geometry, "gear_tooth_count": gear["supplied"]["tooth_count"],
        "power_units": "Fraction of unit incident laser-beam power",
        "rotation_implementation": "Inverse-rotated optical layout in fixed gear coordinates",
        "screen_axes": "u follows the projected tooth-length tangent; v completes the right-handed basis",
        "source_sha256": {str(p.resolve().relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "rays_per_frame": sampling["batches"]*sampling["rays_per_batch"],
        "frames": []}
    save_json(out/"manifest.json", manifest)
    all_u, all_v = [], []
    shape = (config["screen"]["height_px"], config["screen"]["width_px"])
    for index, angle in enumerate(angles):
        frame_started = time.perf_counter()
        image = np.zeros(shape)
        pu_batches, pv_batches, counters = [], [], []
        bounces = np.zeros(sampling["max_surface_bounces"])
        local = object_geometry(geometry, float(angle))
        for batch in range(sampling["batches"]):
            seed = sampling["seed"]+index*sampling["batches"]+batch
            im, pu, pv, count, bounce_power = trace_batch(scene, bsdf, config, seed, local)
            image += im/sampling["batches"]
            pu_batches.append(pu)
            pv_batches.append(pv)
            counters.append(count)
            bounces[:len(bounce_power)] += np.asarray(bounce_power)/sampling["batches"]
        all_u.append(pu_batches)
        all_v.append(pv_batches)
        mi.Bitmap(image.astype(np.float32)).write(str(raw/f"frame_{index:03d}.exr"))
        record = {"index": index, "angle_deg": float(angle),
                  "elapsed_seconds": time.perf_counter()-frame_started,
                  "power": {key: float(np.mean([c[key] for c in counters])) for key in counters[0]},
                  "collector_power_by_bounce": bounces.tolist()}
        record["screen_power_standard_error"] = float(np.std([c["screen_power"] for c in counters], ddof=1)/math.sqrt(len(counters)))
        manifest["frames"].append(record)
        if index == 0 or (index+1) % 10 == 0:
            manifest["elapsed_seconds"] = time.perf_counter()-started
            save_json(out/"manifest.json", manifest)
            print(f"{index+1}/{len(angles)} angles | {angle:.0f} deg | {manifest['elapsed_seconds']:.1f} s | captured {record['power']['screen_power']:.6f}", flush=True)
    np.savez_compressed(out/"profiles.npz", angle_deg=angles,
                        u_batches=np.asarray(all_u), v_batches=np.asarray(all_v))
    manifest["status"] = "rendered"
    manifest["render_seconds"] = time.perf_counter()-started
    save_json(out/"manifest.json", manifest)
    if config.get("healthy_reference"):
        from compare_rotation import analyze
    else:
        from rotation_analyze import analyze
    analyze(out)
    manifest = json.loads((out/"manifest.json").read_text())
    manifest["status"] = "frames_complete"
    manifest["render_and_analysis_seconds"] = time.perf_counter()-started
    save_json(out/"manifest.json", manifest)
    print(f"Completed {len(angles)} angles in {manifest['render_and_analysis_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
