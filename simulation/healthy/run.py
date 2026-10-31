"""Render the healthy-only baseline; raw images precede separate analysis."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".tools" / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import drjit as dr
import mitsuba as mi
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "outputs" / "healthy_baseline")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.rglob("frame_*.exr")):
        raise SystemExit("Output already has rendered frames; choose a new --output directory.")
    (output / "raw").mkdir(exist_ok=True)
    config = json.loads(args.config.read_text())
    if "gear_angles_deg" not in config["inspection"]:
        rotation = config["inspection"]["rotation"]
        config["inspection"]["gear_angles_deg"] = np.arange(rotation["start_deg"], rotation["stop_deg"], rotation["step_deg"]).tolist()
    config["mesh_absolute"] = str((args.config.parent / config["mesh"]).resolve())
    params_path = (args.config.parent / config["gear_parameters"]).resolve()
    gear = json.loads(params_path.read_text())
    mi.set_variant(config["variant"])
    from scene import resolve_geometry, line_bitmap, build_scene

    started = time.perf_counter()
    geometry = resolve_geometry(config, gear)
    bitmap = line_bitmap(config, geometry)
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    (output / "gear_parameters.json").write_text(json.dumps(gear, indent=2)+"\n")
    mi.Bitmap(bitmap).write(str(output / "raw" / "laser_projection.exr"))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    sources = [args.config.resolve()] + [ROOT / name for name in ("run.py", "scene.py", "analyze.py")]
    sources += [ROOT.parent / "gear" / "machining_marks.py", Path(config["mesh_absolute"]), params_path]
    manifest = {
        "status": "rendering", "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "mitsuba": mi.__version__,
                        "drjit": dr.__version__, "numpy": np.__version__,
                        "variant": mi.variant(), "gpus": gpu.stdout.strip(),
                        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]},
        "source_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "geometry": geometry, "frames": [],
        "assumptions": [
            "Revised gear: nominal 0.7 mm mid-tooth top width perpendicular to the tooth length, based on the user's rough measurement. Other tooth geometry remains photo-estimated, not manufacturing geometry.",
            "12 mm f/2.2 replacement lens is hypothetical; 25 mm pupil distance and 30 degree incidence are chosen setup assumptions.",
            "OV9281-sized ideal rectilinear sensor; no measured lens distortion, transmission, or housing.",
            f"Laser: {config['laser']['surface_width_fwhm_mm']} mm reference-plane FWHM, {config['laser']['surface_length_mm']} mm length, {config['laser']['distance_mm']} mm to nominal patch; layout {config['laser'].get('layout', 'local crown')}.",
            "For diameter illumination, the fan plane passes through the rotation axis; width is specified on the horizontal front-disc plane. Surface relief and shadowing change the actual footprint.",
            f"Film {config['camera']['width_px']} x {config['camera']['height_px']}; effective pixel pitch {config['camera']['sensor_width_mm']/config['camera']['width_px']*1000:g} micrometers across the unchanged sensor size.",
            "RGB red projector with arbitrary fixed scale, not a calibrated 650 nm / 1 mW source.",
            "GGX roughconductor, alpha 0.05, material none with reflectance 0.55; not measured steel optical constants.",
            "Existing machining bump: estimated 0.1 mm pitch and assumed 1 micrometer amplitude. No true groove geometry.",
            "100 m focus plane approximates infinity. Camera and laser stay fixed; gear alone rotates about world Z.",
            "Only laser illumination; no ambient light, coherent speckle, diffraction, polarization, sensor noise, saturation, or motion blur.",
            "Low-signal and edge warnings are numerical heuristics, not physical detector thresholds.",
            "Healthy-only sequence; no damaged gear or pose-nuisance comparison, and no detection-limit claim.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    angles = config["inspection"]["gear_angles_deg"]
    print(f"{mi.__version__} / {mi.variant()}; output: {output}", flush=True)
    print(f"Patch (mm): {np.array(geometry['patch_m'])*1000}; pupil radius: {geometry['aperture_radius_m']*1000:.3f} mm", flush=True)
    print(f"Horizontal FOV: {geometry['camera_fov_x_deg']:.3f} deg; focus: {config['camera']['focus_distance_m']} m", flush=True)
    for index, angle in enumerate(angles):
        frame_start = time.perf_counter()
        scene = build_scene(config, gear, geometry, angle, bitmap)
        total = config["render"]["samples_per_pixel"]
        batch = config["render"]["samples_per_pass"]
        accum = np.zeros((config["camera"]["height_px"], config["camera"]["width_px"], 3), dtype=np.float64)
        completed = 0
        pass_index = 0
        seeds = []
        while completed < total:
            spp = min(batch, total-completed)
            seed = config["render"]["seed"] + index*10000 + pass_index
            pixels = np.asarray(mi.render(scene, spp=spp, seed=seed))
            if not np.all(np.isfinite(pixels)) or np.any(pixels < 0):
                raise RuntimeError(f"Invalid linear image values at angle {angle}.")
            accum += pixels*spp
            seeds.append(seed)
            completed += spp
            pass_index += 1
            if len(angles) <= 12 and (pass_index == 1 or completed % 64 == 0 or completed == total):
                print(f"Frame {index+1}/{len(angles)}, {angle:+.2f} deg: {completed}/{total} spp", flush=True)
        raw = (accum/total).astype(np.float32)
        filename = f"raw/frame_{index:03d}.exr"
        mi.Bitmap(raw).write(str(output / filename))
        record = {"index": index, "gear_angle_deg": angle, "file": filename,
                  "seeds": seeds, "seconds": time.perf_counter()-frame_start,
                  "linear_red_sum": float(raw[..., 0].sum(dtype=np.float64))}
        manifest["frames"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        if len(angles) <= 12 or index % 10 == 0 or index == len(angles)-1:
            print(f"Saved {index+1}/{len(angles)}: {angle:+.1f} deg, red sum {record['linear_red_sum']:.6g}; {record['seconds']:.1f} s", flush=True)
        del scene, pixels, accum
        dr.flush_malloc_cache()
    manifest["status"] = "rendered"
    manifest["render_seconds"] = time.perf_counter()-started
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    from analyze import analyze
    analyze(output)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_seconds"] = time.perf_counter()-started
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Completed healthy baseline in {manifest['total_seconds']:.1f} s: {output}", flush=True)


if __name__ == "__main__":
    main()
