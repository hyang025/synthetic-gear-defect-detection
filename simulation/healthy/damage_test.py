"""Matched healthy/damaged captures around the three requested dent crossings."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = PROJECT / "outputs" / "damage_test"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".tools" / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import drjit as dr
import mitsuba as mi
import numpy as np


def main():
    dents = json.loads((OUT / "dents.json").read_text())
    if any(OUT.rglob("frame_*.exr")):
        raise SystemExit("Damage comparison captures already exist; preserve them before rerunning.")
    config = json.loads((ROOT / "full_revolution.json").read_text())
    config["laser"]["incidence_tangent_sign"] = -1
    config["laser"]["wavelength_nm"] = 650
    config["laser"]["wavelength_handling"] = "RGB-red approximation; no wavelength-resolved propagation or measured spectral reflectance."
    config["render"]["samples_per_pixel"] = 1024
    config["animation"] = {"fps": 10, "width_px": 960, "height_px": 288,
                           "filename": "damage_comparison.mp4",
                           "note": "Three local angle windows, not a full revolution or a measured motor rate."}
    offsets = np.round(np.arange(-1.5, 1.501, 0.1), 8)
    samples = [{"dent_id": dent["id"], "diameter_mm": dent["diameter_mm"],
                "depth_mm": dent["depth_mm"], "offset_deg": float(offset),
                "gear_angle_deg": dent["nominal_laser_crossing_deg"]+float(offset)}
               for dent in dents["dents"] for offset in offsets]
    config["inspection"].pop("rotation", None)
    config["inspection"]["gear_angles_deg"] = [sample["gear_angle_deg"] for sample in samples]
    meshes = {"healthy": PROJECT / "gear/model/damage_test/healthy_control.obj",
              "damaged": PROJECT / "gear/model/damage_test/damaged_gear.obj"}
    config["mesh_absolute"] = str(meshes["healthy"])
    gear = json.loads((PROJECT / "gear/model_parameters.json").read_text())
    mi.set_variant(config["variant"])
    from scene import resolve_geometry, line_bitmap, build_scene

    geometry = resolve_geometry(config, gear)
    bitmap = line_bitmap(config, geometry)
    (OUT / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    (OUT / "gear_parameters.json").write_text(json.dumps(gear, indent=2)+"\n")
    sources = [Path(__file__), ROOT / "scene.py", ROOT / "compare_damage.py",
               ROOT / "build_damage.py", PROJECT / "gear/machining_marks.py", *meshes.values()]
    manifest = {
        "status": "rendering", "started_utc": datetime.now(timezone.utc).isoformat(),
        "variant": mi.variant(), "mitsuba_version": mi.__version__, "geometry": geometry,
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "sampling": "Same independent seeds and samples for both cases at each pose. 31 poses per dent, 0.1 degree steps.",
        "frames": [],
        "assumptions": [
            "Camera outside rim, entrance pupil 25 mm from nominal crown; fixed 0.1 mm diameter-wide laser line.",
            "650 nm specified by user; same RGB-red geometric-optics approximation in both cases.",
            "12 mm f/2.2 ideal lens, infinity approximation at 100 m; relative intensity, not calibrated watts or electrons.",
            "All three dents are present in the damaged mesh. No raised lips or changed material properties.",
            "Revised nominal 0.7 mm mid-tooth crown. Both cases use the same healthy CAD base; validity and any repair are recorded in dents.json.",
            "Fine comparison meshing: 0.0005 mm deflection. Retessellation of cut crowns can leave small numerical differences.",
            "Batch-to-batch variability estimates rendering noise only, not physical measurement uncertainty.",
            "These local windows test the three dent passages; not another full-revolution animation.",
        ],
    }
    for case in meshes:
        (OUT / "raw" / case).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"Rendering {len(samples)} matched poses / {len(samples)*2} captures, 1024 spp per capture.", flush=True)
    for index, sample in enumerate(samples):
        frame_start = time.perf_counter()
        scenes = {}
        for case, mesh in meshes.items():
            cfg = {**config, "mesh_absolute": str(mesh)}
            scenes[case] = build_scene(cfg, gear, geometry, sample["gear_angle_deg"], bitmap)
        accum = {case: np.zeros((config["camera"]["height_px"], config["camera"]["width_px"], 3), dtype=np.float64) for case in meshes}
        totals, seeds = {case: [] for case in meshes}, []
        spp = config["render"]["samples_per_pass"]
        passes = config["render"]["samples_per_pixel"]//spp
        for pass_index in range(passes):
            seed = config["render"]["seed"]+index*10000+pass_index
            seeds.append(seed)
            for case, scene in scenes.items():
                pixels = np.asarray(mi.render(scene, spp=spp, seed=seed))
                if not np.isfinite(pixels).all() or np.any(pixels < 0):
                    raise RuntimeError(f"Non-finite or negative RGB values in {case}, sample {index}.")
                accum[case] += pixels/passes
                totals[case].append(float(pixels[..., 0].sum(dtype=np.float64)))
        paths = {}
        for case in meshes:
            paths[case] = f"raw/{case}/frame_{index:03d}.exr"
            mi.Bitmap(accum[case].astype(np.float32)).write(str(OUT / paths[case]))
        delta_batches = np.array(totals["damaged"])-np.array(totals["healthy"])
        healthy_energy = float(accum["healthy"][..., 0].sum())
        damaged_energy = float(accum["damaged"][..., 0].sum())
        record = {"index": index, **sample, "files": paths, "seeds": seeds,
                  "healthy_energy": healthy_energy, "damaged_energy": damaged_energy,
                  "delta_energy_percent": 100*(damaged_energy-healthy_energy)/healthy_energy,
                  "delta_energy_percent_mc_se": float(100*delta_batches.std(ddof=1)/np.sqrt(passes)/healthy_energy),
                  "seconds": time.perf_counter()-frame_start}
        manifest["frames"].append(record)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        if index % 5 == 0 or index == len(samples)-1:
            print(f"{index+1}/{len(samples)} | {sample['dent_id']} | offset {sample['offset_deg']:+.1f} deg | delta E {record['delta_energy_percent']:+.3f}% | {record['seconds']:.1f} s/pair", flush=True)
        del scenes, accum, pixels
        dr.flush_malloc_cache()
    manifest["status"] = "rendered"
    manifest["render_seconds"] = time.perf_counter()-started
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    from compare_damage import analyze
    analyze(OUT)
    manifest["status"] = "complete"
    manifest["total_seconds"] = time.perf_counter()-started
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Damage test complete: {manifest['total_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
