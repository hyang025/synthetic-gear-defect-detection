"""Trace the requested healthy, dented, and tilted ideal-collector cases."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".tools" / "matplotlib"))

import drjit as dr
import mitsuba as mi
import numpy as np

CASES = ("healthy", "dented", "tilted")


def grid_axis(half_extent, settings):
    fine = settings["fine_mesh_half_width_mm"]
    outer = np.linspace(-half_extent, half_extent,
                        math.ceil(2*half_extent/settings["coarse_mesh_step_mm"])+1)
    inner = np.linspace(-fine, fine, round(2*fine/settings["fine_mesh_step_mm"])+1)
    return np.unique(np.round(np.r_[outer[abs(outer) > fine], inner], 9))


def surface_mesh(config, case, bsdf):
    s = config["surface"]
    x = grid_axis(s["length_mm"]/2, s)
    y = grid_axis(s["width_mm"]/2, s)
    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx)
    if case == "dented":
        dent = config["dent"]
        rho2 = ((xx-dent["center_mm"][0])**2+(yy-dent["center_mm"][1])**2)/(dent["diameter_mm"]/2)**2
        z = -dent["depth_mm"]*np.maximum(0, 1-rho2)**2
    top = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))
    bottom = top.copy()
    bottom[:, 2] = -s["thickness_mm"]
    vertices = np.vstack((top, bottom))
    ny, nx = xx.shape
    ids = np.arange(nx*ny).reshape(ny, nx)
    a, b = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()
    c, d = ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    faces = np.vstack((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
    faces = np.vstack((faces, faces[:, ::-1]+len(top)))
    # Counterclockwise perimeter; the side triangles face out of the block.
    edge = np.r_[ids[0, :], ids[1:, -1], ids[-1, -2::-1], ids[-2:0:-1, 0]]
    next_edge = np.roll(edge, -1)
    faces = np.vstack((faces, np.column_stack((edge, edge+len(top), next_edge+len(top))),
                       np.column_stack((edge, next_edge+len(top), next_edge))))
    if case == "tilted":
        angle = math.radians(config["pose"]["healthy_tilt_x_deg"])
        rotation = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)],
                             [0, math.sin(angle), math.cos(angle)]])
        vertices = vertices@rotation.T
    properties = mi.Properties()
    properties["bsdf"] = bsdf
    mesh = mi.Mesh(f"{case}_surface", len(vertices), len(faces), properties,
                   has_vertex_normals=False, has_vertex_texcoords=False)
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float((vertices*0.001).astype(np.float32).ravel())
    params["faces"] = mi.UInt32(faces.astype(np.uint32).ravel())
    params.update()
    return mesh, {"vertices": len(vertices), "triangles": len(faces),
                  "minimum_z_mm": float(vertices[:, 2].min()),
                  "top_minimum_z_mm": float(z.min()), "normal_interpolation": "flat triangle normals"}


def sum_value(value, active=True):
    return float(dr.sum(mi.Float64(dr.select(active, value, 0)))[0])


def trace_batch(scene, bsdf, config, seed, geometry=None):
    """Emitter-origin power tracing with BSDF/PDF weighting and ray visibility."""
    n = config["sampling"]["rays_per_batch"]
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, n)
    length = config["laser"]["line_length_mm"]*0.001
    focus_length = config["sampling"]["central_proposal_length_mm"]*0.001
    fraction = config["sampling"]["central_proposal_fraction"]
    branch = sampler.next_1d() < fraction
    ux = sampler.next_1d()
    x = (ux-0.5)*dr.select(branch, focus_length, length)
    proposal = (1-fraction)/length+dr.select(abs(x) <= focus_length/2, fraction/focus_length, 0)
    source_weight = (1/length)/proposal
    u1 = dr.maximum(sampler.next_1d(), 1e-7)
    u2 = sampler.next_1d()
    sigma = config["laser"]["line_width_fwhm_mm"]*0.001/(2*math.sqrt(2*math.log(2)))
    y = sigma*dr.sqrt(-2*dr.log(u1))*dr.cos(2*math.pi*u2)
    incidence = math.radians(config["laser"]["incidence_deg"])
    s, c = math.sin(incidence), math.cos(incidence)
    origin = mi.Point3f(0, -s*config["laser"]["distance_mm"]*0.001,
                       c*config["laser"]["distance_mm"]*0.001)
    target = mi.Point3f(x, y, 0)
    if geometry is not None:
        origin = mi.Point3f(geometry["laser_origin_m"])
        patch = mi.Point3f(geometry["patch_m"])
        line_tangent = mi.Vector3f(geometry["line_tangent"])
        line_across = mi.Vector3f(geometry["line_across"])
        dr.make_opaque(origin, patch, line_tangent, line_across)
        target = patch+x*line_tangent+y*line_across
    ray = mi.Ray3f(origin, dr.normalize(target-origin))
    si = scene.ray_intersect(ray)
    active = si.is_valid()
    weight = source_weight
    collector_z = mi.Vector3f(0, s, c)
    collector_u = mi.Vector3f(1, 0, 0)
    collector_v = mi.Vector3f(0, c, -s)
    collector_center = mi.Point3f(collector_z*config["collector"]["distance_mm"]*0.001)
    if geometry is not None:
        collector_z = mi.Vector3f(geometry["collector_axis"])
        collector_u = mi.Vector3f(geometry["collector_u"])
        collector_v = mi.Vector3f(geometry["collector_v"])
        collector_center = mi.Point3f(geometry["collector_center_m"])
        dr.make_opaque(collector_z, collector_u, collector_v, collector_center)
    radius = config["collector"]["diameter_mm"]*0.0005
    screen = config["screen"]
    width, height = screen["width_px"], screen["height_px"]
    film = dr.zeros(mi.Float64, width*height)
    angles = config["analysis"]
    bins, half_range = angles["angle_bins"], angles["angle_half_range_deg"]
    angular_u = dr.zeros(mi.Float64, bins)
    angular_v = dr.zeros(mi.Float64, bins)
    counters = {"sampled_source_power": sum_value(source_weight)/n,
                "surface_incident_power": sum_value(source_weight, active)/n,
                "collector_power": 0.0, "screen_power": 0.0,
                "screen_clipped_power": 0.0, "escaped_power": 0.0,
                "depth_truncated_power": 0.0}
    bounce_power = []
    context = mi.BSDFContext(mi.TransportMode.Importance)
    epsilon = config["sampling"]["ray_offset_m"]
    for bounce in range(config["sampling"]["max_surface_bounces"]):
        if not bool(dr.any(active)[0]):
            break
        sample, response = bsdf.sample(context, si, sampler.next_1d(), sampler.next_2d(), active)
        outgoing = dr.normalize(si.to_world(sample.wo))
        weight = dr.select(active, weight*response[0], 0)
        active &= (sample.pdf > 0) & (weight > 0) & (dr.dot(outgoing, si.n) > 0)
        ray = mi.Ray3f(si.p+si.n*epsilon, outgoing)
        next_si = scene.ray_intersect(ray, active=active)
        dz = dr.dot(outgoing, collector_z)
        t = dr.dot(collector_center-ray.o, collector_z)/dr.select(abs(dz) > 1e-12, dz, 1)
        hit = ray.o+t*outgoing-collector_center
        accepted = active & (dz > 0) & (t > 0) & (dr.squared_norm(hit) <= radius**2)
        accepted &= ~next_si.is_valid() | (t < next_si.t)
        captured = sum_value(weight, accepted)/n
        counters["collector_power"] += captured
        bounce_power.append(captured)
        sx = config["collector"]["focal_length_mm"]*dr.dot(outgoing, collector_u)/dr.select(dz > 0, dz, 1)
        sy = config["collector"]["focal_length_mm"]*dr.dot(outgoing, collector_v)/dr.select(dz > 0, dz, 1)
        inside = accepted & (abs(sx) < screen["width_mm"]/2) & (abs(sy) < screen["height_mm"]/2)
        film_x = mi.UInt32(dr.clip(dr.floor((sx/screen["width_mm"]+0.5)*width), 0, width-1))
        film_y = mi.UInt32(dr.clip(dr.floor((sy/screen["height_mm"]+0.5)*height), 0, height-1))
        dr.scatter_reduce(dr.ReduceOp.Add, film, mi.Float64(weight)/n, film_y*width+film_x, inside)
        for axis, histogram in ((collector_u, angular_u), (collector_v, angular_v)):
            theta = dr.atan2(dr.dot(outgoing, axis), dz)*(180/math.pi)
            index = mi.UInt32(dr.clip(dr.floor((theta/half_range+1)*bins/2), 0, bins-1))
            dr.scatter_reduce(dr.ReduceOp.Add, histogram, mi.Float64(weight)/n, index, inside)
        counters["screen_power"] += sum_value(weight, inside)/n
        counters["screen_clipped_power"] += sum_value(weight, accepted & ~inside)/n
        counters["escaped_power"] += sum_value(weight, active & ~accepted & ~next_si.is_valid())/n
        active &= ~accepted & next_si.is_valid()
        si = next_si
    counters["depth_truncated_power"] = sum_value(weight, active)/n
    dr.eval(film, angular_u, angular_v)
    image = np.asarray(film).reshape(height, width).astype(np.float64)
    if not np.isfinite(image).all() or np.any(image < 0):
        raise RuntimeError("Screen accumulation contains invalid values.")
    return image, np.asarray(angular_u).astype(np.float64), np.asarray(angular_v).astype(np.float64), counters, bounce_power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT/"outputs/collector_ideal_25mm")
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    args = parser.parse_args()
    out = args.output.resolve()
    if (out/"manifest.json").exists():
        raise SystemExit(f"A run already exists at {out}. Choose a fresh output directory.")
    config = json.loads(args.config.read_text())
    mi.set_variant(config["variant"])
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("images", "raw"):
        (out/folder).mkdir(exist_ok=True)
    model_dir = ROOT/"model"
    model_dir.mkdir(exist_ok=True)
    material = config["material"]
    bsdf = mi.load_dict({"type": "roughconductor", "material": "none",
                         "distribution": material["distribution"], "alpha": material["alpha"],
                         "specular_reflectance": material["reflectance"]})
    scenes, mesh_records = {}, {}
    for case in CASES:
        mesh, record = surface_mesh(config, case, bsdf)
        model_file = model_dir/f"{case}.ply"
        if model_file.exists():
            raise SystemExit(f"Preserve existing mesh before building: {model_file}")
        mesh.write_ply(str(model_file))
        mesh_records[case] = {**record, "file": str(model_file.relative_to(PROJECT)),
                              "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest()}
        scenes[case] = mi.load_dict({"type": "scene", "surface": mesh})
    (out/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    manifest = {"status": "rendering", "started_utc": datetime.now(timezone.utc).isoformat(),
                "mitsuba_version": mi.__version__, "drjit_version": dr.__version__,
                "variant": mi.variant(), "geometry": mesh_records,
                "power_units": "Fraction of the configured unit incident beam power",
                "source_sampling": "Paired seeds; weighted mixture of uniform line and central proposal",
                "cases": list(CASES), "batches": [],
                "source_sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), ROOT/"analyze.py", args.config.resolve())}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    settings = config["sampling"]
    count = settings["batches"]
    shape = (config["screen"]["height_px"], config["screen"]["width_px"])
    sums = {case: np.zeros(shape) for case in CASES}
    profiles = {case: {"u": [], "v": []} for case in CASES}
    started = time.perf_counter()
    for batch in range(count):
        seed = settings["seed"]+batch
        record = {"index": batch, "seed": seed, "cases": {}}
        for case in CASES:
            image, pu, pv, counters, bounces = trace_batch(scenes[case], bsdf, config, seed)
            sums[case] += image/count
            profiles[case]["u"].append(pu)
            profiles[case]["v"].append(pv)
            record["cases"][case] = {**counters, "collector_power_by_bounce": bounces}
        manifest["batches"].append(record)
        if batch == 0 or (batch+1) % 8 == 0:
            manifest["elapsed_seconds"] = time.perf_counter()-started
            (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            print(f"{batch+1}/{count} paired batches | {manifest['elapsed_seconds']:.1f} s", flush=True)
    for case in CASES:
        # The EXR contains power per screen bin. Area conversion is explicit.
        mi.Bitmap(sums[case].astype(np.float32)).write(str(out/"raw"/f"{case}.exr"))
    np.savez_compressed(out/"profiles.npz", **{
        f"{case}_{axis}_batches": np.asarray(profiles[case][axis]) for case in CASES for axis in ("u", "v")})
    manifest["status"] = "rendered"
    manifest["render_seconds"] = time.perf_counter()-started
    manifest["rays_per_case"] = count*settings["rays_per_batch"]
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    from analyze import analyze
    analyze(out)
    manifest["status"] = "complete"
    manifest["total_seconds"] = time.perf_counter()-started
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Completed {manifest['rays_per_case']:,} rays/case in {manifest['total_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
