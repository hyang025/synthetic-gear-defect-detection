"""Healthy gear, fixed projected line, and infinity-focused finite pupil.

Select a Mitsuba variant before importing this module. All scene lengths are m;
the existing mesh and user-facing dimensional settings are in mm.
"""

import math
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gear"))
from machining_marks import machining_material

T = mi.ScalarTransform4f


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def vector(v):
    return np.array([float(v[i][0]) for i in range(3)])


def resolve_geometry(config, gear):
    """Find the nominal crown on the actual mesh, then reflect about its normal."""
    s, e, inspect = gear["supplied"], gear["estimated"], config["inspection"]
    t = inspect["radial_fraction"]
    radial_span = s["outside_diameter"] / 2 - e["tooth_inner_radius"]
    radius = e["tooth_inner_radius"] + t * radial_span
    sweep = math.radians(e["tooth_sweep_degrees"])
    theta = 2 * math.pi * inspect["tooth_index"] / s["tooth_count"] + sweep * (0.65*t + 0.35*t*t)
    x, y = radius * math.cos(theta), radius * math.sin(theta)
    probe = mi.load_dict({
        "type": "scene", "gear": {
            "type": "obj", "filename": config["mesh_absolute"],
            "to_world": T.scale(0.001),
        },
    })
    hit = probe.ray_intersect(mi.Ray3f([x*0.001, y*0.001, 0.030], [0, 0, -1]))
    if not bool(hit.is_valid()[0]):
        raise RuntimeError("The nominal tooth-crown probe missed the mesh.")
    patch, normal = vector(hit.p), unit(vector(hit.sh_frame.n))
    expected_z = s["total_thickness"] + t * (e["outer_root_height"] - (s["total_thickness"] - e["tooth_height"]))
    if abs(patch[2]*1000 - expected_z) > 0.1:
        raise RuntimeError("The selected patch is not on the expected tooth crown.")
    dtheta = sweep * (0.65 + 0.70*t)
    tangent = np.array([
        radial_span*math.cos(theta) - radius*math.sin(theta)*dtheta,
        radial_span*math.sin(theta) + radius*math.cos(theta)*dtheta,
        e["outer_root_height"] - (s["total_thickness"] - e["tooth_height"]),
    ])
    along = unit(tangent - np.dot(tangent, normal)*normal)
    across = unit(np.cross(along, normal))
    incidence = math.radians(config["laser"]["incidence_deg"])
    to_laser = normal*math.cos(incidence) + across*math.sin(incidence)
    to_camera = normal*math.cos(incidence) - across*math.sin(incidence)
    laser = patch + to_laser*config["laser"]["distance_mm"]*0.001
    camera = patch + to_camera*config["camera"]["patch_to_pupil_mm"]*0.001
    laser_target, laser_up = patch.copy(), along.copy()
    extra = {}
    if config["laser"].get("layout") == "diameter":
        # Put the entire fan plane through the gear axis. A tangent-following
        # short line would miss the bore and is not the requested arrangement.
        radial = unit([patch[0], patch[1], 0])
        transverse = unit(np.cross([0, 0, 1], radial))
        plane_normal = unit(normal - np.dot(normal, transverse)*transverse)
        radial_tangent = unit(np.cross(transverse, plane_normal))
        side = config["laser"].get("incidence_tangent_sign", 1)
        to_laser = plane_normal*math.cos(incidence) + side*radial_tangent*math.sin(incidence)
        to_camera = unit(2*np.dot(normal, to_laser)*normal-to_laser)
        laser = patch + to_laser*config["laser"]["distance_mm"]*0.001
        camera = patch + to_camera*config["camera"]["patch_to_pupil_mm"]*0.001
        laser_target = np.array([0, 0, (s["total_thickness"]-e["tooth_height"])*0.001])
        forward = unit(laser_target-laser)
        laser_up = unit(radial-np.dot(radial, forward)*forward)
        along = radial_tangent
        across = unit(to_laser-np.dot(to_laser, normal)*normal)
        distance = np.linalg.norm(laser_target-laser)
        ends = [laser_target+sign*radial*config["laser"]["surface_length_mm"]*0.0005 for sign in (-1, 1)]
        slopes = [np.dot(end-laser, laser_up)/np.dot(end-laser, forward) for end in ends]
        field_y = 2*max(abs(slope) for slope in slopes)*distance*1.05
        field_x = field_y*config["laser"]["texture_width_px"]/config["laser"]["texture_height_px"]
        extra = {
            "laser_layout": "diameter", "diameter_radial": radial.tolist(),
            "diameter_transverse": transverse.tolist(),
            "laser_reference_plane_z_m": float(laser_target[2]),
            "laser_axis_distance_m": float(distance),
            "projector_field_width_m": float(field_x), "projector_field_height_m": float(field_y),
            "projector_fov_deg": math.degrees(2*math.atan(field_x/(2*distance))),
            "laser_length_fan_deg": math.degrees(math.atan(max(slopes))-math.atan(min(slopes))),
            "laser_width_fwhm_fan_deg": math.degrees(2*math.atan(config["laser"]["surface_width_fwhm_mm"]*0.001/(2*distance))),
            "camera_to_axis_at_reference_plane_mm": float(np.linalg.norm(camera-laser_target)*1000),
            "laser_actual_incidence_deg": math.degrees(math.acos(np.dot(normal, to_laser))),
        }
    # These checks are scene assembly: do not silently aim through another tooth.
    for name, origin in (("laser", laser), ("camera", camera)):
        h = probe.ray_intersect(mi.Ray3f(origin.tolist(), unit(patch-origin).tolist()))
        if not bool(h.is_valid()[0]) or np.linalg.norm(vector(h.p)-patch) > 0.00005:
            raise RuntimeError(f"The nominal {name}-to-patch path is obstructed.")
    c, laser_cfg = config["camera"], config["laser"]
    return {
        "patch_m": patch.tolist(), "normal": normal.tolist(),
        "geometric_normal": unit(vector(hit.n)).tolist(),
        "line_tangent": along.tolist(), "across_tangent": across.tolist(),
        "camera_origin_m": camera.tolist(), "laser_origin_m": laser.tolist(),
        "laser_target_m": laser_target.tolist(), "laser_up": laser_up.tolist(),
        "camera_fov_x_deg": math.degrees(2*math.atan(c["sensor_width_mm"]/(2*c["physical_focal_length_mm"]))),
        "camera_fov_y_deg": math.degrees(2*math.atan(c["sensor_height_mm"]/(2*c["physical_focal_length_mm"]))),
        "aperture_radius_m": c["physical_focal_length_mm"]/(2*c["f_number"])*0.001,
        "pupil_half_angle_deg": math.degrees(math.atan(c["physical_focal_length_mm"]/(2*c["f_number"]*c["patch_to_pupil_mm"]))),
        "projector_fov_deg": math.degrees(2*math.atan(laser_cfg["projection_field_mm"]/(2*laser_cfg["distance_mm"]))),
        "laser_length_fan_deg": math.degrees(2*math.atan(laser_cfg["surface_length_mm"]/(2*laser_cfg["distance_mm"]))),
        "laser_width_fwhm_fan_deg": math.degrees(2*math.atan(laser_cfg["surface_width_fwhm_mm"]*math.cos(incidence)/(2*laser_cfg["distance_mm"]))),
        **extra,
    }


def line_bitmap(config, geometry=None):
    """Gaussian across the line, flat length with cosine-tapered ends.

    Width and length are specified on the nominal tangent plane. The cosine
    factor compensates for oblique projection in the narrow-width direction.
    Curvature, pose, and shadowing change the footprint on the real mesh.
    """
    l = config["laser"]
    if l.get("layout") == "diameter":
        # Calibrate the fan on the horizontal plane of the front disc. Its
        # center plane contains the rotation axis at every height. Surface
        # relief changes width slightly, but cannot move the fan off the axis.
        g = geometry
        width, height = l["texture_width_px"], l["texture_height_px"]
        x = ((np.arange(width)+0.5)/width-0.5)*g["projector_field_width_m"]
        y = ((np.arange(height)+0.5)/height-0.5)*g["projector_field_height_m"]
        x, y = np.meshgrid(x, y)
        origin = np.asarray(g["laser_origin_m"])
        forward = unit(np.asarray(g["laser_target_m"])-origin)
        up = np.asarray(g["laser_up"])
        local_x = unit(np.cross(up, forward))
        rays = forward*g["laser_axis_distance_m"]-x[..., None]*local_x-y[..., None]*up
        distance = (g["laser_reference_plane_z_m"]-origin[2])/rays[..., 2]
        points = origin+distance[..., None]*rays
        across = points@np.asarray(g["diameter_transverse"])*1000
        along = points@np.asarray(g["diameter_radial"])*1000
        intensity = np.exp(-4*math.log(2)*(across/l["surface_width_fwhm_mm"])**2)
        edge = np.clip((np.abs(along)-(l["surface_length_mm"]/2-l["end_transition_mm"]))/l["end_transition_mm"], 0, 1)
        intensity *= 0.5*(1+np.cos(math.pi*edge))*(distance > 0)
        return (intensity[..., None]*np.array(l["rgb"])).astype(np.float32)
    size = l["texture_resolution"]
    coordinates = ((np.arange(size)+0.5)/size-0.5)*l["projection_field_mm"]
    x, y = np.meshgrid(coordinates, coordinates)
    width = l["surface_width_fwhm_mm"]*math.cos(math.radians(l["incidence_deg"]))
    intensity = np.exp(-4*math.log(2)*(x/width)**2)
    edge = np.clip((np.abs(y)-(l["surface_length_mm"]/2-l["end_transition_mm"])) / l["end_transition_mm"], 0, 1)
    intensity *= 0.5*(1+np.cos(math.pi*edge))
    return (intensity[..., None]*np.array(l["rgb"])).astype(np.float32)


def material(config, gear, angle_deg):
    base = {
        "type": "roughconductor", "material": "none",
        "distribution": config["material"]["distribution"],
        "alpha": config["material"]["alpha"],
        "specular_reflectance": config["material"]["reflectance"],
    }
    if not config["material"]["machining_marks"]:
        return base
    result = machining_material(gear)
    result["bsdf_0"] = base
    result["bsdf_1"]["bsdf"] = base
    # Keep the existing bump field attached to the rotating gear, not the world.
    inverse = T.scale(1000) @ T.rotate([0, 0, 1], -angle_deg)
    result["weight"]["to_model"] = inverse
    result["bsdf_1"]["texture"]["to_model"] = inverse
    return result


def build_scene(config, gear, geometry, angle_deg, bitmap):
    c, r = config["camera"], config["render"]
    return mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": r["max_depth"]},
        "sensor": {
            "type": "thinlens",
            "to_world": T.look_at(origin=geometry["camera_origin_m"], target=geometry["patch_m"], up=geometry["line_tangent"]),
            "fov": geometry["camera_fov_x_deg"], "fov_axis": "x",
            "aperture_radius": geometry["aperture_radius_m"],
            "focus_distance": c["focus_distance_m"], "near_clip": 0.00001,
            "far_clip": 200.0,
            "sampler": {"type": "independent", "sample_count": r["samples_per_pass"]},
            "film": {
                "type": "hdrfilm", "width": c["width_px"], "height": c["height_px"],
                "pixel_format": "rgb", "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        },
        "gear": {
            "type": "obj", "filename": config["mesh_absolute"],
            "to_world": T.rotate([0, 0, 1], angle_deg) @ T.scale(0.001),
            "bsdf": material(config, gear, angle_deg),
        },
        "laser": {
            "type": "projector", "fov": geometry["projector_fov_deg"], "fov_axis": "x",
            "to_world": T.look_at(origin=geometry["laser_origin_m"], target=geometry["laser_target_m"], up=geometry["laser_up"]),
            "irradiance": {"type": "bitmap", "bitmap": mi.Bitmap(bitmap), "raw": True,
                           "filter_type": "bilinear", "wrap_mode": "clamp"},
            "scale": config["laser"]["projector_scale"],
        },
    })
