"""Illustrate the proposed outside-rim camera using the saved gear geometry.

The RGB overview labels the requested 650 nm laser and the recorded
optical positions, with schematic device housings.
"""

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".tools" / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")
from scene import resolve_geometry, unit, vector

T = mi.ScalarTransform4f
OUT = ROOT.parent / "outputs" / "setup_overview"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "images").mkdir(exist_ok=True)
BASE = ROOT.parent / "outputs" / "full_revolution"
CONFIG = json.loads((BASE / "config.json").read_text())
GEAR = json.loads((BASE / "gear_parameters.json").read_text())
CONFIG["laser"]["incidence_tangent_sign"] = -1
CONFIG["laser"]["wavelength_nm"] = 650
CONFIG["laser"]["wavelength_handling"] = "User-confirmed wavelength; RGB display approximation."
GEOMETRY = resolve_geometry(CONFIG, GEAR)
P = np.array(GEOMETRY["patch_m"])
C = np.array(GEOMETRY["camera_origin_m"])
L = np.array(GEOMETRY["laser_origin_m"])
N = np.array(GEOMETRY["normal"])
RADIAL = np.array(GEOMETRY["diameter_radial"])
Q = unit(C-P)
LASER_BACK = unit(L-np.array(GEOMETRY["laser_target_m"]))
RED, BLUE, GREEN, AMBER = "#bd303b", "#176dad", "#288059", "#bd7a15"


def diffuse(color):
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": color}}


def cylinder(start, end, radius, color):
    return {"type": "cylinder", "p0": np.asarray(start).tolist(), "p1": np.asarray(end).tolist(),
            "radius": radius, "bsdf": diffuse(color)}


def sphere(center, radius, color):
    return {"type": "sphere", "center": np.asarray(center).tolist(), "radius": radius,
            "bsdf": diffuse(color)}


def body_scene():
    # Device bodies are schematic markers. Pupil position and radius follow
    # the recorded optical model; mechanical clearance needs a separate check.
    return {
        "gear": {"type": "obj", "filename": CONFIG["mesh_absolute"], "to_world": T.scale(0.001),
                 "bsdf": {"type": "roughconductor", "material": "none", "alpha": 0.22,
                          "specular_reflectance": {"type": "rgb", "value": [0.5, 0.53, 0.56]}}},
        "camera_barrel": cylinder(C+Q*0.0003, C+Q*0.010, 0.0045, [0.025, 0.10, 0.18]),
        "camera_body": {
            "type": "cube", "to_world": T.look_at(origin=(C+Q*0.017).tolist(), target=(C+Q*0.027).tolist(), up=[0, 1, 0]) @ T.scale([0.009, 0.007, 0.007]),
            "bsdf": diffuse([0.055, 0.23, 0.40]),
        },
        "camera_pupil": {
            "type": "disk", "to_world": T.look_at(origin=C.tolist(), target=P.tolist(), up=[0, 1, 0]) @ T.scale(GEOMETRY["aperture_radius_m"]),
            "bsdf": diffuse([0.025, 0.055, 0.075]),
        },
        "laser_housing": cylinder(L+LASER_BACK*0.0003, L+LASER_BACK*0.022, 0.005, [0.20, 0.035, 0.045]),
        "laser_aperture": sphere(L, 0.0012, [0.9, 0.02, 0.015]),
        "inspection_marker": sphere(P+N*0.00025, 0.0004, [0.95, 0.40, 0.025]),
    }


def render_view(origin, target, half_width, width, height):
    pose = T.look_at(origin=origin, target=target, up=[0, 0, 1])
    scene = {
        "type": "scene", "integrator": {"type": "path", "max_depth": 5, "hide_emitters": True},
        "sensor": {"type": "orthographic", "to_world": pose @ T.scale(half_width),
                   "sampler": {"type": "independent", "sample_count": 64},
                   "film": {"type": "hdrfilm", "width": width, "height": height,
                            "pixel_format": "rgba", "rfilter": {"type": "box"}}},
        **body_scene(),
        "ambient": {"type": "constant", "radiance": 0.7},
        "key": {"type": "rectangle", "to_world": T.look_at(origin=[-0.08, -0.10, 0.38], target=[0, 0, 0.08], up=[0, 1, 0]) @ T.scale(0.13),
                "emitter": {"type": "area", "radiance": 5.0}},
        "fill": {"type": "rectangle", "to_world": T.look_at(origin=[0.18, 0.10, 0.20], target=[0, 0, 0.10], up=[0, 1, 0]) @ T.scale(0.1),
                 "emitter": {"type": "area", "radiance": 2.0}},
    }
    pixels = mi.render(mi.load_dict(scene), spp=64, seed=650)
    rgba = np.asarray(mi.Bitmap(pixels).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=True)).astype(float)
    rgb = np.clip(rgba[..., :3]+255*(1-rgba[..., 3:4]/255), 0, 255).astype(np.uint8)
    inverse = pose.inverse()

    def project(points):
        result = []
        for point in np.atleast_2d(points):
            local = inverse @ mi.ScalarPoint3f(point.tolist())
            result.append([width/2-float(local.x)*width/(2*half_width),
                           height/2-float(local.y)*width/(2*half_width)])
        return np.array(result)
    return rgb, project


def label(ax, xy, text, location, color, size=11):
    ax.annotate(text, xy=xy, xytext=location, textcoords="axes fraction",
                ha="left", va="center", color=color, fontsize=size,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3, "alpha": 0.9},
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1})


def arrow(ax, project, start, end, color, lw=2.0):
    a, b = project([start, end])
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw})


print("Rendering the outside-camera setup overview...", flush=True)
full, full_project = render_view([-0.23, -0.50, 0.28], [0.005, 0, 0.15], 0.095, 850, 1500)
detail, detail_project = render_view([-0.090, -0.14, 0.065], [0.012, 0, 0.021], 0.048, 1250, 1050)

# Draw the actual mesh-intersected center of the laser fan, with gaps at the
# bore and any other missed surfaces. The red guide is widened for visibility.
probe = mi.load_dict({"type": "scene", "gear": {"type": "obj", "filename": CONFIG["mesh_absolute"], "to_world": T.scale(0.001)}})
targets = np.array(GEOMETRY["laser_target_m"])+np.linspace(-0.024, 0.024, 600)[:, None]*RADIAL
directions = targets-L
directions /= np.linalg.norm(directions, axis=1)[:, None]
hit = probe.ray_intersect(mi.Ray3f(mi.Point3f(L.tolist()), mi.Vector3f(directions.T)))
footprint = np.asarray(hit.p).T+np.asarray(hit.n).T*0.00008
footprint[~np.asarray(hit.is_valid())] = np.nan

fig = plt.figure(figsize=(16, 11), facecolor="white")
left = fig.add_axes([0.025, 0.16, 0.39, 0.70])
right = fig.add_axes([0.44, 0.34, 0.54, 0.53])
for ax, image, project in ((left, full, full_project), (right, detail, detail_project)):
    ax.imshow(image)
    ax.axis("off")
    points = project(footprint)
    ax.plot(points[:, 0], points[:, 1], color=RED, lw=1.4)

ends = [np.array(GEOMETRY["laser_target_m"])+s*RADIAL*0.022 for s in (-1, 1)]
left.add_patch(Polygon(full_project([L, *ends]), color=RED, alpha=0.10))
arrow(left, full_project, L, P, RED, 1.5)
arrow(left, full_project, P, C, BLUE, 1.5)
label(left, full_project([L])[0], "650 nm laser\n300 mm", (0.05, 0.89), RED)
label(left, full_project([L*0.48+P*0.52])[0], "Laser fan", (0.36, 0.55), RED, 10)
label(left, full_project([C])[0], "Camera outside rim", (0.50, 0.13), BLUE)
label(left, full_project([[0, 0, 0.006]])[0], "Rotating gear\nØ44 mm", (0.04, 0.13), "#444444")
left.set_title("Full layout", loc="left", fontsize=13, pad=12)

to_laser = unit(L-P)
arrow(right, detail_project, P+to_laser*0.045, P, RED)
arrow(right, detail_project, P, C, BLUE)
arrow(right, detail_project, P, P+N*0.022, GREEN, 1.7)
label(right, detail_project([C])[0], "Camera pupil\n25 mm from tooth", (0.66, 0.75), BLUE)
label(right, detail_project([P+N*0.019])[0], "Surface normal", (0.05, 0.90), GREEN)
label(right, detail_project([P+to_laser*0.035])[0], "Incident ray", (0.02, 0.64), RED)
label(right, detail_project([P])[0], "Sloped tooth top", (0.54, 0.11), AMBER)
label(right, detail_project([footprint[150]])[0], "0.1 mm laser line", (0.01, 0.12), RED, 10)
right.set_title("Tooth view", loc="left", fontsize=13, pad=12)

# Small angle diagram uses the actual reflected directions, in the incidence
# plane, so 'opposite side' is tied to the tilted surface normal.
angle_ax = fig.add_axes([0.44, 0.115, 0.25, 0.19])
tangent = unit(to_laser-np.dot(to_laser, N)*N)
for direction, color, text in ((to_laser, RED, "laser"), (unit(C-P), BLUE, "camera")):
    xy = [np.dot(direction, tangent), np.dot(direction, N)]
    angle_ax.plot([0, xy[0]], [0, xy[1]], color=color, lw=2)
    angle_ax.text(xy[0], xy[1]+0.07, text, color=color, ha="center", fontsize=10)
angle_ax.plot([0, 0], [0, 1.0], "--", color=GREEN, lw=1)
angle_ax.plot([-0.8, 0.8], [0, 0], color="gray", lw=3)
for sign in (-1, 1):
    a = np.linspace(0, sign*math.pi/6, 30)
    angle_ax.plot(0.5*np.sin(a), 0.5*np.cos(a), color="#555555", lw=0.8)
    angle_ax.text(sign*0.18, 0.59, "30°", ha="center", fontsize=10)
angle_ax.set(xlim=(-1, 1), ylim=(-0.12, 1.2))
angle_ax.set_aspect("equal")
angle_ax.axis("off")
angle_ax.set_title("Incidence and reflection", fontsize=10)
fig.text(0.71, 0.235, "Camera placement", color=BLUE, fontsize=12, weight="bold")
fig.text(0.71, 0.20, f"Pupil to gear axis: {np.linalg.norm(C[:2])*1000:.1f} mm\nGear radius: 22 mm", fontsize=10, va="top", linespacing=1.5)
fig.text(0.04, 0.955, "Laser-line inspection setup", fontsize=22, weight="bold")
fig.savefig(OUT / "images" / "outside_camera_overview.png", dpi=180)
plt.close(fig)

metadata = {
    "purpose": "Overview of the revised 0.7 mm tooth model and outside-rim camera used in the new optical runs.",
    "laser_wavelength_nm": 650, "wavelength_source": "User confirmed",
    "color_model": "RGB rendering",
    "selected_surface": "Sloped tooth crown",
    "camera_patch_distance_mm": float(np.linalg.norm(C-P)*1000),
    "camera_radius_from_axis_mm": float(np.linalg.norm(C[:2])*1000),
    "gear_outer_radius_mm": GEAR["supplied"]["outside_diameter"]/2,
    "laser_patch_distance_mm": float(np.linalg.norm(L-P)*1000),
    "incidence_deg": math.degrees(math.acos(np.dot(unit(L-P), N))),
    "reflection_deg": math.degrees(math.acos(np.dot(unit(C-P), N))),
    "geometry": GEOMETRY,
    "notes": ["Outside-rim camera, with a 25 mm pupil distance and the corresponding laser incidence side.",
              "Optical coordinates and pupil size are to scale; device housings are schematic.",
              "Red footprint follows intersections with the existing gear, with a widened display stroke.",
              "Superseded wide-tooth runs are preserved in the archive."],
}
(OUT / "overview_parameters.json").write_text(json.dumps(metadata, indent=2)+"\n")
print(json.dumps({k: v for k, v in metadata.items() if k != "geometry"}, indent=2), flush=True)
print(f"Saved {OUT / 'images' / 'outside_camera_overview.png'}", flush=True)
