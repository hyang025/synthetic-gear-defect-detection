"""Render the reconstructed gear in Mitsuba and add dimension annotations.

From the Mitsuba folder: .venv/bin/python gear/render_gear.py
These views present the model geometry and dimensions.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".tools" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mitsuba as mi
import numpy as np


PARAMS = json.loads((ROOT / "model_parameters.json").read_text())
S, E = PARAMS["supplied"], PARAMS["estimated"]
OUT = ROOT.parent / "outputs" / "gear" / "images"
OUT.mkdir(parents=True, exist_ok=True)
mi.set_variant("cuda_ad_rgb")
from machining_marks import machining_material

T = mi.ScalarTransform4f
BLUE, AMBER = "#12669b", "#976017"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def point_mm(values):
    return [v * 0.001 for v in values]


def softbox(origin, size_mm, radiance):
    return {
        "type": "rectangle",
        "to_world": T.look_at(origin=point_mm(origin), target=[0, 0, 0.0065], up=[0, 1, 0])
        @ T.scale(size_mm * 0.001),
        "bsdf": {"type": "diffuse", "reflectance": 0.0},
        "emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance}},
    }


def render_view(origin, up, target=(0, 0, 6.5), half_width_mm=30, spp=96):
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6, "hide_emitters": True},
        "sensor": {
            "type": "orthographic",
            "to_world": T.look_at(origin=point_mm(origin), target=point_mm(target), up=up)
            @ T.scale(half_width_mm * 0.001),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {"type": "hdrfilm", "width": 1200, "height": 1200,
                     "pixel_format": "rgba",
                     "rfilter": {"type": "box"}},
        },
        "gear": {
            "type": "obj", "filename": str(ROOT / "model" / "large_gear.obj"),
            "to_world": T.scale(0.001),
            "bsdf": machining_material(PARAMS),
        },
        "background": {"type": "constant", "radiance": 1.0},
        "key": softbox((-55, -55, 85), 32, 7),
        "fill": softbox((65, 25, 50), 27, 3),
        "rear": softbox((-35, -30, -65), 35, 6),
    })
    # Bound GPU memory while other requested render sequences may be running.
    pixels = None
    for start in range(0, spp, 8):
        count = min(8, spp-start)
        batch = np.asarray(mi.render(scene, spp=count, seed=11+start)).copy()
        pixels = batch*count/spp if pixels is None else pixels+batch*count/spp
    bitmap = mi.Bitmap(pixels).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8,
                                       srgb_gamma=True)
    rgba = np.asarray(bitmap).astype(np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    return np.clip(rgba[:, :, :3] + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def dimension(ax, p1, p2, text, color=BLUE, vertical=False):
    ax.annotate("", xy=p1, xytext=p2,
                arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.3})
    x, y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    ax.text(x, y, text, color=color, ha="center", va="center",
            rotation=90 if vertical else 0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5})


def leader(ax, point, label_at, text, color=BLUE, ha="left"):
    ax.annotate(text, xy=point, xytext=label_at, color=color, ha=ha, va="center",
                fontsize=10.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1})


def frame(title, image, side=False):
    fig, ax = plt.subplots(figsize=(11, 5.7) if side else (9, 9))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.045, right=0.955, bottom=0.10, top=0.88)
    extent = (-30, 30, -23.5, 36.5) if side else (-30, 30, -30, 30)
    ax.imshow(image, extent=extent)
    ax.set_xlim(-33, 33) if side else ax.set_xlim(-36, 36)
    ax.set_ylim(-5, 18) if side else ax.set_ylim(-32, 32)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle(title, fontsize=19, x=0.07, ha="left", y=0.965, weight="bold")
    return fig, ax


def footer(fig):
    fig.text(0.07, 0.063, "[S] Supplied dimensions", color=BLUE, fontsize=10.5)
    fig.text(0.39, 0.063, "[E] Photo-based estimate", color=AMBER, fontsize=10.5)


print("Rendering the front view...", flush=True)
front = render_view((0, 0, 130), (0, 1, 0))
fig, ax = frame("Large gear / front", front)
fig.text(0.07, 0.912, "35 teeth [S]    |    Dimensions in mm", color=BLUE, fontsize=12)
for x in (-22, 22):
    ax.plot([x, x], [0, -27], color=BLUE, lw=0.8)
dimension(ax, (-22, -26), (22, -26), "Outside Ø44 [S]")
leader(ax, (3.9, 3.9), (16, 25), "Bore Ø11 [S]")
leader(ax, (-6.6, 0), (-33, 18), "Keyway width 2.5 [S]")
ax.add_patch(Circle((0, 0), 12.5, fill=False, ls=(0, (4, 4)), lw=0.8,
                    color=AMBER, alpha=0.7))
leader(ax, (-12.5, 0), (-33, -19), "Pitch circle Ø25 [E]", AMBER)
leader(ax, (8.84, -8.84), (17, -19), "4 × Ø5 blind holes [E]\nØ6 entries [E]", AMBER)
footer(fig)
leader(ax, (18.5, -1.8), (19, 13), "Mid-tooth top ≈0.7 mm\nUser measurement", BLUE)
fig.savefig(OUT / "front_dimensions.png", dpi=180)
plt.close(fig)

print("Rendering the side view...", flush=True)
side = render_view((0, -130, 6.5), (0, 0, 1))
fig, ax = frame("Large gear / side", side, side=True)
fig.text(0.07, 0.902, "Dimensions in mm", fontsize=11)
for x in (-22, 22):
    ax.plot([x, x], [6.4, -3.8], color=BLUE, lw=0.8)
dimension(ax, (-22, -3.3), (22, -3.3), "Outside Ø44 [S]")
ax.plot([10.2, 28], [0, 0], color=BLUE, lw=0.8)
ax.plot([15.5, 28], [13, 13], color=BLUE, lw=0.8)
dimension(ax, (27, 0), (27, 13), "13 total [S]", vertical=True)
ax.plot([-14.5, -10.5], [0, 0], color=AMBER, lw=0.8)
dimension(ax, (-14, 0), (-14, 6), "6 [E]", AMBER, vertical=True)
leader(ax, (10.5, 2.5), (14.5, 0.7), "Hub Ø21 [E]", AMBER)
footer(fig)
fig.savefig(OUT / "side_dimensions.png", dpi=180)
plt.close(fig)

print("Rendering the front three-quarter view...", flush=True)
front_iso = render_view((70, -95, 90), (0, 0, 1))
fig, ax = frame("Large gear / front three-quarter", front_iso)
fig.text(0.07, 0.912, "Ø44 × 13 mm  |  35 teeth  |  Bore Ø11  |  Keyway 2.5  [S]",
         fontsize=11, color=BLUE)
footer(fig)
fig.savefig(OUT / "front_three_quarter.png", dpi=180)
plt.close(fig)

print("Rendering the rear three-quarter view...", flush=True)
rear_iso = render_view((70, -95, -75), (0, 0, -1))
fig, ax = frame("Large gear / rear three-quarter", rear_iso)
fig.text(0.07, 0.912, "Estimated rear hub: Ø21 mm, 6 mm projection [E]", fontsize=11, color=AMBER)
footer(fig)
fig.savefig(OUT / "rear_three_quarter.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(16, 15), facecolor="white")
for ax, filename in zip(axes.flat, ("front_three_quarter.png", "front_dimensions.png",
                                    "rear_three_quarter.png", "side_dimensions.png")):
    ax.imshow(plt.imread(OUT / filename))
    ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, hspace=0.015, wspace=0.005)
fig.savefig(OUT / "gear_overview.png", dpi=180)
plt.close(fig)
print(f"Saved dimensioned views in {OUT}", flush=True)

print("Rendering the machining-mark close-up...", flush=True)
closeup = render_view((8, -35, 28), (0, 0, 1), target=(0, -18.2, 11.7),
                      half_width_mm=4, spp=256)
fig, ax = plt.subplots(figsize=(10, 9.5), facecolor="white")
fig.subplots_adjust(left=0.055, right=0.945, bottom=0.14, top=0.88)
ax.imshow(closeup, extent=(-4, 4, -4, 4))
ax.axis("off")
fig.suptitle("Machining marks", x=0.07, y=0.97,
             ha="left", fontsize=18, weight="bold")
detail = PARAMS["surface_detail"]
fig.text(0.07, 0.916,
         f"Spacing ≈ {detail['estimated_mark_spacing_mm']:.2f} mm [E]  |  Amplitude {detail['assumed_bump_amplitude_mm']*1000:g} µm (assumed)",
         fontsize=11.5, color=AMBER)
ax.plot([-3.5, -3.0], [-3.5, -3.5], color="white", lw=7, solid_capstyle="butt")
ax.plot([-3.5, -3.0], [-3.5, -3.5], color=BLUE, lw=3, solid_capstyle="butt")
ax.text(-3.25, -3.3, "0.5 mm", color=BLUE, ha="center", fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
fig.savefig(OUT / "machining_closeup.png", dpi=180)
plt.close(fig)
print(f"Saved {OUT / 'machining_closeup.png'}", flush=True)

print("Rendering the measured tooth-top width...", flush=True)
metadata = json.loads((ROOT / "model" / "build_metadata.json").read_text())
section = metadata["crown_section"]
point, normal = np.array(section["center_mm"]), np.array(section["normal"])
along, across = np.array(section["along_tangent"]), np.array(section["across_tangent"])
if normal[2] < 0:
    normal = -normal
detail_view = render_view(point+25*normal, along.tolist(), target=point, half_width_mm=1.8, spp=128)
fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
fig.subplots_adjust(left=0.07, right=0.93, bottom=0.13, top=0.87)
ax.imshow(detail_view, extent=(-1.8, 1.8, -1.8, 1.8))
ax.axis("off")
dimension(ax, (-0.35, -1.25), (0.35, -1.25), "≈0.7 mm", BLUE)
for x in (-0.35, 0.35):
    ax.plot([x, x], [0, -1.32], color=BLUE, lw=0.8)
ax.plot([-0.35, 0.35], [0, 0], color=BLUE, lw=1.2)
fig.suptitle("Tooth-top width", x=0.07, y=0.965, ha="left", fontsize=20, weight="bold")
fig.text(0.07, 0.915, "Radial midpoint · Perpendicular width", fontsize=11)
fig.savefig(OUT / "tooth_width.png", dpi=180)
plt.close(fig)
