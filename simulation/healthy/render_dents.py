"""Presentation views of the actual dent geometry, separate from camera captures."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = PROJECT / "outputs" / "damage_test"
(OUT / "images").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".tools" / "matplotlib"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")
from scene import material, unit

T = mi.ScalarTransform4f
config = json.loads((ROOT / "full_revolution.json").read_text())
config["material"]["alpha"] = 0.18
gear = json.loads((PROJECT / "gear/model_parameters.json").read_text())
dents = json.loads((OUT / "dents.json").read_text())["dents"]


def render(origin_mm, target_mm, up, half_width_mm, size):
    origin, target = np.asarray(origin_mm)*0.001, np.asarray(target_mm)*0.001
    scene = mi.load_dict({
        "type": "scene", "integrator": {"type": "path", "max_depth": 5, "hide_emitters": True},
        "sensor": {"type": "orthographic", "to_world": T.look_at(origin=origin.tolist(), target=target.tolist(), up=np.asarray(up).tolist()) @ T.scale(half_width_mm*0.001),
                   "sampler": {"type": "independent", "sample_count": 8},
                   "film": {"type": "hdrfilm", "width": size, "height": size, "pixel_format": "rgba", "rfilter": {"type": "box"}}},
        "gear": {"type": "obj", "filename": str(PROJECT / "gear/model/damage_test/damaged_gear.obj"), "to_world": T.scale(0.001), "bsdf": material(config, gear, 0)},
        "background": {"type": "constant", "radiance": 0.5},
        "key": {"type": "rectangle", "to_world": T.look_at(origin=[-0.03, -0.05, 0.07], target=[0, 0, 0.01], up=[0, 1, 0]) @ T.scale(0.025), "emitter": {"type": "area", "radiance": 5}},
        "fill": {"type": "rectangle", "to_world": T.look_at(origin=[0.04, 0.05, 0.05], target=[0, 0, 0.01], up=[0, 1, 0]) @ T.scale(0.025), "emitter": {"type": "area", "radiance": 2}},
    })
    pixels = np.zeros((size, size, 4), dtype=np.float32)
    for i in range(12):
        pixels += np.asarray(mi.render(scene, spp=8, seed=1300+i))/12
    rgba = np.asarray(mi.Bitmap(pixels).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=True)).astype(float)
    return np.clip(rgba[..., :3]+255*(1-rgba[..., 3:4]/255), 0, 255).astype(np.uint8)


print("Rendering the dent locations and geometric close-ups...", flush=True)
overview = render([0, 0, 100], [0, 0, 10], [0, 1, 0], 25, 900)
fig, axes = plt.subplots(2, 2, figsize=(10, 9.6), layout="constrained", facecolor="white")
top = axes[0, 0]
top.imshow(overview, extent=(-25, 25, -25, 25))
top.axis("off")
top.set_title("Dent locations", fontsize=13)
colors = ["#bc3a40", "#176dad", "#258158"]
for letter, dent, color, xytext in zip("ABC", dents, colors, [(17, 21), (-24, 12), (13, -23)]):
    p = np.array(dent["center_mm"])
    top.scatter(p[0], p[1], s=90, facecolors="none", edgecolors=color, lw=1.5)
    top.annotate(letter, xy=p[:2], xytext=xytext, color=color,
                 arrowprops={"arrowstyle": "->", "color": color}, fontsize=13, weight="bold")
for index, (dent, color) in enumerate(zip(dents, colors)):
    p, n = np.array(dent["center_mm"]), np.array(dent["normal"])
    radial = unit([p[0], p[1], 0])
    up = unit(radial-np.dot(radial, n)*n)
    side = unit(np.cross(up, n))
    image = render(p+3*n+0.9*side, p, up, 0.35, 600)
    ax = axes.flat[index+1]
    ax.imshow(image, extent=(-0.35, 0.35, -0.35, 0.35))
    ax.axis("off")
    ax.set_title(f"{'ABC'[index]} · Ø{dent['diameter_mm']:g} × {dent['depth_mm']:g} mm deep", color=color, fontsize=13)
    ax.plot([-0.29, -0.19], [-0.29, -0.29], color="white", lw=6)
    ax.plot([-0.29, -0.19], [-0.29, -0.29], color="#222222", lw=2)
    ax.text(-0.24, -0.26, "0.1 mm", ha="center", color="#222222", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
fig.suptitle("Gear dents", fontsize=22, weight="bold")
fig.savefig(OUT / "images" / "dent_geometry.png", dpi=170)
plt.close(fig)
print(f"Saved {OUT / 'images' / 'dent_geometry.png'}", flush=True)
