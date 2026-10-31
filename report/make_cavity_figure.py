"""Draw the report's simulation figure at print size: one cavity of each type and healthy reflection
patterns with the laser on a tooth-top edge and in the groove. Run with the Mitsuba environment."""

import json
from pathlib import Path
import sys

MITSUBA = Path(__file__).resolve().parents[1]/"simulation"
sys.path.insert(0, str(MITSUBA/"collector"))
from knn_vectors import angle_bins
from prototype_geometry import clipped_triangles, obj_triangles

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mitsuba as mi
import numpy as np

OUT = Path(__file__).resolve().parent/"figures/cavity_types.png"
CHOSEN = [("G1_D1", "Tooth-top edge"), ("G1_D4", "Crown"), ("G1_D5", "Flank")]
RUN = MITSUBA/"outputs/sequence_0p1875_25mm"
HALF_RANGE, BIN = 40, 1.0


def cavity_panel(ax, mesh, defect, title):
    centre, n = np.asarray(defect["surface_point_mm"]), np.asarray(defect["surface_normal"])
    u = np.asarray(defect["along_tangent"])
    u -= n*(u@n)
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(n, u), n))
    extent = max(defect["diameter_mm"]*0.8, defect["depth_mm"]*1.5)
    local = (mesh-centre)@basis
    keep = np.all(local.min(axis=1) < extent, axis=1) & np.all(local.max(axis=1) > -extent, axis=1)
    patch = clipped_triangles(local[keep], [-extent]*3, [extent, extent, extent*0.35])
    normals = np.cross(patch[:, 1]-patch[:, 0], patch[:, 2]-patch[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.asarray([-0.3, -0.5, 0.8])/np.linalg.norm([-0.3, -0.5, 0.8])
    shade = 0.35+0.6*np.maximum(normals@light, 0)
    ax.add_collection3d(Poly3DCollection(patch, facecolors=np.column_stack((shade*0.84, shade*0.88, shade*0.94, np.ones(len(shade)))),
                                         linewidths=0))
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent*0.35))
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=35, azim=65 if defect["side"] > 0 and defect["kind"] == "top_edge" else -65)
    ax.set_title(title, fontsize=8, pad=0)
    ax.set_axis_off()


def reflection_panel(ax, frame, flat, count, title):
    image = np.asarray(mi.Bitmap(str(RUN/frame["raw"]))).astype(np.float64).ravel()
    inside = flat >= 0
    binned = np.bincount(flat[inside], weights=image[inside], minlength=count*count).reshape(count, count)
    ax.imshow(np.sqrt(binned/image.sum()), cmap="gray", origin="lower",
              extent=[-HALF_RANGE, HALF_RANGE, -HALF_RANGE, HALF_RANGE])
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([-40, 0, 40])
    ax.set_yticks([-40, 0, 40])
    ax.tick_params(labelsize=6, length=2, pad=1)


def main():
    models = json.loads((MITSUBA/"gear/model/patchcore_hemisphere_0p2/models.json").read_text())
    gear = next(g for g in models["gears"] if g["id"] == "gear_1")
    mesh = obj_triangles(MITSUBA/gear["mesh"])
    manifest = json.loads((RUN/"capture_manifest.json").read_text())
    cases = json.loads((RUN/"cases.json").read_text())
    step, pitch = cases["step_deg"], 360/35
    crossings = {d["id"]: d["crossing_angle_deg"] for c in cases["cases"] if c["case"] == "d020_gear_1" for d in c["defects"]}
    healthy = {f["index"]: f for f in manifest["frames"] if f["case"] == "healthy_a"}
    edge = healthy[round(crossings["G1_D1"]/step) % len(healthy)]
    groove = healthy[round(((crossings["G1_D4"]+pitch/2) % 360)/step) % len(healthy)]
    flat, count = angle_bins(manifest["trace_config"], HALF_RANGE, BIN)
    mi.set_variant("scalar_rgb")
    fig = plt.figure(figsize=(5.4, 1.25))
    for column, (name, title) in enumerate(CHOSEN):
        cavity_panel(fig.add_subplot(1, 5, column+1, projection="3d"), mesh,
                     next(d for d in gear["defects"] if d["id"] == name), title)
    reflection_panel(fig.add_subplot(1, 5, 4), edge, flat, count, "Edge")
    reflection_panel(fig.add_subplot(1, 5, 5), groove, flat, count, "Groove")
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0.08, wspace=0.4)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
