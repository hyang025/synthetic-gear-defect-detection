"""Preview the fifteen spherical cuts with short dimension captions."""

import argparse
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".tools/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def obj_triangles(path):
    vertices, triangles = [], []
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                triangles.append([int(value.split("/")[0])-1 for value in fields[1:4]])
    return np.asarray(vertices)[np.asarray(triangles)]


def clipped_triangles(triangles, lower, upper):
    result = []
    for triangle in triangles:
        polygon = list(triangle)
        for dimension in range(3):
            for boundary, sign in ((lower[dimension], 1), (upper[dimension], -1)):
                cropped = []
                if not polygon:
                    break
                for start, end in zip(polygon, polygon[1:]+polygon[:1]):
                    start_inside = sign*(start[dimension]-boundary) >= 0
                    end_inside = sign*(end[dimension]-boundary) >= 0
                    if start_inside:
                        cropped.append(start)
                    if start_inside != end_inside:
                        t = (boundary-start[dimension])/(end[dimension]-start[dimension])
                        cropped.append(start+t*(end-start))
                polygon = cropped
        for index in range(1, len(polygon)-1):
            result.append([polygon[0], polygon[index], polygon[index+1]])
    return np.asarray(result)


def main(config_path):
    config = json.loads(config_path.read_text())
    metadata = json.loads((PROJECT/config["model_output"]/"models.json").read_text())
    out = PROJECT/config["output"]
    (out/"images").mkdir(parents=True, exist_ok=True)
    row_count = len(metadata["gears"])
    column_count = max(len(gear["defects"]) for gear in metadata["gears"])
    fig = plt.figure(figsize=(15, 3*row_count), layout="constrained")
    for row, gear in enumerate(metadata["gears"]):
        mesh = obj_triangles(PROJECT/gear["mesh"])
        for col, defect in enumerate(gear["defects"]):
            ax = fig.add_subplot(row_count, column_count, row*column_count+col+1, projection="3d")
            center = np.asarray(defect["surface_point_mm"])
            n = np.asarray(defect["surface_normal"])
            u = np.asarray(defect["along_tangent"])
            u -= n*(u@n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            basis = np.column_stack((u, v, n))
            extent = max(defect["diameter_mm"]*0.8, defect["depth_mm"]*1.5)
            local = (mesh-center)@basis
            low, high = local.min(axis=1), local.max(axis=1)
            selected = np.all(low < extent, axis=1) & np.all(high > -extent, axis=1)
            patch = local[selected]
            patch = clipped_triangles(patch, [-extent]*3, [extent, extent, extent*0.35])
            normals = np.cross(patch[:, 1]-patch[:, 0], patch[:, 2]-patch[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            light = np.asarray([-0.3, -0.5, 0.8])
            light /= np.linalg.norm(light)
            shade = 0.35+0.6*np.maximum(normals@light, 0)
            colors = np.column_stack((shade*0.84, shade*0.88, shade*0.94, np.ones(len(shade))))
            ax.add_collection3d(Poly3DCollection(patch, facecolors=colors, linewidths=0))
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent*0.35))
            ax.set_box_aspect((1, 1, 0.7))
            ax.view_init(elev=35, azim=65 if defect["side"] > 0 and defect["kind"] == "top_edge" else -65)
            scale_length = defect["diameter_mm"]
            ax.plot([-scale_length/2, scale_length/2], [-extent*0.85]*2, [-extent*0.8]*2, color="black", lw=2)
            ax.set_title(f"{defect['id']} · {defect['kind'].replace('_', ' ')}\nØ{defect['diameter_mm']:.2f} mm · {defect['depth_mm']*1000:.0f} µm", fontsize=10)
            ax.set_axis_off()
    subject = "Validation gear" if metadata["gears"][0]["id"] == "validation_gear" else f"{row_count} test gears"
    fig.suptitle(config.get("geometry_title", f"{subject} · spherical cuts · bars show nominal diameter"))
    fig.savefig(out/"images/defect_geometry.png", dpi=170)
    plt.close(fig)
    print("Saved images/defect_geometry.png", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=PROJECT/"collector/prototype.json")
    args = parser.parse_args()
    main(args.config.resolve())
