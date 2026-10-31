"""Collect the October 1 demo assets and draw the proposed bench arrangement."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT/"outputs/oct_1_demo"
RUN = PROJECT/"outputs/patchcore_hemisphere_0p2_25mm"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".tools/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
import numpy as np


def setup_figure(path):
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    ax.set(xlim=(-6.2, 10.5), ylim=(-2.6, 6.4), aspect="equal")
    ax.axis("off")
    red, blue = "#c62828", "#246a9b"
    ax.add_patch(Ellipse((0, -0.7), 3.0, 0.9, facecolor="#c4cbd1", edgecolor="#54616b", lw=1.5))
    ax.add_patch(Rectangle((-1.3, -1.25), 2.6, 0.4, facecolor="#72808b", edgecolor="#54616b"))
    ax.add_patch(Polygon([(-0.8, -0.55), (-0.35, 0), (0.75, -0.06), (1.0, -0.55)],
                         facecolor="#d9dfe3", edgecolor="#54616b", lw=1.3))
    ax.plot([-0.35, 0.55], [0, -0.05], color=red, lw=4)
    ax.text(0, -2.0, "Gear + encoder turntable\nØ44 mm · 35 teeth", ha="center", va="center", fontsize=11)
    ax.add_patch(FancyArrowPatch((-1.25, -0.75), (1.1, -0.75), connectionstyle="arc3,rad=0.65",
                                arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#54616b"))
    laser = np.array([-4.3, 3.6])
    tooth = np.array([0.0, 0.0])
    axis = np.array([0.8, 0.6])
    across = np.array([-0.6, 0.8])
    lens = axis*3.8
    screen = lens+axis*3.0
    camera = screen+axis*2.7
    ax.add_patch(Rectangle(laser+[-0.5, -0.3], 1.0, 0.6, facecolor="#eceff1", edgecolor="#54616b"))
    ax.annotate("", xy=tooth, xytext=laser+[0.25, -0.2],
                arrowprops={"arrowstyle": "-|>", "color": red, "lw": 2})
    ax.text(-5.3, 4.45, "650 nm line laser\n≈1 mW · 300 mm working distance", fontsize=11, ha="left")
    ax.text(-2.7, 0.65, "6.7 × 0.1 mm line", color=red, fontsize=11, ha="center")
    for shift in (-1.0, 0, 1.0):
        target = lens+across*shift-axis*0.12
        ax.annotate("", xy=target, xytext=tooth,
                    arrowprops={"arrowstyle": "-|>", "color": red, "lw": 1.2, "alpha": 0.8})
        ax.plot([target[0], screen[0]+across[0]*shift*0.7],
                [target[1], screen[1]+across[1]*shift*0.7], color=red, lw=1.1, alpha=0.7)
    t = np.linspace(-1.1, 1.1, 100)
    curved = lens+t[:, None]*across-(0.32*(1-(t/1.1)**2))[:, None]*axis
    plane = lens+t[:, None]*across+0.09*axis
    ax.add_patch(Polygon(np.vstack((curved, plane[::-1])), facecolor="#b9e2f1", edgecolor=blue, lw=1.4))
    ax.plot(*np.array([screen-across*1.45, screen+across*1.45]).T, color="#947650", lw=7)
    ax.add_patch(Rectangle(camera+[-0.45, -0.25], 0.9, 0.5, facecolor="#a9b2ba", edgecolor="#37474f"))
    ax.annotate("", xy=screen+axis*0.2, xytext=camera-axis*0.5,
                arrowprops={"arrowstyle": "->", "color": blue, "lw": 1.5})
    ax.text(1.0, 4.9, "PCX collector\nØ50 mm · EFL 50 mm", color=blue, fontsize=11, ha="center")
    ax.annotate("Curved side faces gear", xy=lens-axis*0.27, xytext=(1.25, 3.5),
                fontsize=10, color=blue, ha="right", arrowprops={"arrowstyle": "-", "color": blue})
    ax.text(7.0, 1.6, "Thin translucent screen\n120–150 mm wide", color="#725a3d", fontsize=11, ha="center")
    ax.text(9.05, 4.8, "OV9281 monochrome\n12 mm M12 lens\nFocus on screen", fontsize=10, ha="center")
    def dimension(start, end, offset, text):
        start, end = start+offset*across, end+offset*across
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "<->", "color": "#54616b", "lw": 1.1})
        middle = (start+end)/2
        ax.text(middle[0]+0.2, middle[1]-0.35, text, fontsize=10,
                color="#37474f", ha="center", rotation=np.degrees(np.arctan2(axis[1], axis[0])))
    dimension(tooth, lens, -1.5, "25 mm to collector plane")
    dimension(lens, screen, -1.5, "50 mm ideal focal plane")
    ax.text(5.2, -1.75, "Physical screen travel: 30–70 mm behind rear lens surface", fontsize=10, ha="center")
    ax.set_title("Proposed LISA bench setup · schematic spacing", fontsize=16, pad=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    manifest_path = OUT/"asset_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Preserve the collected demo assets.")
    for folder in ("images", "results", "models/healthy", "animations", "heatmaps"):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    copied, linked = {}, {}
    def copy(source, destination):
        source, destination = PROJECT/source, OUT/destination
        shutil.copy2(source, destination)
        copied[str(destination.relative_to(OUT))] = {"source": str(source.relative_to(PROJECT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    def link(source, destination):
        source, destination = PROJECT/source, OUT/destination
        destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())
        linked[str(destination.relative_to(OUT))] = str(source.relative_to(PROJECT))
    for name, destination in (
        ("gear_overview.png", "02_gear_dimensions.png"),
        ("tooth_width.png", "02_tooth_top_width.png"),
        ("machining_closeup.png", "02_machining_detail.png")):
        copy(f"outputs/gear/images/{name}", f"images/{destination}")
    for name, destination in (
        ("defect_geometry.png", "03_defect_shapes.png"),
        ("test_captures.png", "04_damaged_reflection_patterns.png"),
        ("size_comparison.png", "05_size_comparison.png"),
        ("rotation_scores.png", "05_full_rotation_scores.png"),
        ("defect_detection.png", "05_defect_detection.png")):
        copy(str((RUN/"images"/name).relative_to(PROJECT)), f"images/{destination}")
    copy("outputs/collector_rotation_25mm/images/focal_plane.png", "images/04_healthy_reflection_patterns.png")
    copy("outputs/collector_rotation_25mm/animations/collector_rotation.mp4", "animations/healthy_reflection_rotation.mp4")
    for name in ("config.json", "metrics.json", "threshold.json", "size_comparison.csv", "defect_predictions.csv",
                 "defects.json", "evaluation_manifest.json", "capture_manifest.json"):
        copy(str((RUN/name).relative_to(PROJECT)), f"results/{name}")
    for case in ("gear_1", "gear_4", "gear_5"):
        link(f"gear/model/patchcore_hemisphere_0p2/{case}", f"models/{case}")
    for name, destination in (("large_gear.FCStd", "healthy_gear.FCStd"), ("large_gear.step", "healthy_gear.step"),
                              ("damage_test/healthy_control.obj", "healthy_capture_mesh.obj")):
        link(f"gear/model/{name}", f"models/healthy/{destination}")
    link(str(RUN.relative_to(PROJECT)), "latest_experiment")
    link("outputs/patchcore_hemisphere_dense_25mm/histogram/patchcore_model.pt", "heatmaps/frozen_histogram_model.pt")
    setup_figure(OUT/"images/01_physical_setup.png")
    manifest_path.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "latest_experiment": str(RUN.relative_to(PROJECT)), "copied_assets": copied, "linked_assets": linked,
        "setup_source": "README.md optical arrangement and collector/rotation.json",
        "setup_presentation": "Schematic spacing with the proposed PCX collector and screen camera"}, indent=2)+"\n")
    print(f"Collected demo assets in {OUT.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
