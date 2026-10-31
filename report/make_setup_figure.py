"""Draw the report's optical-layout figure at print size with only the part labels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent/"figures/setup_schematic_report.png"


def main():
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    ax.set(xlim=(-5.3, 10.0), ylim=(-1.9, 6.2), aspect="equal")
    ax.axis("off")
    red, blue, grey = "#c62828", "#246a9b", "#54616b"
    ax.add_patch(Ellipse((0, -0.7), 3.0, 0.9, facecolor="#c4cbd1", edgecolor=grey, lw=0.8))
    ax.add_patch(Rectangle((-1.3, -1.25), 2.6, 0.4, facecolor="#72808b", edgecolor=grey, lw=0.8))
    ax.add_patch(Polygon([(-0.8, -0.55), (-0.35, 0), (0.75, -0.06), (1.0, -0.55)],
                         facecolor="#d9dfe3", edgecolor=grey, lw=0.8))
    ax.add_patch(FancyArrowPatch((-1.25, -0.75), (1.1, -0.75), connectionstyle="arc3,rad=0.65",
                                 arrowstyle="-|>", mutation_scale=6, lw=0.7, color=grey))
    laser, tooth = np.array([-4.3, 3.6]), np.array([0.0, 0.0])
    axis, across = np.array([0.8, 0.6]), np.array([-0.6, 0.8])
    lens = axis*3.8
    screen, camera = lens+axis*3.0, lens+axis*5.7
    ax.add_patch(Rectangle(laser+[-0.5, -0.3], 1.0, 0.6, facecolor="#eceff1", edgecolor=grey, lw=0.8))
    ax.annotate("", xy=tooth, xytext=laser+[0.25, -0.2], arrowprops={"arrowstyle": "-|>", "color": red, "lw": 1.2})
    for shift in (-1.0, 0, 1.0):
        target = lens+across*shift-axis*0.12
        ax.annotate("", xy=target, xytext=tooth, arrowprops={"arrowstyle": "-|>", "color": red, "lw": 0.7, "alpha": 0.8})
        ax.plot([target[0], screen[0]+across[0]*shift*0.7], [target[1], screen[1]+across[1]*shift*0.7],
                color=red, lw=0.7, alpha=0.7)
    t = np.linspace(-1.1, 1.1, 100)
    curved = lens+t[:, None]*across-(0.32*(1-(t/1.1)**2))[:, None]*axis
    plane = lens+t[:, None]*across+0.09*axis
    ax.add_patch(Polygon(np.vstack((curved, plane[::-1])), facecolor="#b9e2f1", edgecolor=blue, lw=0.8))
    ax.plot(*np.array([screen-across*1.45, screen+across*1.45]).T, color="#947650", lw=3.5)
    ax.add_patch(Rectangle(camera+[-0.45, -0.25], 0.9, 0.5, facecolor="#a9b2ba", edgecolor="#37474f", lw=0.8))
    ax.annotate("", xy=screen+axis*0.2, xytext=camera-axis*0.5, arrowprops={"arrowstyle": "->", "color": blue, "lw": 0.9})
    size = 8
    ax.text(*(laser+[0, 0.55]), "Laser", fontsize=size, ha="center")
    ax.text(-1.75, -0.8, "Gear", fontsize=size, ha="right", va="center")
    ax.text(*(lens+across*1.4+[-0.2, 0.1]), "Lens", fontsize=size, ha="center", color=blue)
    ax.text(*(screen-across*1.6+[0.4, -0.5]), "Screen", fontsize=size, ha="center", color="#725a3d")
    ax.text(*(camera+[0, 0.45]), "Camera", fontsize=size, ha="center")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
