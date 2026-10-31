"""Write the seeded configuration for six classifier-training gears with random spherical cavities."""

import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SEED = 20261030
GEAR_COUNT = 6
TOOTH_COUNT = 35
KINDS = ["top_edge", "top_edge", "top_edge", "crown", "flank"]


def main():
    rng = np.random.default_rng(SEED)
    gears, seeds = [], {}
    for number in range(1, GEAR_COUNT+1):
        gear_id = f"train_{number}"
        start = int(rng.integers(0, 7))
        kinds = rng.permutation(KINDS)
        defects = []
        for index, kind in enumerate(kinds):
            tooth = (start+7*index+int(rng.integers(-1, 2))) % TOOTH_COUNT
            diameter = float(np.round(rng.uniform(0.10, 0.20)/0.005)*0.005)
            side = 0 if kind == "crown" else int(rng.choice([-1, 1]))
            defects.append({"id": f"T{number}_D{index+1}", "tooth": tooth,
                            "radial_fraction": float(np.round(rng.uniform(0.36, 0.65), 2)),
                            "kind": str(kind), "side": side,
                            "diameter_mm": round(diameter, 3), "depth_mm": round(diameter/2, 4)})
        gears.append({"id": gear_id, "role": "train", "defects": defects})
        seeds[gear_id] = 20000000+1000000*number
    config = {
        "source_run": "outputs/patchcore_prototype_25mm",
        "output": "outputs/classifier_training_25mm",
        "model_output": "gear/model/classifier_training",
        "cad_source": "gear/model/large_gear.FCStd",
        "geometry_title": "Six training gears · spherical cuts · Ø0.10–0.20 mm · depth = radius",
        "rotation_step_deg": 1,
        "defect_sample_offsets_deg": [-0.4, -0.2, 0, 0.2, 0.4],
        "seed_offsets": seeds,
        "cavity_geometry": "Sphere radius equal to depth, centered on the CAD surface or tooth-top edge",
        "sampling": {"generator": "collector/training_set_generate.py", "seed": SEED,
                     "diameter_mm": "uniform 0.10–0.20, rounded to 0.005", "depth": "diameter / 2",
                     "radial_fraction": "uniform 0.36–0.65", "kinds_per_gear": "3 top edge, 1 crown, 1 flank, shuffled",
                     "tooth_spacing": "7 teeth with ±1 jitter from a random start"},
        "gears": gears,
    }
    path = PROJECT/"collector/training_set.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n")
    print(f"Saved {path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
