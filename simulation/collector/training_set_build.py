"""FreeCAD entry point for the six classifier-training gears with random spherical cuts."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT/"collector"))
import prototype_build

prototype_build.CONFIG = PROJECT/"collector/training_set.json"
prototype_build.main()
