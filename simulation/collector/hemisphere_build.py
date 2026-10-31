"""FreeCAD entry point for surface- and edge-centered 0.10 mm sphere cuts."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT/"collector"))
import prototype_build

prototype_build.CONFIG = PROJECT/"collector/hemisphere.json"
prototype_build.main()
