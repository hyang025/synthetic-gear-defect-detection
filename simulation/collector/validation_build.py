"""FreeCAD entry point for the additional threshold-validation gear."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT/"collector"))
import prototype_build

print(f"FreeCAD entry namespace: {__name__}; arguments: {sys.argv}", flush=True)
prototype_build.CONFIG = PROJECT/"collector/validation_patchcore.json"
prototype_build.main()
