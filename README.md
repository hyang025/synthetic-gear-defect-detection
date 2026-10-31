# Using Synthetic Data for Automatic Gear Defect Detection

Source code for the ECE 4424 capstone milestone.

- `simulation/`: Mitsuba 3 simulation, damaged-gear builds (FreeCAD), and the PCA/KNN detector
- `report/`: milestone report and figure scripts
- `rig/`: turntable CAD scripts and STL files

The simulation needs Mitsuba 3.9.1 (`simulation/requirements.txt`); the detector needs scikit-learn. Rendered frames are not included; the scripts in `simulation/collector/` regenerate them.
