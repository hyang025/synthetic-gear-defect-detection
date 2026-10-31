# Motor holder and slide base

Moving motor holder and its 15 mm slide base. Motor face height is 42 mm above the board.

Mounting holes: four Ø3.8 mm holes on a 33 × 18 mm pattern; shaft offset from the near column is 9.7 mm. The pattern is photo-derived (approximately ±0.3 mm).

The gearbox recess includes the approved 1.55 mm near-side relief. User measurements: raised pad OD7.7 mm and 2 mm pad-to-case edge gap. Recess wall: 6.45 mm from the screw centre, providing 0.6 mm nominal clearance and a 3.85 mm lip. Pocket depth is 4 mm; mounting plate floor is 3 mm.

Use four M3 × 6 motor screws with four approximately 0.6 mm washers; nominal thread engagement is 2.4 mm. Actual usable thread depth and revised housing fit need checking.

The base uses four M4 × 12 clamps, four 1 mm washers and four captive M4 nuts, with 5 mm holder feet on a 6 mm base. Four #6 × 1/2 inch flat-head wood screws secure the base.

## Current print files

| Component | Quantity | STL | PLA estimate | Filament cost |
| --- | ---: | --- | ---: | ---: |
| motor holder | 1 | [motor_holder_r06.stl](motor_holder_r06.stl) | 57.9 g | USD 1.16 |
| motor slide base | 1 | [motor_slide_base_r06.stl](motor_slide_base_r06.stl) | 52.3 g | USD 1.05 |

Folder total: approximately **110.2 g PLA / USD 2.21**.
Estimates use PLA at USD20/kg, density 1.24 g/cm³, 5% waste and the existing 0.6 mm voxel shell estimate. Allow approximately ±20%; supports and hardware are excluded. Confirm with the slicer.

Editable CAD and STEP files are in [output/model](output/model/). Dimensioned drawings and previews are in [output/render](output/render/).
Use the [assembly manual](../manual.pdf) for illustrated steps, exact print orientations, hardware quantities and support placement.

## Printing and checks

Bambu Lab A1 mini, 0.4 mm nozzle: PLA, 0.20 mm layers, six walls (approximately 2.5–2.7 mm), six top/bottom layers (1.2 mm). Use 40% gyroid for structural parts and 100% infill for both gear-adapter pieces. Optional 3 mm brim.
Fit, strength and fastening take priority. Support flat M4 washer lands, horizontal fastener bores and nut-pocket roofs where needed, then remove support completely. Individual orientations and support notes are in the [print table](../scripts/two_belt_print_estimates.csv).

CAD/STL checks cover valid single solids, watertight meshes, bed placement, dimensions and the A1 mini envelope. The gusseted frame also has CAD checks for bearing insertion, bridge seating and driver/wrench access. Seven other component CAD, STEP and STL files are unchanged from the previous two-piece release. Physical fit and loaded stiffness need checks with the actual hardware.

## Reference photographs

| Original inbox name | Current reference | View |
| --- | --- | --- |
| R0000051.JPG | [motor_reference_side_01.JPG](scan/motor_reference_side_01.JPG) | Gearbox side, motor can and shaft |
| R0000052.JPG | [motor_reference_shaft_face_01.JPG](scan/motor_reference_shaft_face_01.JPG) | Shaft face and four apparent mounting holes |
| R0004472.JPG | [motor_reference_shaft_face_leveled_02.JPG](scan/motor_reference_shaft_face_leveled_02.JPG) | Leveled shaft face with both metric ruler directions; r05 hole calibration |

Original image data and hashes are preserved in [scan_sources.json](../scripts/scan_sources.json).

Earlier versions are in the [Lisa archive](../../../archive/lisa/). Regenerate the complete set from repository root with `projects/lisa/scripts/build_two_belt.sh`.
