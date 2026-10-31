# Encoder holder and slide base

Current encoder holder and its 15 mm slide base. Encoder face height is 58 mm above the board.

The user previously confirmed the encoder mounting pattern. Slots are 5.5 × 3.8 mm at radius 14.5 mm, at 0/30, 120/150 and 240/270 degrees. Use the actual three-hole mounting group and retain case fasteners.

The underside seat is Ø39.5 mm × 2 mm deep, leaving a 5 mm plate floor. The boss opening is Ø21.5 mm. Housing dimensions and exact boss height remain reference estimates.

Use three M3 × 8 screws with approximately 0.6 mm washers for nominal 2.4 mm engagement. The alternative is M3 × 10 with an additional rigid 2 mm spacer per screw. Check actual usable thread depth and pulley clearance.

The base uses four M4 × 12 clamps, four 1 mm washers and four captive M4 nuts, with 5 mm holder feet on a 6 mm base. Four #6 × 1/2 inch flat-head wood screws secure the base.

## Current print files

| Component | Quantity | STL | PLA estimate | Filament cost |
| --- | ---: | --- | ---: | ---: |
| encoder holder | 1 | [encoder_holder_r06.stl](encoder_holder_r06.stl) | 78.5 g | USD 1.57 |
| encoder slide base | 1 | [encoder_slide_base_r06.stl](encoder_slide_base_r06.stl) | 39.7 g | USD 0.79 |

Folder total: approximately **118.2 g PLA / USD 2.36**.
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
| R0000048.JPG | [encoder_reference_shaft_face_01.JPG](scan/encoder_reference_shaft_face_01.JPG) | Shaft face, boss and apparent hole pattern |
| R0000050.JPG | [encoder_reference_side_label_01.JPG](scan/encoder_reference_side_label_01.JPG) | Body length, shaft and model label |

Original image data and hashes are preserved in [scan_sources.json](../scripts/scan_sources.json).

Earlier versions are in the [Lisa archive](../../../archive/lisa/). Regenerate the complete set from repository root with `projects/lisa/scripts/build_two_belt.sh`.
