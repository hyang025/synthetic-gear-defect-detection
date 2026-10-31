# Gear adapter

Two current components: keyed sleeve with support flange, and removable clamp cap. Print these small parts first to check the actual gear and shaft fit.

Sleeve OD11.9 mm; 2.8 mm key, outer X=7.2 mm and 10 mm key height; sleeve height 17 mm above a 2 mm flange. Shaft bore is 6.3 mm with a D flat at Y=2.65 mm. The sleeve split is 1.2 mm.

Cap OD28 mm, sleeve bore Ø12.2 mm and height 8 mm; split 2.2 mm. The M3 passage is round Ø3.4 mm and the retained nut pocket is 5.8 mm across flats × 2.8 mm deep. Use one M3 × 16 screw and one M3 hex nut.

Gear bore and key fit are photo-derived. Actual grip, torque capacity and long-term PLA clamp durability need physical checks.

## Current print files

| Component | Quantity | STL | PLA estimate | Filament cost |
| --- | ---: | --- | ---: | ---: |
| gear adapter sleeve | 1 | [gear_adapter_sleeve_r06.stl](gear_adapter_sleeve_r06.stl) | 2.5 g | USD 0.05 |
| gear adapter clamp cap | 1 | [gear_adapter_clamp_cap_r06.stl](gear_adapter_clamp_cap_r06.stl) | 4.0 g | USD 0.08 |

Folder total: approximately **6.5 g PLA / USD 0.13**.
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
| R0004473.JPG | [encoder_gear_hub_face_01.JPG](scan/encoder_gear_hub_face_01.JPG) | Gear rear hub, bore and keyway |
| R0004474.JPG | [encoder_gear_teeth_face_01.JPG](scan/encoder_gear_teeth_face_01.JPG) | Gear toothed face, bore and keyway |
| R0004475.JPG | [encoder_gear_side_01.JPG](scan/encoder_gear_side_01.JPG) | Gear side and hub height |

Original image data and hashes are preserved in [scan_sources.json](../scripts/scan_sources.json).

Earlier versions are in the [Lisa archive](../../../archive/lisa/). Regenerate the complete set from repository root with `projects/lisa/scripts/build_two_belt.sh`.
