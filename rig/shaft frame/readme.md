# Shaft frame

Two current components: one rigid body with integrated feet, uprights and lower bearing deck, plus a removable upper bearing bridge. The two X sides remain open for bearing insertion and tools.

The first-level rectangular screw-joint flanges are removed. Twelve local triangular gussets brace the feet, lower deck and upper bridge seats: four at each level, 4 mm thick, centred at X±24 mm. Foot ribs reach 8 mm and rise 16 mm; lower-deck ribs reach 12 mm and drop 12 mm; seat ribs reach 8 mm and drop 10 mm. These are selected design dimensions; stiffness and load capacity have not been measured.

Installed envelope stays 64 × 124 × 92 mm. Lower deck is 6 mm thick at Z43.5–49.5 mm; upright walls are 8 mm thick. Four upper bridge joints use round Ø4.4 mm holes at X±24, Z78 mm. Bearing mounting pattern stays 32 × 32 mm with Ø4.4 mm holes and Ø8 mm shaft openings.

The narrow upper bridge seats remain functional: they carry the removable bridge and hold it in position while the screws are fitted. They are 4.5 mm thick with seating faces at Z70 mm, reinforced by four local ribs. Two Ø4 × 2 mm pins have 0.5 mm lead-ins. The front socket is Ø4.5 mm; the rear slot is 4.5 × 5.5 mm, both 2.25 mm deep. Assumed fit allowances: 0.25 mm at each bridge end, 0.25 mm radial pin clearance and 0.25 mm axial clearance; rear slot adds 1 mm along Y.

Use eight M4 × 16 bearing screws, four M4 × 25 bridge screws and twelve plain M4 nuts for this frame. Four #6 × 1/2 inch wood screws secure the integrated feet. Omit extra bearing washers to retain upper-pulley clearance. No adhesive is needed.

Bench assembly: remove all deck supports; slide the lower bearing in from an open X side and bolt it under the deck before mounting to plywood. Prefit the upper bearing to its bridge. Load the bare 6 mm shaft, both 80T pulleys and both closed belts with the bridge removed. Lower the bridge and bearing over the shaft, locate the pins and seat both shoulders. Start all four side bolts, hold the nuts through the open sides and tighten evenly. Add and adjust collars and gear afterwards; do not force axial bearing preload.

The 6 mm D shaft is 100 mm long. Bearing centres remain Z40/Z95.5 mm and belt centres Z61/Z77 mm. The illustrated manual includes the bearing sandwich sequence, screw directions and tool access. Main body prints feet down with support under its lower bearing deck; bridge prints broad deck down. Remove supports through the open sides before assembly.

## Current print files

| Component | Quantity | STL | PLA estimate | Filament cost |
| --- | ---: | --- | ---: | ---: |
| shaft frame body | 1 | [shaft_frame_body_r06.stl](shaft_frame_body_r06.stl) | 108.7 g | USD 2.17 |
| shaft upper bearing bridge | 1 | [shaft_upper_bearing_bridge_r06.stl](shaft_upper_bearing_bridge_r06.stl) | 42.1 g | USD 0.84 |

Folder total: approximately **150.8 g PLA / USD 3.01**.
Estimates use PLA at USD20/kg, density 1.24 g/cm³, 5% waste and the existing 0.6 mm voxel shell estimate. Allow approximately ±20%; supports and hardware are excluded. Confirm with the slicer.

Editable CAD and STEP files are in [output/model](output/model/). Dimensioned drawings and previews are in [output/render](output/render/).
Use the [assembly manual](../manual.pdf) for illustrated steps, exact print orientations, hardware quantities and support placement.

## Printing and checks

Bambu Lab A1 mini, 0.4 mm nozzle: PLA, 0.20 mm layers, six walls (approximately 2.5–2.7 mm), six top/bottom layers (1.2 mm). Use 40% gyroid for structural parts and 100% infill for both gear-adapter pieces. Optional 3 mm brim.
Fit, strength and fastening take priority. Support flat M4 washer lands, horizontal fastener bores and nut-pocket roofs where needed, then remove support completely. Individual orientations and support notes are in the [print table](../scripts/two_belt_print_estimates.csv).

CAD/STL checks cover valid single solids, watertight meshes, bed placement, dimensions and the A1 mini envelope. The gusseted frame also has CAD checks for bearing insertion, bridge seating and driver/wrench access. Seven other component CAD, STEP and STL files are unchanged from the previous two-piece release. Physical fit and loaded stiffness need checks with the actual hardware.

Earlier versions are in the [Lisa archive](../../../archive/lisa/). Regenerate the complete set from repository root with `projects/lisa/scripts/build_two_belt.sh`.
