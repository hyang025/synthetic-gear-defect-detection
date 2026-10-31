"""Write focused notes for the current Lisa component and assembly folders."""
import json
import os
from pathlib import Path
from urllib.parse import quote

SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent
REPORT=json.loads((SCRIPTS/'two_belt_r06_report.json').read_text())
SCANS=json.loads((SCRIPTS/'scan_sources.json').read_text())['references']


def link(path,folder):
    return quote(os.path.relpath(path,folder),safe='/')


def main():
    total=REPORT['assembly_material_estimate']
    groups={folder:[(name,p) for name,p in REPORT['parts'].items() if Path(p['stl']).parts[0]==folder]
            for folder in ['motor holder','encoder holder','shaft frame','gear adapter']}
    intro={
        'motor holder':('Motor holder and slide base',[
            'Moving motor holder and its 15 mm slide base. Motor face height is 42 mm above the board.',
            'Mounting holes: four Ø3.8 mm holes on a 33 × 18 mm pattern; shaft offset from the near column is 9.7 mm. The pattern is photo-derived (approximately ±0.3 mm).',
            'The gearbox recess includes the approved 1.55 mm near-side relief. User measurements: raised pad OD7.7 mm and 2 mm pad-to-case edge gap. Recess wall: 6.45 mm from the screw centre, providing 0.6 mm nominal clearance and a 3.85 mm lip. Pocket depth is 4 mm; mounting plate floor is 3 mm.',
            'Use four M3 × 6 motor screws with four approximately 0.6 mm washers; nominal thread engagement is 2.4 mm. Actual usable thread depth and revised housing fit need checking.',
            'The base uses four M4 × 12 clamps, four 1 mm washers and four captive M4 nuts, with 5 mm holder feet on a 6 mm base. Four #6 × 1/2 inch flat-head wood screws secure the base.']),
        'encoder holder':('Encoder holder and slide base',[
            'Current encoder holder and its 15 mm slide base. Encoder face height is 58 mm above the board.',
            'The user previously confirmed the encoder mounting pattern. Slots are 5.5 × 3.8 mm at radius 14.5 mm, at 0/30, 120/150 and 240/270 degrees. Use the actual three-hole mounting group and retain case fasteners.',
            'The underside seat is Ø39.5 mm × 2 mm deep, leaving a 5 mm plate floor. The boss opening is Ø21.5 mm. Housing dimensions and exact boss height remain reference estimates.',
            'Use three M3 × 8 screws with approximately 0.6 mm washers for nominal 2.4 mm engagement. The alternative is M3 × 10 with an additional rigid 2 mm spacer per screw. Check actual usable thread depth and pulley clearance.',
            'The base uses four M4 × 12 clamps, four 1 mm washers and four captive M4 nuts, with 5 mm holder feet on a 6 mm base. Four #6 × 1/2 inch flat-head wood screws secure the base.']),
        'shaft frame':('Shaft frame',[
            'Two current components: one rigid body with integrated feet, uprights and lower bearing deck, plus a removable upper bearing bridge. The two X sides remain open for bearing insertion and tools.',
            'The first-level rectangular screw-joint flanges are removed. Twelve local triangular gussets brace the feet, lower deck and upper bridge seats: four at each level, 4 mm thick, centred at X±24 mm. Foot ribs reach 8 mm and rise 16 mm; lower-deck ribs reach 12 mm and drop 12 mm; seat ribs reach 8 mm and drop 10 mm. These are selected design dimensions; stiffness and load capacity have not been measured.',
            'Installed envelope stays 64 × 124 × 92 mm. Lower deck is 6 mm thick at Z43.5–49.5 mm; upright walls are 8 mm thick. Four upper bridge joints use round Ø4.4 mm holes at X±24, Z78 mm. Bearing mounting pattern stays 32 × 32 mm with Ø4.4 mm holes and Ø8 mm shaft openings.',
            'The narrow upper bridge seats remain functional: they carry the removable bridge and hold it in position while the screws are fitted. They are 4.5 mm thick with seating faces at Z70 mm, reinforced by four local ribs. Two Ø4 × 2 mm pins have 0.5 mm lead-ins. The front socket is Ø4.5 mm; the rear slot is 4.5 × 5.5 mm, both 2.25 mm deep. Assumed fit allowances: 0.25 mm at each bridge end, 0.25 mm radial pin clearance and 0.25 mm axial clearance; rear slot adds 1 mm along Y.',
            'Use eight M4 × 16 bearing screws, four M4 × 25 bridge screws and twelve plain M4 nuts for this frame. Four #6 × 1/2 inch wood screws secure the integrated feet. Omit extra bearing washers to retain upper-pulley clearance. No adhesive is needed.',
            'Bench assembly: remove all deck supports; slide the lower bearing in from an open X side and bolt it under the deck before mounting to plywood. Prefit the upper bearing to its bridge. Load the bare 6 mm shaft, both 80T pulleys and both closed belts with the bridge removed. Lower the bridge and bearing over the shaft, locate the pins and seat both shoulders. Start all four side bolts, hold the nuts through the open sides and tighten evenly. Add and adjust collars and gear afterwards; do not force axial bearing preload.',
            'The 6 mm D shaft is 100 mm long. Bearing centres remain Z40/Z95.5 mm and belt centres Z61/Z77 mm. The illustrated manual includes the bearing sandwich sequence, screw directions and tool access. Main body prints feet down with support under its lower bearing deck; bridge prints broad deck down. Remove supports through the open sides before assembly.']),
        'gear adapter':('Gear adapter',[
            'Two current components: keyed sleeve with support flange, and removable clamp cap. Print these small parts first to check the actual gear and shaft fit.',
            'Sleeve OD11.9 mm; 2.8 mm key, outer X=7.2 mm and 10 mm key height; sleeve height 17 mm above a 2 mm flange. Shaft bore is 6.3 mm with a D flat at Y=2.65 mm. The sleeve split is 1.2 mm.',
            'Cap OD28 mm, sleeve bore Ø12.2 mm and height 8 mm; split 2.2 mm. The M3 passage is round Ø3.4 mm and the retained nut pocket is 5.8 mm across flats × 2.8 mm deep. Use one M3 × 16 screw and one M3 hex nut.',
            'Gear bore and key fit are photo-derived. Actual grip, torque capacity and long-term PLA clamp durability need physical checks.'])}
    for folder,parts in groups.items():
        directory=ROOT/folder;directory.mkdir(parents=True,exist_ok=True)
        title,paragraphs=intro[folder]
        lines=['# '+title,'']
        for paragraph in paragraphs:lines.extend([paragraph,''])
        lines.extend(['## Current print files','',
            '| Component | Quantity | STL | PLA estimate | Filament cost |',
            '| --- | ---: | --- | ---: | ---: |'])
        for name,part in parts:
            e=part['material_estimate'];stl=ROOT/part['stl']
            lines.append(f"| {name.removesuffix('_r06').replace('_',' ')} | {part['default_quantity']} | [{stl.name}]({link(stl,directory)}) | {e['estimated_filament_g']:.1f} g | USD {e['estimated_filament_cost_usd']:.2f} |")
        grams=sum(p['material_estimate']['estimated_filament_g'] for _,p in parts)
        cost=sum(p['material_estimate']['estimated_filament_cost_usd'] for _,p in parts)
        lines.extend(['',f'Folder total: approximately **{grams:.1f} g PLA / USD {cost:.2f}**.',
            'Estimates use PLA at USD20/kg, density 1.24 g/cm³, 5% waste and the existing 0.6 mm voxel shell estimate. Allow approximately ±20%; supports and hardware are excluded. Confirm with the slicer.','',
            'Editable CAD and STEP files are in [output/model](output/model/). Dimensioned drawings and previews are in [output/render](output/render/).',
            f'Use the [assembly manual]({link(ROOT/"manual.pdf",directory)}) for illustrated steps, exact print orientations, hardware quantities and support placement.','',
            '## Printing and checks','',
            'Bambu Lab A1 mini, 0.4 mm nozzle: PLA, 0.20 mm layers, six walls (approximately 2.5–2.7 mm), six top/bottom layers (1.2 mm). Use 40% gyroid for structural parts and 100% infill for both gear-adapter pieces. Optional 3 mm brim.',
            'Fit, strength and fastening take priority. Support flat M4 washer lands, horizontal fastener bores and nut-pocket roofs where needed, then remove support completely. Individual orientations and support notes are in the [print table](../scripts/two_belt_print_estimates.csv).','',
            'CAD/STL checks cover valid single solids, watertight meshes, bed placement, dimensions and the A1 mini envelope. The gusseted frame also has CAD checks for bearing insertion, bridge seating and driver/wrench access. Seven other component CAD, STEP and STL files are unchanged from the previous two-piece release. Physical fit and loaded stiffness need checks with the actual hardware.',''])
        photos=[s for s in SCANS if Path(s['path']).parts[0]==folder]
        if photos:
            lines.extend(['## Reference photographs','',
                '| Original inbox name | Current reference | View |','| --- | --- | --- |'])
            for photo in photos:
                lines.append(f"| {photo['original_inbox_name']} | [{Path(photo['path']).name}]({link(ROOT/photo['path'],directory)}) | {photo['view']} |")
            lines.extend(['','Original image data and hashes are preserved in [scan_sources.json](../scripts/scan_sources.json).',''])
        lines.extend(['Earlier versions are in the [Lisa archive](../../../archive/lisa/). Regenerate the complete set from repository root with `projects/lisa/scripts/build_two_belt.sh`.',''])
        (directory/'readme.md').write_text('\n'.join(lines))

    assembly=ROOT/'assembly'
    lines=['# Lisa assembly','',
        'Shared assembly CAD, renders, reference inputs and download bundles for the current eight-piece set. The combined manual stays at the project root.','',
        '- [Assembly manual](../manual.pdf)',
        '- [Assembly CAD](output/model/two_belt_assembly_r06.FCStd)',
        '- [All eight STLs](output/model/lisa_two_belt_r06_stls.zip)',
        '- [CAD, drawings and source bundle](output/model/lisa_two_belt_r06.zip)',
        '- [Assembly views](output/render/)',
        '- [Parts and hardware quantities](../scripts/two_belt_parts.csv)','',
        'Current reference inputs are under `output/model/reference/`. The gear reference is the exact saved shape extracted from the archived shaft-concept document; obsolete coupling and bearing inputs are archived. Supplier provenance remains in the project scripts.','',
        '## Shared reference photographs','',
        '| Original inbox name | Current reference | View |','| --- | --- | --- |']
    for photo in SCANS:
        if Path(photo['path']).parts[0]=='assembly':
            lines.append(f"| {photo['original_inbox_name']} | [{Path(photo['path']).name}]({link(ROOT/photo['path'],assembly)}) | {photo['view']} |")
    lines.extend(['','Image hashes and previous paths are in [scan_sources.json](../scripts/scan_sources.json).',''])
    (assembly/'readme.md').write_text('\n'.join(lines))

    lines=['# Lisa: two-belt motor, gear shaft and encoder assembly','',
        'Motor 20T → lower shaft 80T gives quarter-speed at the gear. Upper shaft 80T → encoder 20T returns the encoder to motor speed. Both belts are closed-loop 250 × 6 mm GT2 / 2GT, 2 mm pitch. Both device slides have 15 mm adjustment. The user reports the components are on hand; actual specifications and fit need checking.','',
        '| Folder | Current printed pieces | Contents |','| --- | ---: | --- |',
        '| [motor holder](motor%20holder/) | 2 | Motor holder and slide base |',
        '| [encoder holder](encoder%20holder/) | 2 | Encoder holder and slide base |',
        '| [shaft frame](shaft%20frame/) | 2 | Integrated body and removable upper bridge |',
        '| [gear adapter](gear%20adapter/) | 2 | Keyed sleeve and clamp cap |',
        '| [assembly](assembly/) | 0 | Assembly CAD, shared renders, inputs and bundles |','',
        '- [Assembly manual](manual.pdf)',
        '- [All eight STLs](assembly/output/model/lisa_two_belt_r06_stls.zip)',
        '- [Complete CAD, STEP, drawings and sources](assembly/output/model/lisa_two_belt_r06.zip)',
        '- [Hardware quantities](scripts/two_belt_parts.csv) and [print estimates](scripts/two_belt_print_estimates.csv)','',
        'The motor recess includes the 1.55 mm near-side relief based on the user\'s 7.7 mm pad diameter and 2 mm pad-to-case gap. The nominal clearance is 0.6 mm; the remaining lip is 3.85 mm. The frame uses one body with twelve gussets and a removable upper bridge. Old first-level screw flanges are removed; the narrow upper bridge seats remain. Detailed dimensions and sources are in the component notes and manual.','',
        '## Printing','',
        'A1 mini / 0.4 mm nozzle: PLA, 0.20 mm layers, six walls (approximately 2.5–2.7 mm), six top/bottom layers (1.2 mm), 40% gyroid. Both adapter pieces use 100% infill. Optional 3 mm brim. Fit, strength and fastening take priority; use supports under the integral lower bearing deck and where needed at washer lands, horizontal holes and nut-pocket roofs.',
        f"Complete set estimate: **{total['estimated_filament_g']:.1f} g PLA / USD {total['estimated_filament_cost_usd']:.2f}** at USD20/kg, density 1.24 g/cm³ and 5% waste. Geometric shell/infill estimate, approximately ±20%; supports, hardware, electricity and labour are excluded.",'',
        'Current CAD and mesh checks passed, including two-piece frame insertion and tool access. Seven other component models are unchanged. Actual gear fit, motor fit, pulley grip and loaded stiffness need physical checks.','',
        'Rebuild from repository root with `projects/lisa/scripts/build_two_belt.sh`. Parameters, scripts and reports stay in [scripts](scripts/). Previous designs and notes are in the [Lisa archive](../../archive/lisa/). The [cleanup manifest](scripts/folder_cleanup_manifest.json) records file moves and hashes.','']
    (ROOT/'readme.md').write_text('\n'.join(lines))
    print('Updated four focused component notes, the assembly index and project overview.')


if __name__=='__main__':main()
