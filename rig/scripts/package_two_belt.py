"""Package exactly the eight current print files and their editable sources."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from project_paths import ASSEMBLY_MODEL, ASSEMBLY_RENDER, REFERENCE

SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent
REPORT=json.loads((SCRIPTS/'two_belt_r06_report.json').read_text())
MODEL=ASSEMBLY_MODEL


def main():
    stls=[ROOT/p['stl'] for p in REPORT['parts'].values()]
    assert len(stls)==len(REPORT['parts']) and all(p.exists() for p in stls)
    print_zip=MODEL/'lisa_two_belt_r06_stls.zip'
    with ZipFile(print_zip,'w',ZIP_DEFLATED) as archive:
        for path in stls:archive.write(path,path.name)
    part_folders={Path(p['stl']).parts[0] for p in REPORT['parts'].values()}
    files={ROOT/'readme.md',ROOT/'manual.pdf',ROOT/'assembly/readme.md',*stls}
    files.update(ROOT/folder/'readme.md' for folder in part_folders)
    files.update(SCRIPTS.glob('*.py')); files.update(SCRIPTS.glob('*.sh'))
    files.update(SCRIPTS.glob('*.json')); files.update(SCRIPTS.glob('*.csv'))
    for part in REPORT['parts'].values():
        for key in ('cad','step','preview_render'):files.add(ROOT/part[key])
        drawing=ROOT/part['drawings']
        files.update(drawing.with_suffix(ext) for ext in ('.svg','.png','.pdf'))
        if part.get('seat_detail'):
            files.update((ROOT/part['seat_detail']).with_suffix(ext) for ext in ('.svg','.png','.pdf'))
    files.add(ROOT/REPORT['assembly_cad'])
    # Dependencies used to reconstruct the known device interfaces/reference gear.
    files.add(ROOT/'motor holder/output/model/reference/motor_encoder_assembly_reference.FCStd')
    files.update(p for p in REFERENCE.rglob('*') if p.is_file())
    files.update(ASSEMBLY_RENDER.glob('two_belt_*_r06.png'))
    files.update(ASSEMBLY_RENDER.glob('two_belt_*_r06_raw.png'))
    files.update(ASSEMBLY_RENDER.glob('two_belt_*_r06.blend'))
    files.add(ASSEMBLY_RENDER/'assembly_manual.pdf')
    files.update(ROOT/folder/'output/render/assembly_manual.pdf' for folder in part_folders)
    files.add(ROOT/'motor holder/output/render/reference/motor_mount_measurement_reference_r05.png')
    files.update((ROOT/'motor holder/scan').glob('*'))
    for folder in part_folders|{'assembly'}:
        files.update((ROOT/folder/'scan').glob('*'))
    bundle=MODEL/'lisa_two_belt_r06.zip'
    with ZipFile(bundle,'w',ZIP_DEFLATED) as archive:
        for path in sorted(files):
            assert path.is_file(),path
            archive.write(path,Path('lisa')/path.relative_to(ROOT))
        archive.writestr('START_HERE.txt',
            'LISA R06: TWO-BELT PRINTABLE FIT-TEST SET\n'
            'Start with lisa/manual.pdf for all drawings, hardware, orientation and assembly.\n'
            'Print one of each of the eight r06 STLs in motor holder, encoder holder, shaft frame and gear adapter.\n'
            'Assembly CAD, shared renders and download bundles are under lisa/assembly/output.\n'
            'All STLs are millimeters and lie on the printing bed in intended orientation.\n'
            'Motor gearbox recess: near-side wall relieved by 1.55 mm; 0.60 mm nominal clearance.\n'
            'User dimensions: raised circular pad OD7.70 mm; 2.00 mm pad-to-gearbox-edge gap.\n'
            'See the motor-holder underside detail in the manual and dry-fit the revised holder.\n'
            'Print the two small gear-adapter pieces first; actual gear fit is photo-derived.\n'
            'PLA: 0.20 mm layers, six walls, six top/bottom layers, 40% gyroid.\n'
            f"Both gear-adapter pieces use 100% infill. Approximate total: {REPORT['assembly_material_estimate']['estimated_filament_g']:.1f} g / USD {REPORT['assembly_material_estimate']['estimated_filament_cost_usd']:.2f}.\n"
            'Local supports at four M4 washer lands on each holder: eight small areas total.\n'
            'Fit, strength and fastening take priority; support-free printing is not a goal.\n'
            'Frame bridge and clamp cap have round fastener bores.\n'
            'Support beneath the integral lower bearing deck; remove through the open sides.\n'
            'Support horizontal M4/M3 passages and the nut-pocket roof as needed; clean fully.\n'
            'Motor20T -> shaft80T; shaft80T -> encoder20T. Two 250x6 mm GT2 belts.\n'
            'Gusseted shaft frame and removable upper bearing bridge. Both device slides adjust.\n'
            'Obsolete first-level screw ledges removed; twelve local gussets brace the frame.\n'
            'Both slides use M4x12 and 1 mm washers. Bearings: eight M4x16; bridge: four M4x25.\n'
            'Fit lower bearing before mounting to wood. Load both pulleys and belts before closing bridge.\n'
            'Twelve #6x1/2 inch flat-head wood screws mount the bases and frame to plywood.\n'
            'Motor M3x6; encoder M3x8; adapter clamp M3x16 plus retained M3 nut.\n'
            'Physical fit, shaft/pulley grip and loaded stiffness require checks; no load rating.\n'
            'CAD/STEP and source Blender scenes are under output/model and output/render.\n'
            'Scripts, parameter sources, supplier links, quantities and reports are in lisa/scripts.\n'
            'Rebuild using the toolchain installed in the original 3dprint_generator repository.\n')
    for path in (print_zip,bundle):
        with ZipFile(path) as archive:
            assert archive.testzip() is None
            names=[name for name in archive.namelist() if name.endswith('.stl')]
            assert len(names)==len(REPORT['parts']) and all('_r06.stl' in name for name in names)
        print('Packaged',path.relative_to(ROOT),path.stat().st_size,'bytes')
    expected=set(stls)
    actual={p for folder in part_folders for p in (ROOT/folder).glob('*.stl')}
    assert actual==expected,'Unexpected print files in active component folders'
    print('Packaged eight current r06 print files from four component folders.')


if __name__=='__main__':main()
