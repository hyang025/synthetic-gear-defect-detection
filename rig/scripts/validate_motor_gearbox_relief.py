"""Check the approved motor recess change against its archived native model.

Run with scripts/freecad-python.sh from the original repository, which contains
the pre-change archive named in motor_gearbox_relief_archive_manifest.json.
"""
from pathlib import Path
import hashlib
import json
import math

import FreeCAD as App
import Part

PROJECT=Path(__file__).resolve().parents[1]
SCRIPTS=PROJECT/'scripts'


def main():
    manifest=json.loads((SCRIPTS/'motor_gearbox_relief_archive_manifest.json').read_text())
    archive=PROJECT.parents[1]/manifest['archive_path']
    relative='motor holder/output/model/motor_holder_r06.FCStd'
    old_doc=App.openDocument(str(archive/relative))
    new_doc=App.openDocument(str(PROJECT/relative))
    old=old_doc.getObject('MotorHolderStrong').Shape
    new=new_doc.getObject('MotorHolderStrong').Shape
    assert new.isValid() and len(new.Solids)==1
    new.check(True)
    old_seat=old_doc.getObject('MotorUndersideHousingSeat').Shape.BoundBox
    new_seat=new_doc.getObject('MotorUndersideHousingSeat').Shape.BoundBox
    assert math.isclose(old_seat.XMin,-14.6,abs_tol=1e-7)
    assert math.isclose(new_seat.XMin,-16.15,abs_tol=1e-7)
    for field in ['XMax','YMin','YMax','ZMin','ZMax']:
        assert math.isclose(getattr(old_seat,field),getattr(new_seat,field),abs_tol=1e-7),field
    for field in ['XMin','XMax','YMin','YMax','ZMin','ZMax']:
        assert math.isclose(getattr(old.BoundBox,field),getattr(new.BoundBox,field),abs_tol=1e-7),field
    added=new.cut(old).Volume
    assert added<1e-6
    removed=old.cut(new)
    relief_box=Part.makeBox(1.55,35.2,4,App.Vector(-16.15,-17.6,32))
    assert removed.cut(relief_box).Volume<1e-5
    assert relief_box.cut(removed).Volume<1e-5
    assert math.isclose(removed.Volume,218.24,abs_tol=1e-5)

    def holes(doc):
        return sorted((round(o.Placement.Base.x,6),round(o.Placement.Base.y,6),round(o.Radius.Value,6))
                      for o in doc.Objects if o.Name.startswith('MotorAttachment') and o.TypeId=='Part::Cylinder')
    assert holes(old_doc)==holes(new_doc) and len(holes(new_doc))==4
    case_edge=-9.7-7.7/2-2
    assert math.isclose(case_edge-new_seat.XMin,.6,abs_tol=1e-7)
    assert math.isclose(new_seat.XMin-new.BoundBox.XMin,3.85,abs_tol=1e-7)
    assert math.isclose(new.BoundBox.ZMax-new_seat.ZMax,3,abs_tol=1e-7)
    assert math.isclose(new_seat.ZMax+6,42,abs_tol=1e-7)
    for y in [-9,0,9]:
        for z in [32.1,34,35.9]:
            assert not new.isInside(App.Vector(case_edge+.01,y,z),1e-7,False)
            assert new.isInside(App.Vector(-16.16,y,z),1e-7,False)
        assert new.isInside(App.Vector(case_edge+.01,y,36.1),1e-7,False)
    step=Part.Shape();step.read(str(PROJECT/'motor holder/output/model/motor_holder_r06.step'))
    assert step.isValid() and len(step.Solids)==1
    # STEP translation perturbs trim tolerances slightly; require agreement
    # within one part per million, well below mesh-export deflection.
    step_error=abs(step.Volume/new.Volume-1)
    assert step_error<1e-6,step_error
    current=json.loads((SCRIPTS/'two_belt_r06_report.json').read_text())
    unchanged=[];changed_since_baseline=[]
    for relative,digest in manifest['original_STL_sha256'].items():
        if relative=='motor holder/motor_holder_r06.stl':continue
        if hashlib.sha256((PROJECT/current['parts'][Path(relative).stem]['stl']).read_bytes()).hexdigest()==digest:
            unchanged.append(current['parts'][Path(relative).stem]['stl'])
        else:
            changed_since_baseline.append(current['parts'][Path(relative).stem]['stl'])
    report={'valid_single_solid':True,'CAD_BOP_check_passed':True,'STEP_reopened':True,
        'STEP_relative_volume_error':step_error,
        'previous_pocket_wall_x_mm':old_seat.XMin,'new_pocket_wall_x_mm':new_seat.XMin,
        'relief_width_mm':old_seat.XMin-new_seat.XMin,'removed_volume_mm3':removed.Volume,
        'only_approved_relief_volume_changed':True,'added_material_mm3':added,
        'user_pad_diameter_mm':7.7,'user_pad_edge_to_case_edge_mm':2,
        'screw_centre_to_case_edge_mm':5.85,'screw_centre_to_pocket_wall_mm':6.45,
        'nominal_fit_clearance_mm':case_edge-new_seat.XMin,
        'remaining_lip_mm':new_seat.XMin-new.BoundBox.XMin,
        'mounting_plate_floor_mm':new.BoundBox.ZMax-new_seat.ZMax,
        'motor_seat_above_board_mm':new_seat.ZMax+6,
        'screw_hole_centres_and_diameters_unchanged':True,
        'other_STLs_unchanged':unchanged,
        'other_STLs_changed_since_motor_relief_baseline':changed_since_baseline,
        'physical_fit_verified':False}
    (SCRIPTS/'motor_gearbox_relief_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Validated 1.55 mm relief, 0.60 mm clearance, 3.85 mm lip and 3 mm mounting floor; {len(changed_since_baseline)} other STLs revised since that baseline.')
    print(f'STEP relative volume error: {step_error:.3g}; removed material: {removed.Volume:.2f} mm3.')
    App.closeDocument(new_doc.Name);App.closeDocument(old_doc.Name)


if __name__=='__main__':main()
