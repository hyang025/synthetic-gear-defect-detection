"""Build the approved two-belt parts with editable CSG, STL and native TechDraw.

Run inside FreeCAD's Qt event loop using build_two_belt.sh. The legacy fitted
device interfaces and the approved draft layout are reused; r05 sources stay
available as references. New output is a physical fit-test release, not a load
rating. All lengths are mm and all released STLs lie on the printing bed.
"""
import json
import math
import os
import sys
from pathlib import Path

import FreeCAD as App
import FreeCADGui as Gui
import Import
import MeshPart
import Part

SCRIPTS=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPTS))
import generate_mounts as cad
import generate_two_belt_draft as layout
from project_paths import part_folder, ASSEMBLY_MODEL, SUPPLIER_REFERENCE

ROOT=SCRIPTS.parent
PARAMS=json.loads((SCRIPTS/'two_belt_parameters.json').read_text())
CFG={**cad.CONFIG,**PARAMS}
cad.CONFIG=CFG
cad.REV='r06'
cad.REPORT={'revision':CFG['revision'],'parameters':CFG,'parts':{},'drawings':[]}
V=App.Vector
BUILD=ROOT/'.build/two_belt_r06'
BUILD.mkdir(parents=True,exist_ok=True)
layout.MESHES=BUILD
ASSEMBLY_ONLY=os.environ.get('LISA_ASSEMBLY_ONLY')=='1'
REBUILD_PARTS=set(filter(None,os.environ.get('LISA_REBUILD_PARTS','').split(',')))
RELEASE=(json.loads((SCRIPTS/'two_belt_r06_report.json').read_text()) if ASSEMBLY_ONLY or REBUILD_PARTS else
         {'revision':CFG['revision'],'parameters':CFG,'parts':{},'drawings':[]})
if ASSEMBLY_ONLY:
    assert RELEASE['parameters']==CFG,'Assembly-only rebuild requires unchanged printed-part parameters'
if REBUILD_PARTS:
    RELEASE['parameters']=CFG
    RELEASE['revision']=CFG['revision']
    RELEASE.pop('drawing_layout_verification',None)
SOURCE_DOCS=[]


def param_box(b,name,x,y,z,dx,dy,dz,expressions=None):
    obj=b.box(name,x,y,z,dx,dy,dz)
    for key,value in (expressions or {}).items():
        obj.setExpression(key,'Parameters.'+value)
    return obj


def y_tool(b,name,x,y,z,length):
    obj=b.cylinder(name+'Round',x,y,z,CFG['m4_clearance_mm']/2,length)
    obj.Placement.Rotation=App.Rotation(V(0,0,1),V(0,1,0))
    obj.setExpression('Radius','Parameters.M4ClearanceMm / 2')
    return obj


def panel(b,joint_levels=(8,51.5,72)):
    body=param_box(b,'PanelBlank',-32,0,0,64,8,86,
        {'Length':'FrameWidthMm','Width':'FramePanelThicknessMm','Height':'FramePanelHeightMm'})
    tools=[]
    for i,z in enumerate((4,46)):
        tools.append(param_box(b,'Window'+str(i),-18,-1,z,36,10,31,
            {'Length':'FrameWindowWidthMm','Height':'FrameWindowHeightMm'}))
    for x in (-24,24):
        for z in joint_levels:tools.append(y_tool(b,f'Join{x}{z}'.replace('.','_'),x,-1,z,10))
    return b.cut('FramePanel',body,b.union('PanelTools',tools))


def foot(b,front,join_holes=True,include_flange=True):
    flange=22 if front else 0
    pieces=[param_box(b,'FootBase',-32,0,0,64,30,6,
        {'Length':'FrameWidthMm','Width':'FootDepthMm','Height':'FootThicknessMm'})]
    if include_flange:pieces.append(b.box('FootFlange',-32,flange,6,64,8,16))
    body=b.union('FootStructure',pieces)
    tools=[]
    for x in (-24,24):
        if join_holes:tools.append(y_tool(b,f'FootJoin{x}',x,flange-1,14,10))
        tools.append(cad.countersunk_wood_hole(b,f'FootWood{x}',x,6 if front else 24,6,6))
    return b.cut('FootRail',body,b.union('FootTools',tools))


def bearing_plate(b,upper,join_holes=True,include_flanges=True):
    deck_z=16 if upper else 0
    flange_z=0 if upper else 6
    pieces=[param_box(b,'BearingDeck',-32,-40,deck_z,64,80,6,
            {'Length':'FrameWidthMm','Width':'BearingPlateDepthMm','Height':'BearingPlateThicknessMm'})]
    if include_flanges:
        pieces.extend(b.box('EdgeFlange'+str(i),-32,y,flange_z,64,8,16) for i,y in enumerate((-40,32)))
    body=b.union('PlateStructure',pieces)
    center=b.cylinder('ShaftClearance',0,0,deck_z-1,4,8)
    center.setExpression('Radius','Parameters.ShaftClearanceMm / 2')
    tools=[center]
    for x in (-16,16):
        for y in (-16,16):
            hole=b.cylinder(f'BearingMount{x}{y}',x,y,deck_z-1,2.2,8)
            hole.setExpression('Radius','Parameters.M4ClearanceMm / 2')
            hole.setExpression('Placement.Base.x',('' if x>0 else '-')+'Parameters.BearingMountPitchMm / 2')
            hole.setExpression('Placement.Base.y',('' if y>0 else '-')+'Parameters.BearingMountPitchMm / 2')
            tools.append(hole)
    if join_holes:
        for x in (-24,24):
            for y in (-40,32):tools.append(y_tool(b,f'PlateJoin{x}{y}',x,y-1,8 if upper else 14,10))
    return b.cut('BearingPlate',body,b.union('PlateTools',tools))


def frame_body(b):
    """One solid base; keep the two X sides open for bearing and tool access."""
    pieces=[]
    for front in (True,False):
        upright=panel(b,(72,))
        upright.Placement.Base=V(0,-48 if front else 40,6)
        rail=foot(b,front,False,False)
        rail.Placement.Base=V(0,-62 if front else 32,0)
        pieces.extend([upright,rail])
    lower=bearing_plate(b,False,False,False)
    lower.Placement.Base=V(0,0,43.5)
    pieces.append(lower)
    for i,y in enumerate((-42,32)):
        seat=param_box(b,'BridgeSeat'+str(i),-32,y,65.5,64,10,4.5,
            {'Height':'BridgeSeatThicknessMm','Placement.Base.z':'BridgeSeatZMm - Parameters.BridgeSeatThicknessMm'})
        pieces.append(seat)
    # Local ribs replace the old full-width foot and lower-plate screw flanges.
    # Keep the lower housing's side-entry path and both bearing wrench paths open.
    thickness=CFG['frame_gusset_thickness_mm']
    for side in (-1,1):
        for x in (-24,24):
            label=('Front' if side<0 else 'Rear')+('Left' if x<0 else 'Right')
            wall=side*40
            for kind,z,reach,height,under in (
                ('Foot',6,CFG['frame_foot_gusset_reach_mm'],CFG['frame_foot_gusset_height_mm'],False),
                ('LowerDeck',43.5,CFG['frame_deck_gusset_reach_mm'],CFG['frame_deck_gusset_height_mm'],True),
                ('BridgeSeat',65.5,CFG['frame_seat_gusset_reach_mm'],CFG['frame_seat_gusset_height_mm'],True)):
                points=[(wall,z),(wall-side*reach,z),(wall,z-height if under else z+height)]
                pieces.append(b.triangular_ramp(label+kind+'Gusset',x-thickness/2,thickness,points))
    for i,y in enumerate((-36,36)):
        pin=b.cylinder('BridgeLocator'+str(i),0,y,70,2,1.5)
        pin.setExpression('Radius','Parameters.BridgeLocatorDiameterMm / 2')
        pin.setExpression('Height','Parameters.BridgeLocatorHeightMm - Parameters.BridgeLocatorLeadMm')
        pin.setExpression('Placement.Base.z','Parameters.BridgeSeatZMm')
        # Lead-in chamfer guides the bridge onto its two seats without bolt force.
        tip=b.doc.addObject('Part::Cone','LocatorLead'+str(i))
        tip.Radius1=2;tip.Radius2=1.5;tip.Height=.5
        tip.Placement.Base=V(0,y,71.5)
        tip.setExpression('Radius1','Parameters.BridgeLocatorDiameterMm / 2')
        tip.setExpression('Radius2','Parameters.BridgeLocatorDiameterMm / 2 - Parameters.BridgeLocatorLeadMm')
        tip.setExpression('Height','Parameters.BridgeLocatorLeadMm')
        tip.setExpression('Placement.Base.z','Parameters.BridgeSeatZMm + Parameters.BridgeLocatorHeightMm - Parameters.BridgeLocatorLeadMm')
        pieces.extend([pin,tip])
    return b.union('IntegratedShaftFrame',pieces)


def upper_bridge(b):
    bridge=bearing_plate(b,True)
    tools=[]
    for i,y in enumerate((-40,40-CFG['bridge_end_clearance_mm'])):
        strip=b.box('BridgeEndRelief'+str(i),-33,y,-1,66,CFG['bridge_end_clearance_mm'],24)
        strip.setExpression('Width','Parameters.BridgeEndClearanceMm')
        tools.append(strip)
    depth=CFG['bridge_locator_height_mm']+CFG['bridge_locator_axial_clearance_mm']
    width=CFG['bridge_locator_diameter_mm']+2*CFG['bridge_locator_radial_clearance_mm']
    front=b.cylinder('FrontLocatorSocket',0,-36,-.1,width/2,depth+.1)
    front.setExpression('Radius','Parameters.BridgeLocatorDiameterMm / 2 + Parameters.BridgeLocatorRadialClearanceMm')
    front.setExpression('Height','Parameters.BridgeLocatorHeightMm + Parameters.BridgeLocatorAxialClearanceMm + 0.1')
    rear=b.capsule('RearLocatorSlot',0,36,-.1,width+CFG['bridge_rear_slot_extra_length_mm'],width,depth+.1,90)
    tools.extend([front,rear])
    return b.cut('UpperBearingBridge',bridge,b.union('BridgeAssemblyClearances',tools))


def remove_layout(names):
    for name in names:
        entry=next(e for e in layout.REPORT['items'] if e['name']==name)
        layout.DOC.removeObject(entry['cad_object'])
        del layout.SHAPES[name]
        (layout.MESHES/(name+'.stl')).unlink(missing_ok=True)
        if name in layout.PRINTED:layout.PRINTED.remove(name)
        for group in ('items','detail_items','exploded_items'):
            layout.REPORT[group]=[e for e in layout.REPORT[group] if e['name']!=name]


def add_layout_part(name,shape):
    if name in layout.SHAPES:update_layout(name,shape)
    elif name.startswith('Frame'):
        layout.add(name,shape,'frame','integrated body with twelve gussets; obsolete screw ledges removed',True)
    else:layout.add(name,shape,'adapter','removable cap clamps the keyed sleeve around the D shaft',True)


def encoder_base(b):
    body=b.box('EncoderBaseBlank',-38,-52,0,76,104,6)
    body.setExpression('Height','Parameters.SlideBaseThicknessMm')
    tools=[]
    for x in (-24,24):
        for y in (-36,36):
            tools.append(b.capsule(f'ClampSlot{x}{y}',x,y,-1,19.6,4.6,8))
            tools.append(b.box(f'NutChannel{x}{y}',x-11.8,y-3.7,-.1,23.6,7.4,3.5))
    for x in (-31,31):
        for y in (-46,46):tools.append(cad.countersunk_wood_hole(b,f'WoodSeat{x}{y}',x,y,6,6))
    return b.cut('EncoderSlideBase',body,b.union('EncoderBaseTools',tools))


def sleeve(b):
    outer=b.cylinder('SleeveOutside',0,0,2,CFG['adapter_sleeve_od_mm']/2,CFG['adapter_sleeve_height_mm'])
    outer.setExpression('Radius','Parameters.AdapterSleeveOdMm / 2')
    outer.setExpression('Height','Parameters.AdapterSleeveHeightMm')
    flange=b.cylinder('GearSupportFlange',0,0,0,10,2)
    flange.setExpression('Radius','Parameters.AdapterFlangeOdMm / 2')
    flange.setExpression('Height','Parameters.AdapterFlangeThicknessMm')
    key=b.box('GearKey',5.6,-CFG['adapter_key_width_mm']/2,2,CFG['adapter_key_outer_x_mm']-5.6,CFG['adapter_key_width_mm'],10)
    key.setExpression('Width','Parameters.AdapterKeyWidthMm')
    key.setExpression('Height','Parameters.AdapterKeyHeightMm')
    bore=b.cylinder('RoundBore',0,0,-1,CFG['adapter_shaft_bore_mm']/2,22)
    bore.setExpression('Radius','Parameters.AdapterShaftBoreMm / 2')
    flat=b.box('DFlat',-5,CFG['adapter_shaft_flat_y_mm'],-2,10,5,24)
    flat.setExpression('Placement.Base.y','Parameters.AdapterShaftFlatYMm')
    d_bore=b.cut('DShaftBore',bore,flat)
    split=b.box('SleeveSplit',-CFG['adapter_split_width_mm']/2,-12,-1,CFG['adapter_split_width_mm'],12,22)
    split.setExpression('Width','12')
    split.setExpression('Length','Parameters.AdapterSplitWidthMm')
    split.setExpression('Placement.Base.x','-Parameters.AdapterSplitWidthMm / 2')
    return b.cut('KeyedSleeve',b.union('SleeveBlank',[outer,flange,key]),b.union('SleeveTools',[d_bore,split]))


def hex_prism(axis,af,depth,origin):
    radius=af/math.sqrt(3)
    if axis=='x':
        pts=[V(origin[0],origin[1]+radius*math.cos(i*math.pi/3),origin[2]+radius*math.sin(i*math.pi/3)) for i in range(6)]
        direction=V(depth,0,0)
    else:
        pts=[V(origin[0]+radius*math.cos(i*math.pi/3),origin[1]+radius*math.sin(i*math.pi/3),origin[2]) for i in range(6)]
        direction=V(0,0,depth)
    return Part.Face(Part.makePolygon(pts+[pts[0]])).extrude(direction)


def cap(b):
    body=b.cylinder('ClampOutside',0,0,0,11,8)
    body.setExpression('Radius','Parameters.AdapterCapOdMm / 2')
    body.setExpression('Height','Parameters.AdapterCapHeightMm')
    bore=b.cylinder('SleeveClearance',0,0,-1,6.1,10)
    bore.setExpression('Radius','Parameters.AdapterCapBoreMm / 2')
    split=b.box('ClampSplit',-CFG['adapter_cap_split_width_mm']/2,-14,-1,CFG['adapter_cap_split_width_mm'],14,10)
    split.setExpression('Length','Parameters.AdapterCapSplitWidthMm')
    split.setExpression('Placement.Base.x','-Parameters.AdapterCapSplitWidthMm / 2')
    tools=[bore,split]
    for x in (-14,6):tools.append(b.box('ClampFlat'+str(x),x,-14,-1,8,10,10))
    axis_y=CFG['adapter_clamp_axis_y_mm']
    hole=b.cylinder('ClampScrewRound',-14,axis_y,4,1.7,28)
    hole.Placement.Rotation=App.Rotation(V(0,0,1),V(1,0,0))
    hole.setExpression('Radius','Parameters.AdapterM3HoleMm / 2')
    # The nut enters from the right outer face and is held against rotation.
    # Keep the full hexagonal bearing pocket; its short horizontal roof bridges.
    nut=b.feature('M3NutPocket',hex_prism('x',CFG['adapter_nut_af_mm'],CFG['adapter_nut_pocket_depth_mm'],(3.2,axis_y,4)))
    tools.extend([hole,nut])
    return b.cut('AdapterClampCap',body,b.union('ClampTools',tools))


def update_layout(name,shape):
    assert shape.isValid() and len(shape.Solids)==1,name
    layout.SHAPES[name]=shape
    entry=next((e for e in layout.REPORT['items'] if e['name']==name),None)
    obj=layout.DOC.getObject(entry['cad_object']) if entry else None
    if obj:obj.Shape=shape
    for group in ('items','detail_items','exploded_items'):
        for entry in layout.REPORT[group]:
            if entry['name']==name:entry['mesh']=layout.mesh(shape,name)


def make_part(stem,cad_name,builder,assembly_offset,rotation,orientation,notes,kind=None,infill=.4,support='None intended; inspect slicer preview'):
    if ASSEMBLY_ONLY or (REBUILD_PARTS and stem not in REBUILD_PARTS):
        metadata=RELEASE['parts'][stem+'_r06']
        path=ROOT/metadata['cad']
        doc=App.openDocument(str(path))
        shape=layout.moved(doc.getObject(metadata['cad_object']).Shape,*assembly_offset)
        add_layout_part(cad_name,shape)
        SOURCE_DOCS.append((path,metadata['cad_object']))
        App.closeDocument(doc.Name)
        return
    b=cad.Builder('Lisa_'+stem+'_r06')
    part=builder(b)
    b.finish(part)
    folder=part_folder(stem)
    (folder/'output/model').mkdir(parents=True,exist_ok=True)
    (folder/'output/render').mkdir(parents=True,exist_ok=True)
    name=stem+'_r06'
    shape=part.Shape.copy()
    shape.rotate(V(),V(1,0,0),rotation)
    bb=shape.BoundBox
    shape.translate(V(-(bb.XMin+bb.XMax)/2,-(bb.YMin+bb.YMax)/2,-bb.ZMin))
    path=folder/(name+'.stl')
    MeshPart.meshFromShape(Shape=shape,LinearDeflection=.045,AngularDeflection=.10,Relative=False).write(str(path))
    model=folder/'output/model'; render=folder/'output/render'
    Import.export([part],str(model/(name+'.step')))
    scale=.72 if max(part.Shape.BoundBox.XLength,part.Shape.BoundBox.YLength,part.Shape.BoundBox.ZLength)>84 else .90
    if stem=='shaft_frame_body':scale=.60
    cad.native_drawing(b,part,name,render,notes,scale,mount_kind=kind,
        view_positions={'Side':(183,38)} if stem=='shaft_frame_body' else None)
    if stem=='motor_holder':
        cad.housing_seat_drawing(b,part,name,render,'motor')
    cad_path=model/(name+'.FCStd')
    b.doc.saveAs(str(cad_path))
    SOURCE_DOCS.append((cad_path,part.Name))
    preview=BUILD/(name+'_local.stl')
    MeshPart.meshFromShape(Shape=part.Shape,LinearDeflection=.045,AngularDeflection=.10,Relative=False).write(str(preview))
    RELEASE['parts'][name]={'stl':str(path.relative_to(ROOT)),'cad':str(cad_path.relative_to(ROOT)),
        'step':str((model/(name+'.step')).relative_to(ROOT)),'cad_object':part.Name,'assembly_object':cad_name,
        'default_quantity':1,'volume_mm3':part.Shape.Volume,
        'bounds_mm':[shape.BoundBox.XLength,shape.BoundBox.YLength,shape.BoundBox.ZLength],
        'print_orientation':orientation,'rotation_x_deg':rotation,'support':support,
        'preview_mesh':str(preview.relative_to(ROOT)),'infill_fraction':infill,
        'drawings':str((render/(name+'_dimensioned.svg')).relative_to(ROOT)),
        'notes':notes,'assembly_offset_mm':list(assembly_offset)}
    if stem=='motor_holder':
        RELEASE['parts'][name]['seat_detail']=str((render/(name+'_seat_detail.svg')).relative_to(ROOT))
        RELEASE['motor_gearbox_relief']=cad.REPORT['motor_gearbox_relief']
    assembly=layout.moved(part.Shape,*assembly_offset)
    add_layout_part(cad_name,assembly)
    print('Exported',name,'mm',RELEASE['parts'][name]['bounds_mm'],flush=True)
    App.closeDocument(b.doc.Name)


def holder(b,kind):
    original=cad.stand(b,kind,sliding=kind=='Encoder')
    h=CFG[kind.lower()+'_stand_height_mm']
    xmin,length=(-20,62) if kind=='Motor' else (-30,60)
    walls=[b.box(kind+'Wall'+str(i),xmin,y,0,length,8,h) for i,y in enumerate((-30,22))]
    for wall in walls:wall.setExpression('Height','Parameters.'+kind+'StandHeightMm')
    return b.union(kind+'HolderStrong',[original]+walls)


def m4_fasteners():
    # Supplier CAD gives diameter-7, 4-mm-high socket heads and 1-mm washers.
    # Represent threads by their nominal outer envelopes for clearance checks.
    for name in list(layout.SHAPES):
        if name.endswith('Screw') and name.startswith(('Frame','LowerBearingMount','UpperBearingMount')):
            old=layout.SHAPES[name]
            is_frame=name.startswith('Frame')
            if is_frame:
                direction=V(0,1 if 'Front' in name else -1,0)
                origin=V(old.BoundBox.Center.x,-48 if 'Front' in name else 48,old.BoundBox.Center.z)
                length=25
            else:
                direction=V(0,0,1 if name.startswith('Lower') else -1)
                origin=V(old.BoundBox.Center.x,old.BoundBox.Center.y,37.5 if name.startswith('Lower') else 98)
                length=16
            screw=Part.makeCylinder(2,length,origin,direction).fuse(Part.makeCylinder(3.5,4,origin-direction*4,direction))
            update_layout(name,screw)
    # Replace encoder clamp envelopes with full screws, washers and captive nuts.
    for name in list(layout.SHAPES):
        if name.startswith('EncoderSlideClamp'):
            entry=next(e for e in layout.REPORT['items'] if e['name']==name)
            layout.DOC.removeObject(entry['cad_object']); del layout.SHAPES[name]
            layout.REPORT['items']=[e for e in layout.REPORT['items'] if e['name']!=name]
    for x in (-24,24):
        for y in (-36,36):
            px=-layout.C+x
            screw=Part.makeCylinder(2,12,V(px,y,0)).fuse(Part.makeCylinder(3.5,4,V(px,y,12)))
            washer=layout.ring(px,y,11,4,2,1)
            nut=hex_prism('z',7,3.2,(px,y,.2)).cut(Part.makeCylinder(2,5,V(px,y,-.5)))
            # The unrotated hex has its flat faces along Y: 7 mm across flats.
            for suffix,shape in (('Screw',screw),('Washer',washer),('Nut',nut)):
                layout.add(f'EncoderClamp{x}{y}'+suffix,shape,'steel','M4 x 12 mm clamp; full 3.2 mm captive-nut engagement')


def main():
    if not App.GuiUp:raise RuntimeError('Run with build_two_belt.sh inside FreeCAD Qt')
    App.ParamGet('User parameter:BaseApp/Preferences/Document').SetInt('CountBackupFiles',0)
    layout.main(export=False)
    remove_layout([n for n in layout.SHAPES if
        (n in ('FrameFrontPanel','FrameRearPanel','FrameFrontFoot','FrameRearFoot','FrameLowerPlate')) or
        (n.startswith(('FrameFront','FrameRear')) and n.endswith(('Screw','Nut')) and 'UpperPlate' not in n)])
    layout.DOC.Label='Lisa r06 two-belt printable fit-test assembly'
    notes_motor=['4 round M3 holes: D 3.8; pattern 33 x 18','Shaft to near column: 9.7 (leveled photo)',
        'Motor boss opening D 16; underside seat depth 4','Net device plate 3; M3 x 6 + 0.6 washer',
        'M4 feet 5 thick; 4 clamps M4 x 12','Broad side walls 8 thick; motor can exposed',
        'Print deck down; seating pocket upward','Local support at four flat washer lands',
        'Near-side recess widened 1.55; clearance 0.60',
        'Pad OD7.70 / edge gap2.00 (user); lip3.85']
    notes_encoder=['M3 mounting slots: 5.5 x 3.8 at R14.5','Use actual 3-hole group; case screws stay intact',
        'Boss opening D 21.5; body seat D39.5 / depth 2','Net device plate 5; M3 x 8 + 0.6 washer',
        'M4 feet 5 thick; 4 clamps M4 x 12','Face 58 above plywood on 6 mm slide base',
        'Print deck down; seating pocket upward','Local support at four flat washer lands']
    make_part('motor_holder','MotorHolderDraft',lambda b:holder(b,'Motor'),(layout.C,0,6),180,
        'Top deck on bed; feet and underside housing seat upward',notes_motor,'MotorHolder',support='Local support at four M4 washer lands in the foot driver chambers')
    make_part('motor_slide_base','MotorSlideBase',cad.slide_base,(layout.C,0,0),180,
        'Countersinks toward bed; captive nut channels upward',[
        '4 M4 slots: 19.6 x 4.6; 15 travel along X','Clamp centers: 50 x 72; X center offset 11',
        'Captive nut channels: 23.6 x 7.4 x 3.4','4 wood holes on 84 x 92 pattern',
        '#6: clearance 4.2; seat D8 / 82 degrees','6 mm base; M4 x 12 + 1 mm washer',
        'Print countersinks down / nut channels up'],'MotorSlideBase')
    make_part('encoder_holder','EncoderHolderDraft',lambda b:holder(b,'Encoder'),(-layout.C,0,6),180,
        'Top deck on bed; feet and underside housing seat upward',notes_encoder,'EncoderHolder',support='Local support at four M4 washer lands in the foot driver chambers')
    make_part('encoder_slide_base','EncoderSlideBaseDraft',encoder_base,(-layout.C,0,0),180,
        'Countersinks toward bed; captive nut channels upward',[
        '4 M4 slots: 19.6 x 4.6; 15 travel along X','Clamp centers: 48 x 72','Nut channels: 23.6 x 7.4 x 3.4',
        '4 wood holes on 62 x 92 pattern','#6: clearance 4.2; seat D8 / 82 degrees',
        '6 mm base; M4 x 12 + 1 mm washer','Fix base to wood before fitting encoder holder'])
    make_part('shaft_frame_body','FrameMainBody',frame_body,(0,0,0),0,
        'Both integrated feet on bed; lower bearing deck horizontal',[
        'Wood holes: D4.2 / seatD8; pattern48 x112',
        '64 x 124 x 92; walls8; twelve gussets4 thick',
        'Lower bearing deck Z43.5 to 49.5; thickness6',
        'Bearing pattern 32 x 32; 4 holes D4.4',
        'Shaft opening D8; bearing underneath deck',
        'Bridge seats Z70; 4.5 thick; locators D4 x 2',
        'Four upper bridge joints D4.4; X +/-24; Z78',
        'Gussets: foot8x16 / deck12x12 / seat8x10',
        'Support beneath lower deck; remove through sides'],
        support='Support beneath the lower bearing deck and overhanging seats; local support in horizontal M4 bores. Remove through the open X sides before fitting bearings.')
    make_part('shaft_upper_bearing_bridge','FrameUpperPlate',upper_bridge,(0,0,70),180,
        'Broad top deck on bed; edge flanges and locator recesses upward',[
        'Bridge 64 x 79.5 x 22; deck thickness6',
        '4 bearing holes D4.4 on 32 x 32 pattern',
        'Shaft clearance D8; bearing above bridge',
        '4 side joints D4.4; X +/-24; local Z8',
        'End clearance 0.25 each side (assumed)',
        'Front socket D4.5 x 2.25; rear slot4.5 x5.5',
        'Locators align; shoulders carry bridge load',
        '4 M4x25 + nuts; bearing uses 4 M4x16',
        'Load both pulleys and belts before closing'],
        support='Local support as needed in four horizontal M4 joining bores; keep locator recesses and bearing faces clean.')
    make_part('gear_adapter_sleeve','PrintedGearAdapter',sleeve,(0,0,109),0,
        'Support flange on bed; keyed sleeve upward',[
        'Keyed sleeve OD11.9; flange OD20 x 2','D shaft clearance 6.3; flat at Y=2.65',
        'Key 2.8 wide; outer X=7.2; height 10','Sleeve height 17 above 2 mm flange',
        'Split 1.2; clamp cap compresses sleeve','Gear bore/key are photo-derived; test fit first',
        'Print flange down; 100% infill; no supports'],infill=1)
    make_part('gear_adapter_clamp_cap','GearAdapterClampCap',cap,(0,0,121),0,
        'Flat annular face on bed; shaft opening vertical',[
        'Cap OD28 / sleeve bore D12.2 / height 8','Split 2.2; M3 round bore D3.4',
        'Captive nut pocket AF5.8 x 2.8 deep','Clamp bolt M3 x 16; standard M3 hex nut',
        'Nut enters from outer right-hand face','Fit cap after sliding gear onto keyed sleeve',
        'Round bores retain material around fasteners',
        '100% infill; remove bore/nut-pocket supports'],infill=1,
        support='Support as needed in horizontal M3 passage and nut-pocket ceiling; remove fully before fastening')
    # M3 clamp nut is recessed by 2.8 mm; 16 mm screw engages it fully.
    axis_y=CFG['adapter_clamp_axis_y_mm']
    update_layout('AdapterClampM3',Part.makeCylinder(1.5,16,V(-6,axis_y,125),V(1,0,0)).fuse(Part.makeCylinder(2.75,3,V(-9,axis_y,125),V(1,0,0))))
    nut=hex_prism('x',5.5,2.4,(3.6,axis_y,125)).cut(Part.makeCylinder(1.5,4,V(3,axis_y,125),V(1,0,0)))
    layout.add('AdapterClampNutM3',nut,'steel','M3 nut retained in the clamp-cap hex pocket')
    m4_fasteners()
    # Actual shaft manufacturer shape is rotated from Y axis to the upright Z axis.
    steel=Part.read(str(SUPPLIER_REFERENCE/'2101-0006-0100.STEP'))
    steel.rotate(V(),V(1,0,0),90); steel.translate(V(0,0,28))
    update_layout('SteelDShaft100mm',steel)
    checks=[]
    pairs=[('PrintedGearAdapter','ExistingGearReference'),('GearAdapterClampCap','ExistingGearReference'),
           ('PrintedGearAdapter','SteelDShaft100mm'),('GearAdapterClampCap','PrintedGearAdapter'),
           ('AdapterClampM3','GearAdapterClampCap'),('AdapterClampM3','PrintedGearAdapter'),
           ('AdapterClampNutM3','GearAdapterClampCap'),('AdapterClampNutM3','PrintedGearAdapter')]
    pairs.extend((name,part) for name in layout.SHAPES if name.startswith('EncoderClamp')
                 for part in ('EncoderHolderDraft','EncoderSlideBaseDraft'))
    for a,b in pairs:
        volume=layout.SHAPES[a].common(layout.SHAPES[b]).Volume
        assert volume<1e-4,(a,b,volume)
        checks.append({'parts':[a,b],'overlap_mm3':round(volume,6)})
    for x in (-14,36):
        for y in (-36,36):
            head=Part.makeCylinder(3.5,4,V(layout.C+x,y,12))
            for fixed in ('MotorHolderDraft','MotorBody'):
                assert head.common(layout.SHAPES[fixed]).Volume<1e-4,('motor socket head',x,y,fixed)
    layout.REPORT['status']='r06 printable fit-test set; physical fit and loaded stiffness unverified'
    layout.REPORT['revision']=CFG['revision']
    layout.REPORT['notes']=[n for n in layout.REPORT['notes'] if 'No deliverable STL' not in n]
    layout.REPORT['notes']=[n.replace('Two 8 mm side panels, two foot rails and two flanged bearing plates bolt together.',
        'Integral frame body and removable upper bearing bridge; four M4 x 25 side joints. Load both pulleys and belts before closing the bridge.') for n in layout.REPORT['notes']]
    layout.REPORT['notes'].extend(['Encoder and motor feet use flat 5 mm M4 washer lands and M4 x 12 clamp screws.',
        'Gear adapter is two printed pieces: keyed sleeve and split cap with retained M3 nut.',
        'Obsolete first-level screw flanges are replaced by twelve local gussets at the feet, lower bearing deck and retained upper bridge seats.',
        'Round fastener bores retain material around frame joints and the gear clamp. Use supports where needed; support-free printing is not a design goal.'])
    for entry in layout.REPORT['items']:
        obj=layout.DOC.getObject(entry['cad_object'])
        obj.DesignStatus=('r06 printable fit-test part' if entry['printed'] else 'Hardware/reference envelope; verify actual fit')
        if entry['name']=='PrintedGearAdapter':
            entry['role']='split keyed sleeve with support flange; removable cap clamps upper neck'
            obj.Label='Gear adapter sleeve — r06'
        if entry['name']=='FrameUpperPlate':
            entry['role']='removable upper bearing bridge; seating shoulders, front pin and rear slot'
            obj.Label='Upper bearing bridge — r06'
    layout.REPORT['anchors_mm']={}  # finalize fills the shared layout anchors
    layout.finalize('two_belt_assembly_r06.FCStd','two_belt_r06_assembly_report.json')
    layout.REPORT['validation']['adapter_and_encoder_clamp_checks']=checks
    layout.REPORT['validation']['motor_socket_head_clearance']='diameter 7 x height 4 verified against holder/body'
    from check_frame_assembly import check
    assembly_checks=check(layout)
    layout.REPORT['validation']['frame_assembly']=assembly_checks
    (SCRIPTS/'frame_assembly_checks.json').write_text(json.dumps(assembly_checks,indent=2)+'\n')
    layout.REPORT['validation']['limitations']='Physical fit, pulley axial dimensions/grip, gear bore/key fit and loaded stiffness require checks. No load rating or vibration performance is established.'
    (SCRIPTS/'two_belt_r06_assembly_report.json').write_text(json.dumps(layout.REPORT,indent=2)+'\n')
    RELEASE['assembly_report']='scripts/two_belt_r06_assembly_report.json'
    RELEASE['assembly_cad']=str((ASSEMBLY_MODEL/'two_belt_assembly_r06.FCStd').relative_to(ROOT))
    if REBUILD_PARTS:
        updated={d['svg']:d for d in cad.REPORT['drawings']}
        RELEASE['drawings']=[updated.pop(d['svg'],d) for d in RELEASE['drawings']]+list(updated.values())
    elif not ASSEMBLY_ONLY:
        RELEASE['drawings']=cad.REPORT['drawings']
    for path,name in SOURCE_DOCS:
        doc=App.openDocument(str(path))
        assert doc.getObject(name).Shape.isValid(),path
        App.closeDocument(doc.Name)
    RELEASE['native_documents_reopened']=True
    (SCRIPTS/'two_belt_r06_report.json').write_text(json.dumps(RELEASE,indent=2)+'\n')
    Path(sys.argv[1],'freecad_complete').write_text(CFG['revision']+'\n')
    print(f'r06: {len(RELEASE["parts"])} printable parts, CAD, STEP, TechDraw and assembly geometry complete.')


if __name__=='__main__':main()
