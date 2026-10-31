"""FreeCAD source for the discussed two-belt layout; visual draft, not print release.

Run with scripts/freecad-python.sh. Existing r05 printable files are untouched.
Purchased bearings/collars use supplier STEP. Pulley axial dimensions and the
gear adapter remain provisional; this draft checks packaging and assembly access.
"""
import json
import math
from pathlib import Path

import FreeCAD as App
import MeshPart
import Part
from project_paths import ASSEMBLY_MODEL, REFERENCE, SUPPLIER_REFERENCE

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
MODEL = ASSEMBLY_MODEL
REF = SUPPLIER_REFERENCE
MESHES = ROOT / '.build/two_belt_draft'
MESHES.mkdir(parents=True, exist_ok=True)
V = App.Vector
PITCH = 2.0
R, r = 80 * PITCH / (2 * math.pi), 20 * PITCH / (2 * math.pi)


def belt_length(c):
    delta = R - r
    return 2 * math.sqrt(c*c-delta*delta) + math.pi*(R+r) + 2*delta*math.asin(delta/c)


lo, hi = R+r, 250.0
for _ in range(80):
    mid = (lo+hi)/2
    if belt_length(mid) < 250:
        lo = mid
    else:
        hi = mid
C = (lo+hi)/2
DOC = App.newDocument('LisaTwoBeltDraft')
COLORS = {'frame': (.07,.42,.61), 'motor': (.08,.48,.60), 'encoder': (.13,.55,.40),
          'steel': (.50,.55,.60), 'aluminum': (.72,.75,.80), 'dark': (.10,.12,.15),
          'adapter': (.95,.37,.06), 'belt': (.025,.032,.04)}
SHAPES = {}
PRINTED = []
REPORT = {'status': 'Visual layout draft; not a printable release', 'units': 'mm',
          'belt': {'type': 'GT2 / 2GT', 'pitch_mm': 2, 'width_mm': 6,
                   'pitch_length_mm': 250, 'quantity': 2, 'teeth': 125},
          'shaft_spacing_mm': C, 'belt_centers_mm': [61,77],
          'shaft_bottom_mm': 28, 'shaft_length_mm': 100,
          'bearing_centers_mm': [40,95.5], 'bearing_spacing_mm': 55.5,
          'speed_ratios_to_motor': {'motor': 1, 'gear_shaft': .25, 'encoder': 1},
          'encoder_revolutions_per_gear_revolution': 4,
          'notes': ['Motor 20T drives lower shaft 80T; upper shaft 80T drives encoder 20T.',
                    'Both 80T pulleys lie between two separate purchased bearings.',
                    'Lower bearing is underneath its plate; upper bearing is on top of its plate.',
                    'Collars remain outside both plates. Lower pinch bolt is reached through the front opening.',
                    'Two 8 mm side panels, two foot rails and two flanged bearing plates bolt together.',
                    'Motor and encoder have independent 15 mm tension adjustment.',
                    'Motor face stays at 42 mm. Encoder face is raised to 58 mm for the upper belt.',
                    'Pulley hub/flange heights, gear dimensions/key fit, loads and mounting screw engagement need final checks.',
                    'No TPU pads in this draft. Plywood footprint shown is illustrative; thickness is 10 mm.',
                    'No deliverable STL is generated. Existing r05 files remain the earlier fit-test set.'],
          'items': [], 'detail_items': [], 'exploded_items': [], 'validation': {}}


def mesh(shape, name):
    path = MESHES / (name + '.stl')
    MeshPart.meshFromShape(Shape=shape, LinearDeflection=.10,
                          AngularDeflection=.18, Relative=False).write(str(path))
    return str(path.relative_to(ROOT))


def add(name, shape, color, role, printed=False, exploded=(0,0,0)):
    assert shape.isValid() and shape.Volume > 0, name
    if printed:
        assert len(shape.Solids) == 1, (name, len(shape.Solids))
        PRINTED.append(name)
    obj = DOC.addObject('Part::Feature', name)
    obj.Label = name + ' — ' + role
    obj.Shape = shape
    obj.addProperty('App::PropertyString','DesignStatus','Provenance')
    obj.DesignStatus = 'Draft: ' + role
    if App.GuiUp:
        obj.ViewObject.ShapeColor=COLORS[color]
    entry = {'name': name, 'cad_object':obj.Name,'mesh': mesh(shape,name), 'color': COLORS[color],
             'role': role, 'printed': printed, 'exploded_offset_mm': list(exploded)}
    REPORT['items'].append(entry)
    SHAPES[name] = shape
    if name.startswith(('Frame','LowerBearing','UpperBearing','LowerCollar','UpperCollar')) or name in (
            'SteelDShaft100mm','LowerShaft80T','UpperShaft80T','PrintedGearAdapter','ExistingGearReference','AdapterClampM3'):
        # Front frame panel/rail are removed for a clear shaft assembly view.
        if name not in ('FrameFrontPanel','FrameFrontFoot'):
            REPORT['detail_items'].append(entry)
    return entry


def moved(shape, x=0, y=0, z=0):
    shape = shape.copy()
    shape.translate(V(x,y,z))
    return shape


def ring(x,y,z,outer,inner,height):
    return Part.makeCylinder(outer,height,V(x,y,z)).cut(
        Part.makeCylinder(inner,height+2,V(x,y,z-1)))


def slot_x(x,y,z,length,width,height):
    half=(length-width)/2
    return Part.makeBox(2*half,width,height,V(x-half,y-width/2,z)).multiFuse([
        Part.makeCylinder(width/2,height,V(x-half,y,z)),
        Part.makeCylinder(width/2,height,V(x+half,y,z))]).removeSplitter()


def y_hole(x,y,z,length,roof=0):
    radius=2.2
    hole=Part.makeCylinder(radius,length,V(x,y,z),V(0,1,0))
    if roof:
        a=radius/math.sqrt(2)
        pts=[V(x-a,y,z+roof*a),V(x+a,y,z+roof*a),V(x,y,z+roof*radius*math.sqrt(2))]
        hole=hole.fuse(Part.Face(Part.makePolygon(pts+[pts[0]])).extrude(V(0,length,0)))
    return hole


def wood_hole(x,y):
    depth=(4-2.1)/math.tan(math.radians(41))
    return Part.makeCylinder(2.1,8,V(x,y,-1)).fuse(
        Part.makeCone(2.1,4,depth,V(x,y,6-depth)))


def wood_screw(name,x,y):
    depth=(4-2.1)/math.tan(math.radians(41))
    shape=Part.makeCylinder(1.75,12.7-depth,V(x,y,6-12.7)).fuse(
        Part.makeCone(2.1,4,depth,V(x,y,6-depth)))
    add(name,shape,'steel','#6 x 1/2 inch flat-head plywood screw envelope')


def bolt_nut(name, origin, axis, length, nut_distance):
    axis=V(*axis)
    origin=V(*origin)
    screw=Part.makeCylinder(2,length,origin,axis).fuse(
        Part.makeCylinder(3.5,3,origin-axis*3,axis))
    # Hex nut initially points along +Z, then is aligned to the screw axis.
    pts=[V(4*math.cos(i*math.pi/3),4*math.sin(i*math.pi/3),0) for i in range(6)]
    nut=Part.Face(Part.makePolygon(pts+[pts[0]])).extrude(V(0,0,3.2)).cut(
        Part.makeCylinder(2,5,V(0,0,-1)))
    transform=App.Placement(origin+axis*nut_distance,App.Rotation(V(0,0,1),axis))
    nut.Placement=transform
    add(name+'Screw',screw,'dark',f'M4 x {length} mm reference fastener')
    add(name+'Nut',nut,'steel','M4 hex nut reference')


def supplier(path, axis_center, base_y, target_z, spin=0, flip_center=None):
    parts=[]
    for solid in Part.read(str(path)).Solids:
        shape=solid.copy()
        shape.translate(V(-axis_center[0],-base_y,-axis_center[1]))
        shape.rotate(V(),V(1,0,0),90)
        shape.translate(V(0,0,target_z))
        if flip_center is not None:
            shape.rotate(V(0,0,flip_center),V(1,0,0),180)
        if spin:
            shape.rotate(V(),V(0,0,1),spin)
        parts.append(shape)
    return parts


def pulley(name,cx,teeth,hub_bottom):
    # Existing photos/reference CAD supply provisional axial dimensions.
    # Tooth-form details are deliberately omitted from this packaging model.
    radius=teeth*PITCH/(2*math.pi)-.25
    flange=27.5 if teeth==80 else 8.3
    hub=13 if teeth==80 else 6
    shape=Part.makeCylinder(hub,8,V(cx,0,hub_bottom)).multiFuse([
        Part.makeCylinder(flange,1,V(cx,0,hub_bottom+7)),
        Part.makeCylinder(radius,6,V(cx,0,hub_bottom+8)),
        Part.makeCylinder(flange,1,V(cx,0,hub_bottom+14))]).cut(
        Part.makeCylinder(3.05,20,V(cx,0,hub_bottom-1))).removeSplitter()
    add(name,shape,'aluminum',f'{teeth}T existing pulley; axial dimensions provisional')


def belt(name,small_x,z):
    sign=1 if small_x>0 else -1
    nx=(R-r)/C
    ny=math.sqrt(1-nx*nx)
    a,b=V(sign*nx*R,ny*R,0),V(sign*nx*R,-ny*R,0)
    c,d=V(small_x+sign*nx*r,-ny*r,0),V(small_x+sign*nx*r,ny*r,0)
    big=Part.Arc(a,V(-sign*R,0,0),b).toShape()
    small=Part.Arc(c,V(small_x+sign*r,0,0),d).toShape()
    wire=Part.Wire([big,Part.makeLine(b,c),small,Part.makeLine(d,a)])
    face=Part.Face(wire)
    # Place the pitch line within a thin reference band; no decorative teeth.
    outer=face.makeOffset2D(.70)
    inner=face.makeOffset2D(-.70)
    band=outer.cut(inner).extrude(V(0,0,6))
    band.translate(V(0,0,z-3))
    add(name,band,'belt','250 mm pitch length x 6 mm GT2 reference belt')


def main(export=True):
    original=App.openDocument(str(ROOT/'motor holder/output/model/reference/motor_encoder_assembly_reference.FCStd'))
    motor=original.getObject('MotorHolder').Shape.copy()
    # Fill between the existing short posts with two walls, outside the motor body.
    motor=motor.multiFuse([Part.makeBox(62,8,32,V(-20,y,6)) for y in (-30,22)]).removeSplitter()
    add('MotorHolderDraft',moved(motor,C),'motor','r05 fitted top/feet; two broad side walls',True)
    add('MotorSlideBase',moved(original.getObject('MotorSlideBase').Shape,C),
        'motor','existing r05 15 mm motor adjustment',True)
    add('MotorBody',moved(original.getObject('MotorApproximateEnvelope').Shape,C),
        'steel','existing motor reference envelope; can remains exposed')
    for obj in original.Objects:
        if obj.Name.startswith(('MotorAttachmentM3_','MotorClampM4_','MotorWoodScrew')):
            add(obj.Name,moved(obj.Shape,C),'steel','existing r05 motor fastening reference')

    # Encoder is beside the gear shaft, raised 16 mm overall for belt plane 77.
    # The 6 mm slide base contributes 6 mm; the stand grows by 10 mm.
    encoder=original.getObject('EncoderHolder').Shape.copy()
    encoder.translate(V(72.5,0,0))
    feet=encoder.common(Part.makeBox(80,100,12,V(-40,-50,0)))
    upper=encoder.common(Part.makeBox(80,100,60,V(-40,-50,12)))
    upper.translate(V(0,0,10))
    connectors=[Part.makeBox(12,12,10,V(x,y,12)) for x in (-30,18) for y in (-30,18)]
    stand=feet.multiFuse([upper]+connectors).removeSplitter()
    stand=stand.multiFuse([Part.makeBox(60,8,50,V(-30,y,0)) for y in (-30,22)]).removeSplitter()
    stand.translate(V(-C,0,6))
    for x in (-24,24):
        for y in (-36,36):
            stand=stand.cut(Part.makeCylinder(2.2,18,V(-C+x,y,0)))
    add('EncoderHolderDraft',stand.removeSplitter(),'encoder',
        'r05 top interface; face height 58 mm; broad walls and movable feet',True)
    enc_body=moved(original.getObject('EncoderApproximateEnvelope').Shape,72.5-C,0,16)
    add('EncoderBody',enc_body,'dark','existing encoder reference; upper belt plane')
    for obj in original.Objects:
        if obj.Name.startswith('EncoderAttachmentM3_'):
            add(obj.Name,moved(obj.Shape,72.5-C,0,16),'steel','existing M3 encoder screw-head envelope')
    enc_base=Part.makeBox(76,104,6,V(-C-38,-52,0))
    for x in (-24,24):
        for y in (-36,36):
            enc_base=enc_base.cut(slot_x(-C+x,y,-1,19.6,4.6,8))
            enc_base=enc_base.cut(Part.makeBox(23.6,7.4,3.4,V(-C+x-11.8,y-3.7,0)))
            washer=ring(-C+x,y,12,4.5,2.3,1)
            head=Part.makeCylinder(3.5,2.5,V(-C+x,y,13))
            add('EncoderSlideClamp'+str(x)+str(y),washer.fuse(head),'steel',
                'M4 sliding clamp head/washer envelope; screw engagement to finalize')
    for i,(x,y) in enumerate([(x,y) for x in (-31,31) for y in (-46,46)]):
        enc_base=enc_base.cut(wood_hole(-C+x,y))
        wood_screw('EncoderBaseWood'+str(i),-C+x,y)
    add('EncoderSlideBaseDraft',enc_base.removeSplitter(),'encoder',
        '76 x 104 x 6 mm base; independent 15 mm encoder adjustment',True)

    # A dismantlable frame replaces the fused pedestal and large triangular ribs.
    for label,y,side in (('Front',-48,-1),('Rear',40,1)):
        panel=Part.makeBox(64,8,86,V(-32,y,6))
        for z,height in ((10,31),(52,31)):
            panel=panel.cut(Part.makeBox(36,10,height,V(-18,y-1,z)))
        for x in (-24,24):
            for z in (14,57.5,78):
                panel=panel.cut(y_hole(x,y-1,z,10))
        add('Frame'+label+'Panel',panel.removeSplitter(),'frame',
            '8 mm flat-printing X-Z frame panel',True,(0,side*38,0))
        foot_y=-62 if side<0 else 32
        foot=Part.makeBox(64,30,6,V(-32,foot_y,0))
        flange_y=-40 if side<0 else 32
        foot=foot.fuse(Part.makeBox(64,8,16,V(-32,flange_y,6)))
        for x in (-24,24):
            foot=foot.cut(y_hole(x,flange_y-1,14,10,1))
            screw_y=side*56
            foot=foot.cut(wood_hole(x,screw_y))
            wood_screw('Frame'+label+'Wood'+str(x),x,screw_y)
        add('Frame'+label+'Foot',foot.removeSplitter(),'frame',
            'flat base and upright flange; exposed #6 countersinks',True,(0,side*38,0))
        for z,kind in ((14,'Foot'),(57.5,'LowerPlate'),(78,'UpperPlate')):
            for x in (-24,24):
                origin=(x,-48,z) if side<0 else (x,48,z)
                axis=(0,1,0) if side<0 else (0,-1,0)
                bolt_nut('Frame'+label+kind+str(x),origin,axis,25,16)

    for label,z,flange_z,roof in (('Lower',43.5,49.5,1),('Upper',86,70,-1)):
        plate=Part.makeBox(64,80,6,V(-32,-40,z)).cut(
            Part.makeCylinder(4,8,V(0,0,z-1)))
        for x in (-16,16):
            for y in (-16,16):
                plate=plate.cut(Part.makeCylinder(2.2,8,V(x,y,z-1)))
        plate=plate.multiFuse([Part.makeBox(64,8,16,V(-32,y,flange_z)) for y in (-40,32)])
        join_z=57.5 if label=='Lower' else 78
        for x in (-24,24):
            for y in (-40,32):
                plate=plate.cut(y_hole(x,y-1,join_z,10,roof))
        add('Frame'+label+'Plate',plate.removeSplitter(),'frame',
            'separate 6 mm flanged bearing plate; only shaft passes through Ø8 opening',True,
            (0,0,-10 if label=='Lower' else 26))

    # Bearing housings sit on the outside faces of both plates, leaving collars exposed.
    for label,bottom,flip in (('Lower',37.5,40),('Upper',93,None)):
        parts=supplier(REF/'1603-0032-0006 assembly.STEP',
                       (-47.01395694911526,12.441869203883014),16.20286100524982,
                       bottom,flip_center=flip)
        add(label+'BearingHousing',parts[0],'aluminum','manufacturer 1603-0032-0006 CAD')
        for i in (1,2):
            add(label+'BearingSeal'+str(i),parts[i],'dark','manufacturer bearing seal')
        add(label+'Bearing',parts[3],'steel','manufacturer 6 mm ID bearing')
        for x in (-16,16):
            for y in (-16,16):
                if label=='Lower':
                    bolt_nut(label+'BearingMount'+str(x)+str(y),(x,y,37.5),(0,0,1),16,12)
                else:
                    bolt_nut(label+'BearingMount'+str(x)+str(y),(x,y,98),(0,0,-1),16,12)

    shaft=Part.makeCylinder(3,100,V(0,0,28)).cut(
        Part.makeBox(8,4,102,V(-4,2.5,27)))
    add('SteelDShaft100mm',shaft,'steel','6 mm D shaft, 100 mm total length')
    for label,target,flip in (('Lower',28,None),('Upper',28,67.75)):
        parts=supplier(REF/'2910-0919-0006 assembly.STEP',
                       (-31.056451395386375,16.726808067025576),-5.02042300416133,
                       target,spin=90,flip_center=flip)
        for i,shape in enumerate(parts):
            add(label+'Collar'+('PinchBolt' if i else ''),shape,'aluminum' if i==0 else 'dark',
                'manufacturer collar; raised face contacts bearing inner ring; key faces front')

    pulley('Motor20T',C,20,50)
    pulley('LowerShaft80T',0,80,50)
    pulley('UpperShaft80T',0,80,66)
    pulley('Encoder20T',-C,20,66)
    belt('MotorReductionBelt',C,61)
    belt('EncoderBoostBelt',-C,77)

    # Existing photo-estimated gear envelope; orange adapter remains provisional.
    gear_source=Part.Shape()
    gear_source.read(str(REFERENCE/'gear_reference.brep'))
    gear=moved(gear_source,72.5,0,-29)
    add('ExistingGearReference',gear,'steel','existing gear envelope; bevel teeth schematic')
    sleeve=Part.makeCylinder(5.95,10,V(0,0,111)).fuse(
        Part.makeBox(1.6,2.8,10,V(5.6,-1.4,111)))
    adapter=sleeve.multiFuse([Part.makeCylinder(10,2,V(0,0,109)),
                             Part.makeCylinder(10,6,V(0,0,121))])
    d_bore=Part.makeCylinder(3.1,22,V(0,0,108)).cut(
        Part.makeBox(8,4,24,V(-4,2.6,107)))
    adapter=adapter.cut(d_bore).cut(Part.makeBox(.8,12,22,V(-.4,-12,108)))
    for x in (-12,6):
        adapter=adapter.cut(Part.makeBox(6,7,7,V(x,-11,121)))
    adapter=adapter.cut(Part.makeCylinder(1.6,24,V(-12,-7,124),V(1,0,0))).removeSplitter()
    add('PrintedGearAdapter',adapter,'adapter','provisional keyed adapter with accessible M3 clamp',True)
    screw=Part.makeCylinder(1.5,16,V(-6,-7,124),V(1,0,0)).fuse(
        Part.makeCylinder(2.75,3,V(-9,-7,124),V(1,0,0)))
    add('AdapterClampM3',screw,'steel','illustrative M3 x 16 mm clamp screw; nut/retention to finalize')

    if not export:
        return
    finalize()


def finalize(cad_name='two_belt_assembly_draft.FCStd',report_name='two_belt_draft_report.json'):
    validate()
    # The shaft detail is an actual CAD section, with the front half removed.
    section_box=Part.makeBox(180,100,170,V(-90,-100,-10))
    section_items=[]
    for entry in REPORT['detail_items']:
        shape=SHAPES[entry['name']]
        if entry['name']=='SteelDShaft100mm':
            section_items.append(entry)
            continue
        cut=shape.cut(section_box)
        if cut.Volume>1e-4:
            section_items.append({**entry,'mesh':mesh(cut,entry['name']+'_section')})
    REPORT['detail_items']=section_items
    # Only the upper bearing bridge is removed to load the shaft and pulleys.
    for entry in REPORT['items']:
        name=entry['name']
        if name=='FrameUpperPlate' or name.startswith('UpperBearing'):
            offset=(0,0,26)
        elif name.startswith(('FrameFrontUpperPlate','FrameRearUpperPlate')):
            offset=(0,-16 if 'Front' in name else 16,26)
        elif name.startswith(('Frame','LowerBearing')):
            offset=(0,0,0)
        else:
            continue
        REPORT['exploded_items'].append({**entry,'exploded_offset_mm':list(offset)})
    REPORT['anchors_mm']={'gear':[0,-16,113], 'adapter':[0,-7,124],
        'motor':[C+47,-5,29.5], 'encoder':[-C,-15,41],
        'motor20':[C,-6,61], 'lower80':[17,-14,61], 'upper80':[-17,-14,77],
        'encoder20':[-C,-6,77], 'upper_collar':[0,-9,102.5], 'lower_collar':[0,-9,32.5],
        'upper_bearing':[0,-17,95], 'lower_bearing':[0,-17,40],
        'upper_plate':[27,-28,89], 'lower_plate':[27,-28,46.5],
        'panel':[28,-45,70], 'foot':[24,-56,6],
        'motor_slide':[C+36,-36,12], 'encoder_slide':[-C-24,-36,13],
        'lower_belt':[C/2,-21,61], 'upper_belt':[-C/2,-21,77],
        'shaft_top':[0,0,128], 'shaft_bottom':[0,0,28],
        'axis_motor':[C,0,0], 'axis_gear':[0,0,0], 'axis_encoder':[-C,0,0]}
    REPORT['printed_parts']=[{'name':n,'volume_mm3':SHAPES[n].Volume} for n in PRINTED]
    REPORT['hardware']=[
        ['20T pulleys',2,'6 mm shaft bore; existing pulley sets'],
        ['80T pulleys',2,'Both on the separate gear shaft'],
        ['GT2 belts',2,'250 mm pitch length x 6 mm width, 2 mm pitch; both on hand; user report'],
        ['Steel shaft',1,'6 mm D shaft, 100 mm illustrative length'],
        ['Bearing blocks',2,'goBILDA 1603-0032-0006; single bearing in each block'],
        ['Shaft collars',2,'goBILDA 2910-0919-0006; pinch screws included'],
        ['Bearing mounting screws',8,'M4 x 16 mm; 6 mm housing + 6 mm plate + nut'],
        ['Bearing mounting nuts',8,'M4 hex nuts'],
        ['Frame joining screws',4,'M4 x 25 mm; removable upper bearing bridge'],
        ['Frame joining nuts',4,'M4 hex nuts'],
        ['Plywood screws',12,'#6 x 1/2 inch flat head; no TPU pads in draft']]
    DOC.recompute()
    path=MODEL/cad_name
    DOC.saveAs(str(path))
    REPORT['cad_source']=str(path.relative_to(ROOT))
    (SCRIPTS/report_name).write_text(json.dumps(REPORT,indent=2)+'\n')
    print('Saved two-belt CAD '+('draft; existing printable files unchanged.' if 'draft' in cad_name else 'assembly: '+cad_name))
    print(f'Shaft spacing {C:.4f} mm, bearing span 55.5 mm, belt centres 61 / 77 mm.')


def validate():
    # Packaging checks, not a load/stiffness or manufacturing certification.
    checks=[]
    frame=[n for n in PRINTED if n.startswith('Frame')]
    core=['SteelDShaft100mm','LowerShaft80T','UpperShaft80T','MotorReductionBelt','EncoderBoostBelt',
          'LowerBearingHousing','UpperBearingHousing','LowerCollar','UpperCollar',
          'LowerCollarPinchBolt','UpperCollarPinchBolt','ExistingGearReference']
    core += [n for n in SHAPES if n.startswith(('LowerBearingMount','UpperBearingMount','Frame')) and n not in PRINTED]
    for name in core:
        for fixed in PRINTED:
            if fixed=='PrintedGearAdapter':
                continue
            v=SHAPES[name].common(SHAPES[fixed]).Volume
            assert v<1e-4,(name,fixed,v)
            checks.append({'parts':[name,fixed],'overlap_mm3':round(v,6)})
    for i,a in enumerate(PRINTED):
        for b in PRINTED[i+1:]:
            v=SHAPES[a].common(SHAPES[b]).Volume
            assert v<1e-4,(a,b,v)
    # Check each rotating pulley against all stationary frame hardware.
    fasteners=[n for n in SHAPES if n.startswith(('LowerBearingMount','UpperBearingMount','Frame')) and n not in PRINTED]
    for pulley_name in ('LowerShaft80T','UpperShaft80T'):
        for fixed in fasteners:
            v=SHAPES[pulley_name].common(SHAPES[fixed]).Volume
            assert v<1e-4,(pulley_name,fixed,v)
    contacts=[]
    for label in ('Lower','Upper'):
        collar,bearing=SHAPES[label+'Collar'],SHAPES[label+'Bearing']
        gap=collar.distToShape(bearing)[0]
        v=collar.common(bearing).Volume
        assert gap<1e-5 and v<1e-4,(label,gap,v)
        contacts.append({'collar':label,'gap_mm':gap,'overlap_mm3':v})
    for z in (28,98.5):
        sweep=Part.makeCylinder(11.281,9,V(0,0,z))
        for fixed in frame+fasteners+['LowerBearingHousing','UpperBearingHousing']:
            assert sweep.common(SHAPES[fixed]).Volume<1e-4,('collar sweep',z,fixed)
    # Front-facing key path, including passage through the lower panel window.
    key_checks=[]
    for label in ('Lower','Upper'):
        bolt=SHAPES[label+'CollarPinchBolt'].BoundBox
        key=Part.makeCylinder(1.9,bolt.YMin+80,V(bolt.Center.x,-80,bolt.Center.z),V(0,1,0))
        for fixed in frame:
            v=key.common(SHAPES[fixed]).Volume
            assert v<1e-4,('collar key access',label,fixed,v)
            key_checks.append({'collar':label,'fixed_part':fixed,'overlap_mm3':v})
    # Device slide positions remain clear of the fixed central frame at both ends.
    for kind,names in (('Motor',['MotorHolderDraft','MotorBody','Motor20T']),
                       ('Encoder',['EncoderHolderDraft','EncoderBody','Encoder20T'])):
        for dx in (-7.5,0,7.5):
            for name in names:
                shape=moved(SHAPES[name],dx)
                for fixed in frame:
                    v=shape.common(SHAPES[fixed]).Volume
                    assert v<1e-4,(kind,'travel',dx,name,fixed,v)
    # Export/visualization meshes come from valid solids; no load rating is implied.
    REPORT['validation']={'all_shapes_valid':True,'printed_parts_single_solid':len(PRINTED),
        'hardware_support_checks':checks,'collar_contacts':contacts,'collar_key_access':key_checks,
        'pulley_stationary_fastener_checks':'passed','device_travel_frame_checks':'passed',
        'belt_pitch_length_mm':belt_length(C), 'both_belts_between_bearings':40<61<77<95.5,
        'limitations':'No stiffness/load/vibration verification. Pulley axial dimensions, shaft grip, adapter key fit and final mount/clamp screw engagement remain provisional.'}


if __name__=='__main__':
    main()
