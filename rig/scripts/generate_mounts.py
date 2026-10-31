"""Create photo-fit mounts, native CSG documents, TechDraw pages, and an assembly.

Run inside FreeCAD's GUI/event loop through build_mounts.sh. All coordinates mm.
The model keeps primitive/boolean history; major dimensions use expressions
linked to the Parameters object. Re-run from mount_parameters.json for a full
regeneration, including gussets and the supplemental reference assembly.
"""
import json
import math
import sys
import time
from pathlib import Path

import FreeCAD as App
import FreeCADGui as Gui
import Import
import MeshPart
import Part
from PySide6 import QtWidgets
import TechDrawGui

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG = json.loads((SCRIPT_DIR / 'mount_parameters.json').read_text())
REV = CONFIG['revision'].split('_')[0]
V = App.Vector
REPORT = {'revision': CONFIG['revision'], 'parts': {}, 'drawings': [],
          'parameters': CONFIG, 'assembly': []}


class Builder:
    def __init__(self, name):
        self.doc = App.newDocument(name)
        self.param = self.doc.addObject('App::FeaturePython', 'Parameters')
        self.param.Label = 'Dimensions (mm; photo-derived fit inputs)'
        for key, value in CONFIG.items():
            if isinstance(value, (int, float)):
                alias = ''.join(word.title() for word in key.split('_'))
                self.param.addProperty('App::PropertyFloat', alias, 'Inputs')
                setattr(self.param, alias, value)
        self.param.addProperty('App::PropertyString', 'FitStatus', 'Inputs')
        self.param.FitStatus = CONFIG['fit_status']

    def obj(self, typ, name):
        return self.doc.addObject(typ, name)

    def box(self, name, x, y, z, dx, dy, dz):
        obj = self.obj('Part::Box', name)
        obj.Length, obj.Width, obj.Height = dx, dy, dz
        obj.Placement.Base = V(x, y, z)
        return obj

    def cylinder(self, name, x, y, z, radius, height):
        obj = self.obj('Part::Cylinder', name)
        obj.Radius, obj.Height = radius, height
        obj.Placement.Base = V(x, y, z)
        return obj

    def feature(self, name, shape):
        obj = self.obj('Part::Feature', name)
        obj.Shape = shape
        return obj

    def union(self, name, objects):
        if len(objects) == 1:
            return objects[0]
        obj = self.obj('Part::MultiFuse', name)
        obj.Shapes = objects
        obj.Refine = True
        return obj

    def cut(self, name, base, tool):
        obj = self.obj('Part::Cut', name)
        obj.Base, obj.Tool = base, tool
        obj.Refine = True
        return obj

    def capsule(self, name, x, y, z, length, width, height, angle=0):
        # Vertical capsule slot, represented by editable primitives and a fuse.
        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)
        offset = (length - width) / 2
        objects = [self.cylinder(name + 'EndA', x - offset*c, y - offset*s,
                                 z, width/2, height),
                   self.cylinder(name + 'EndB', x + offset*c, y + offset*s,
                                 z, width/2, height)]
        bridge = self.box(name+'Web', -offset, -width/2, 0,
                          length-width, width, height)
        rotation=App.Rotation(V(0,0,1),angle)
        bridge.Placement = App.Placement(
            V(x,y,z)+rotation.multVec(V(-offset,-width/2,0)),rotation)
        objects.append(bridge)
        slot=self.union(name, objects)
        slot.addProperty('App::PropertyFloat','ExpectedVolume','Validation')
        slot.ExpectedVolume=(width*(length-width)+math.pi*width*width/4)*height
        return slot

    def shift_expression(self, object_, axis, expr):
        # Expressions are applied to the primitive placements within a slot.
        for primitive in object_.Shapes:
            current = getattr(primitive.Placement.Base, axis)
            primitive.setExpression('Placement.Base.'+axis,
                                    f'{expr} + ({current:.9f})')

    def triangular_ramp(self, name, x, width, points_yz):
        pts = [V(x,y,z) for y,z in points_yz]
        wire = Part.makePolygon(pts+[pts[0]])
        return self.feature(name, Part.Face(wire).extrude(V(width,0,0)))

    def finish(self, part, visible=True):
        self.doc.recompute()
        for obj in self.doc.Objects:
            if hasattr(obj,'ExpectedVolume'):
                assert math.isclose(obj.Shape.Volume,obj.ExpectedVolume,rel_tol=1e-7),obj.Name
        assert part.Shape.isValid(), part.Name
        assert len(part.Shape.Solids) == 1, (part.Name,len(part.Shape.Solids))
        for obj in self.doc.Objects:
            if hasattr(obj, 'Shape'):
                obj.Visibility = False
        part.Visibility = visible
        part.ViewObject.ShapeColor = (0.14,0.53,0.64)
        view=Gui.activeDocument().activeView()
        if hasattr(view,'viewAxonometric'):
            view.viewAxonometric()
            view.fitAll()


def countersunk_wood_hole(b, name, x, y, top, tool_height):
    hole = b.cylinder(name+'Through',x,y,-1,2.1,tool_height+2)
    # Provisional #6 clearance 4.2, head seat 8, 82 degree included angle.
    depth = (4.0-2.1)/math.tan(math.radians(41))
    cone = b.obj('Part::Cone',name+'HeadSeat')
    cone.Radius1, cone.Radius2, cone.Height = 2.1,4.0,depth
    cone.Placement.Base = V(x,y,top-depth)
    return b.union(name,[hole,cone])


def stand(b, kind, sliding=False):
    h,t = CONFIG[kind.lower()+'_stand_height_mm'],CONFIG['deck_thickness_mm']
    height_property='Parameters.'+kind+'StandHeightMm'
    is_motor = kind == 'Motor'
    xmin,xmax = (-20,42) if is_motor else (-30,30)
    foot_t = 5 if is_motor else CONFIG['encoder_foot_thickness_mm']
    xs = (xmin,xmax-12)
    solids,tools = [],[]
    deck = b.box(kind+'Deck',xmin,-30,h,xmax-xmin,60,t)
    deck.setExpression('Placement.Base.z',height_property)
    deck.setExpression('Height','Parameters.DeckThicknessMm')
    solids.append(deck)
    for side_x,x in enumerate(xs):
        for side_y,sgn in enumerate((-1,1)):
            name=f'{kind}Post{side_x}{side_y}'
            y=-30 if sgn<0 else 18
            post=b.box(name,x,y,0,12,12,h)
            post.setExpression('Height',height_property)
            solids.append(post)
            fy=-42 if sgn<0 else 18
            solids.append(b.box(name+'Foot',x,fy,0,12,24,foot_t))
            # Flared gussets brace the post-to-foot connection.
            ramp=b.triangular_ramp(name+'FootRamp',x,12,
                    [(sgn*30,foot_t),(sgn*42,foot_t),(sgn*30,foot_t+12)])
            solids.append(ramp)
            # Short braces along the outer edge stiffen the deck/post junction.
            # Their upper face is attached to the first printed layers.
            dx = 1 if side_x==0 else -1
            edge = x+12 if side_x==0 else x
            pts=[V(edge,sgn*30,h-10),V(edge,sgn*30,h),
                 V(edge+dx*10,sgn*30,h)]
            brace=b.feature(name+'DeckBrace',Part.Face(Part.makePolygon(pts+[pts[0]]))
                            .extrude(V(0,-sgn*4,0)))
            brace.setExpression('Placement.Base.z',height_property+f' - ({h})')
            solids.append(brace)
            sx,sy=x+6,sgn*36
            # Clear the flare for the driver and washer, above the bearing land.
            if not is_motor and not sliding:
                # Continuous 45-degree access taper into the 82-degree seat.
                # Avoid the inverted annular roof at the old 11-to-8 mm step.
                lead=b.obj('Part::Cone',name+'SlopedDriverAccess')
                lead.Radius1,lead.Radius2,lead.Height=4,5.5,1.5
                lead.Placement.Base=V(sx,sy,foot_t)
                tools.append(lead)
                tools.append(b.cylinder(name+'DriverAccess',sx,sy,foot_t+1.5,5.5,h+t+2))
                tools.append(countersunk_wood_hole(b,name+'WoodScrew',sx,sy,foot_t,foot_t))
            else:
                # Flat washer lands are functional bearing surfaces. Support
                # these four small roofs when printing the holder deck-down.
                tools.append(b.cylinder(name+'DriverAccess',sx,sy,foot_t,5.5,h+t+2))
                tools.append(b.cylinder(name+'ClampBolt',sx,sy,-1,2.3,foot_t+2))
    pocket_depth=CONFIG[kind.lower()+'_body_pocket_depth_mm']
    if is_motor:
        gap=CONFIG['motor_body_pocket_clearance_per_side_mm']
        # Open toward the motor can: a closed end wall would intersect its top.
        if 'motor_mounting_pad_outer_diameter_mm' in CONFIG:
            left_expr=('-Parameters.MotorShaftToLeftHoleColumnMm'
                       ' - Parameters.MotorMountingPadOuterDiameterMm / 2'
                       ' - Parameters.MotorPadEdgeToGearboxEdgeMm'
                       ' - Parameters.MotorBodyPocketClearancePerSideMm')
            left=(-CONFIG['motor_shaft_to_left_hole_column_mm']
                  -CONFIG['motor_mounting_pad_outer_diameter_mm']/2
                  -CONFIG['motor_pad_edge_to_gearbox_edge_mm']-gap)
        else:
            left=-14-gap;left_expr=None
        pocket=b.box('MotorUndersideHousingSeat',left,-17-gap,h-1,
                     xmax+1-left,34+2*gap,pocket_depth+1)
        if left_expr:
            pocket.setExpression('Placement.Base.x',left_expr)
            pocket.setExpression('Length',f'{xmax+1} - ({left_expr})')
    else:
        pocket=b.cylinder('EncoderUndersideHousingSeat',0,0,h-1,
                           CONFIG['encoder_body_pocket_diameter_mm']/2,pocket_depth+1)
        pocket.setExpression('Radius','Parameters.EncoderBodyPocketDiameterMm / 2')
    pocket.setExpression('Placement.Base.z',height_property+' - 1')
    depth_property='Parameters.'+kind+'BodyPocketDepthMm'
    pocket.setExpression('Height',depth_property+' + 1')
    tools.append(pocket)
    if is_motor:
        clearance=CONFIG['motor_boss_clearance_diameter_mm']
        pitch_x,pitch_y=CONFIG['motor_hole_pitch_x_mm'],CONFIG['motor_hole_pitch_y_mm']
        offset=CONFIG['motor_shaft_to_left_hole_column_mm']
        for i,x in enumerate((-offset,pitch_x-offset)):
            for j,y in enumerate((-pitch_y/2,pitch_y/2)):
                tag=f'MotorAttachment{i}{j}'
                hole=b.cylinder(tag,x,y,h-1,
                                CONFIG['motor_attachment_hole_diameter_mm']/2,t+2)
                hole.setExpression('Radius','Parameters.MotorAttachmentHoleDiameterMm / 2')
                xexpr=('-Parameters.MotorShaftToLeftHoleColumnMm' if i==0 else
                       'Parameters.MotorHolePitchXMm - Parameters.MotorShaftToLeftHoleColumnMm')
                yexpr=('Parameters.MotorHolePitchYMm / 2' if j==1 else
                       '-Parameters.MotorHolePitchYMm / 2')
                hole.setExpression('Placement.Base.x',xexpr)
                hole.setExpression('Placement.Base.y',yexpr)
                hole.setExpression('Placement.Base.z',height_property+' - 1')
                tools.append(hole)
    else:
        clearance=CONFIG['encoder_boss_clearance_diameter_mm']
        for i,angle in enumerate(CONFIG['encoder_slot_angles_deg']):
            a=math.radians(angle)
            radius=CONFIG['encoder_slot_radius_mm']
            x,y=radius*math.cos(a),radius*math.sin(a)
            hole=b.capsule(f'EncoderAttachment{i}',x,y,h-1,
                           CONFIG['encoder_slot_length_mm'],
                           CONFIG['encoder_slot_width_mm'],t+2,angle)
            for slot in (hole,):
                b.shift_expression(slot,'x',f'{math.cos(a):.9f} * Parameters.EncoderSlotRadiusMm - ({x})')
                b.shift_expression(slot,'y',f'{math.sin(a):.9f} * Parameters.EncoderSlotRadiusMm - ({y})')
                for primitive in slot.Shapes:
                    z=primitive.Placement.Base.z
                    primitive.setExpression('Placement.Base.z',
                            f'{height_property} + ({z-h})')
            tools.append(hole)
    boss=b.cylinder(kind+'BossClearance',0,0,h-1,clearance/2,t+2)
    boss.setExpression('Placement.Base.z',height_property+' - 1')
    boss.setExpression('Radius',('Parameters.MotorBossClearanceDiameterMm' if is_motor
                                else 'Parameters.EncoderBossClearanceDiameterMm')+' / 2')
    tools.append(boss)
    part=b.cut(kind+'Holder',b.union(kind+'Structure',solids),b.union(kind+'HoleTools',tools))
    b.finish(part)
    # Verify actual material at the seating floor, and the open underside.
    test_x,test_y=(5,12) if is_motor else (0,14)
    assert not part.Shape.isInside(V(test_x,test_y,h+.1),1e-7,False)
    assert part.Shape.isInside(V(test_x,test_y,h+pocket_depth+.1),1e-7,False)
    assert part.Shape.isInside(V(test_x,test_y,h+t-.1),1e-7,False)
    REPORT.setdefault('housing_seats',{})[kind.lower()]={
        'outer_deck_underside_local_z_mm':h,
        'housing_seating_face_local_z_mm':h+pocket_depth,
        'plate_top_local_z_mm':h+t,
        'pocket_depth_mm':pocket_depth,'net_plate_thickness_mm':t-pocket_depth,
        'new_profile_fit_verified':False,
        'verified_material_above_seat':True,
    }
    if is_motor:
        if 'motor_mounting_pad_outer_diameter_mm' in CONFIG:
            column=-CONFIG['motor_shaft_to_left_hole_column_mm']
            case_edge=column-CONFIG['motor_mounting_pad_outer_diameter_mm']/2-CONFIG['motor_pad_edge_to_gearbox_edge_mm']
            wall_x=pocket.Shape.BoundBox.XMin
            assert math.isclose(case_edge-wall_x,gap,abs_tol=1e-7)
            assert not part.Shape.isInside(V(case_edge+.01,0,h+pocket_depth/2),1e-7,False)
            assert part.Shape.isInside(V(wall_x-.01,0,h+pocket_depth/2),1e-7,False)
            assert part.Shape.isInside(V(case_edge+.01,0,h+pocket_depth+.1),1e-7,False)
            REPORT['motor_gearbox_relief']={
                'pad_outside_diameter_user_mm':CONFIG['motor_mounting_pad_outer_diameter_mm'],
                'pad_edge_to_case_edge_user_mm':CONFIG['motor_pad_edge_to_gearbox_edge_mm'],
                'hole_column_x_mm':column,'case_edge_x_mm':case_edge,'pocket_wall_x_mm':wall_x,
                'screw_centre_to_wall_mm':column-wall_x,'assumed_fit_clearance_mm':gap,
                'remaining_near_side_lip_mm':wall_x-xmin,
                'pocket_depth_mm':pocket_depth,'mounting_plate_floor_mm':t-pocket_depth,
                'case_envelope_clear_at_seat':True,'lip_and_floor_material_checked':True,
                'physical_fit_verified':False}
        centers=[]
        for x in (-CONFIG['motor_shaft_to_left_hole_column_mm'],
                  CONFIG['motor_hole_pitch_x_mm']-CONFIG['motor_shaft_to_left_hole_column_mm']):
            for y in (-CONFIG['motor_hole_pitch_y_mm']/2,CONFIG['motor_hole_pitch_y_mm']/2):
                # Check cylindrical wall and true open center in the final CAD.
                assert not part.Shape.isInside(V(x,y,h+t-.5),1e-7,False)
                assert part.Shape.isInside(V(x,y+CONFIG['motor_attachment_hole_diameter_mm']/2+.1,h+t-.5),1e-7,False)
                centers.append([x,y])
        REPORT['motor_mounting_holes']={'centers_shaft_relative_mm':centers,
            'diameter_mm':CONFIG['motor_attachment_hole_diameter_mm'],
            'shape':'round through holes','hardware_fit_verified':False}
    return part


def slide_base(b):
    t=CONFIG['slide_base_thickness_mm']
    # Motor shaft is X=0; its asymmetrical feet are centered at X=11.
    body=b.box('SlideBaseBlank',-39,-52,0,100,104,t)
    body.setExpression('Height','Parameters.SlideBaseThicknessMm')
    tools=[]
    for i,x in enumerate((-14,36)):
        for j,y in enumerate((-36,36)):
            name=f'Rail{i}{j}'
            tools.append(b.capsule(name+'BoltSlot',x,y,-1,
                CONFIG['slide_travel_mm']+4.6,4.6,t+2))
            tools.append(b.box(name+'CaptiveNutChannel',x-11.8,y-3.7,-0.1,23.6,7.4,3.5))
    for i,x in enumerate((-31,53)):
        for j,y in enumerate((-46,46)):
            tools.append(countersunk_wood_hole(b,f'BaseWoodScrew{i}{j}',x,y,t,t))
    part=b.cut('MotorSlideBase',body,b.union('BaseHoleTools',tools))
    b.finish(part)
    return part


def export_part(part,name,part_folder,invert=False,default_count=1):
    folder=part_folder/'output/model'
    folder.mkdir(parents=True,exist_ok=True)
    stl_path=part_folder/(name+'.stl')
    shape=part.Shape.copy()
    if invert:
        shape.rotate(V(0,0,0),V(1,0,0),180)
    # Center the footprint and put the first layer at Z=0.
    bb=shape.BoundBox
    shape.translate(V(-(bb.XMin+bb.XMax)/2,-(bb.YMin+bb.YMax)/2,-bb.ZMin))
    mesh=MeshPart.meshFromShape(Shape=shape,LinearDeflection=0.06,
                              AngularDeflection=0.12,Relative=False)
    mesh.write(str(stl_path))
    Import.export([part],str(folder/(name+'.step')))
    bb=shape.BoundBox
    REPORT['parts'][name]={
        'stl':str(stl_path.relative_to(ROOT)),
        'cad_object':part.Name,'default_quantity':default_count,
        'volume_mm3':part.Shape.Volume,
        'bounds_mm':[bb.XLength,bb.YLength,bb.ZLength],
        'z_min_mm':bb.ZMin,'valid_solid':part.Shape.isValid(),
        'print_orientation':('top countersunk face against bed; underside nut channels upward'
            if part.Name=='MotorSlideBase' else 'top plate against bed; feet and housing pocket upward'),
        'support':('paint local support under four flat M4 washer lands in foot driver chambers'
            if part.Name=='MotorHolder' else 'none required by intended geometry; verify slicer preview'),
    }
    preview=ROOT/'.build'/(name+'_assembly_orientation.stl')
    preview.parent.mkdir(exist_ok=True)
    MeshPart.meshFromShape(Shape=part.Shape,LinearDeflection=.06,
                          AngularDeflection=.12,Relative=False).write(str(preview))
    REPORT['parts'][name]['preview_mesh']=str(preview.relative_to(ROOT))
    return shape


def feature_notes(doc,page,name,notes):
    old=doc.getObject(name+'FeatureNotes')
    if old:doc.removeObject(old.Name)
    for index,line in enumerate(notes):
        object_name=name+'FeatureNote'+str(index)
        old=doc.getObject(object_name)
        if old:doc.removeObject(old.Name)
        annotation=doc.addObject('TechDraw::DrawViewAnnotation',object_name)
        annotation.Text=[line]
        annotation.TextSize=2.5
        page.addView(annotation)
        annotation.X,annotation.Y=220,100-index*3.6


def native_drawing(b,part,name,folder,notes,scale=1.0,mount_kind=None,view_positions=None):
    doc=b.doc
    kind=mount_kind or part.Name
    page=doc.addObject('TechDraw::DrawPage',name+'Drawing')
    page.ProjectionType='Third angle'
    template=doc.addObject('TechDraw::DrawSVGTemplate',name+'Template')
    template.Template=App.getResourceDir()+'Mod/TechDraw/Templates/ISO/A4_Landscape_blank.svg'
    page.Template=template
    views={}
    for label,direction,horizontal,x,y in (
        ('Top',(0,0,1),(1,0,0),68,137),
        ('Front',(0,-1,0),(1,0,0),68,47),
        ('Side',(1,0,0),(0,1,0),183,47),
        ('Isometric',(1,-1,1),(1,1,0),221,139)):
        x,y=(view_positions or {}).get(label,(x,y))
        view=doc.addObject('TechDraw::DrawViewPart',name+label)
        view.Source=[part]
        view.Direction,view.XDirection=V(*direction),V(*horizontal)
        view_scale=scale*.72 if label=='Isometric' else scale
        view.ScaleType,view.Scale='Custom',view_scale
        page.addView(view)
        view.X,view.Y=x,y
        view.HardHidden=False
        views[label]=view
        annotation=doc.addObject('TechDraw::DrawViewAnnotation',name+label+'Caption')
        annotation.Text=[label]
        annotation.TextSize=3.2
        page.addView(annotation)
        normal=V(*direction); normal.normalize()
        xaxis=V(*horizontal); xaxis.normalize()
        yaxis=normal.cross(xaxis)
        projected=[vertex.Point.dot(yaxis) for vertex in part.Shape.Vertexes]
        half_height=(max(projected)-min(projected))*view_scale/2
        annotation.X=x
        annotation.Y=y+half_height+6 if label=='Isometric' else y-half_height-8
        if label!='Isometric':annotation.Y=max(6,annotation.Y)
    doc.recompute()
    vertices=part.Shape.Vertexes
    bb=part.Shape.BoundBox

    def extreme(axis, maximum):
        target=getattr(bb,axis.upper()+('Max' if maximum else 'Min'))
        matches=[(i,p.Point) for i,p in enumerate(vertices,1)
                 if abs(getattr(p.Point,axis)-target)<1e-6]
        # Prefer outside vertices on the bottom/front edge for overall dimensions.
        return 'Vertex'+str(min(matches,key=lambda pair:(pair[1].y,pair[1].z))[0]) if matches else None

    dimensions=[]
    for suffix,viewname,axis,kind,x,y in (
        ('OverallX','Top','x','DistanceX',0,bb.YLength*scale/2+9),
        ('OverallY','Top','y','DistanceY',bb.XLength*scale/2+12,0),
        ('OverallZ','Front','z','DistanceY',-bb.XLength*scale/2-12,0)):
        dim=doc.addObject('TechDraw::DrawViewDimension',name+suffix)
        dim.Type=kind
        dim.References2D=[(views[viewname],('',))]
        references=(extreme(axis,False),extreme(axis,True))
        if all(references):
            dim.References3D=[(part,references)]
        else:
            # Circular silhouettes can have extrema away from topological
            # vertices. Use a hidden geometry-derived overall datum instead.
            line=doc.addObject('Part::Line',name+suffix+'BoundsReference')
            for index,maximum in ((1,False),(2,True)):
                for coordinate in ('X','Y','Z'):
                    value=getattr(bb,coordinate+('Max' if coordinate.lower()==axis and maximum else 'Min'))
                    setattr(line,coordinate+str(index),value)
                    try:
                        line.setExpression(coordinate+str(index),part.Name+'.Shape.BoundBox.'+coordinate+('Max' if coordinate.lower()==axis and maximum else 'Min'))
                    except Exception:
                        pass
            line.Visibility=False
            doc.recompute()
            dim.References3D=[(line,('Vertex1','Vertex2'))]
        dim.FormatSpec='%.1f'
        dim.X,dim.Y=x,y
        page.addView(dim)
        dimensions.append(dim)
    # Native dimensions use actual circles or reference points linked to the
    # same parameters that place the slot primitives, rather than hand labels.
    if kind in ('MotorHolder','EncoderHolder','MotorSlideBase'):
        pairs=[]
        deck_top=bb.ZMax
        if kind=='MotorHolder':
            px,py=CONFIG['motor_hole_pitch_x_mm'],CONFIG['motor_hole_pitch_y_mm']
            offset=CONFIG['motor_shaft_to_left_hole_column_mm']
            pairs=[('MountPitchX',(-offset,-py/2,deck_top),(px-offset,-py/2,deck_top),'DistanceX',-2,-21,px),
                   ('MountPitchY',(-offset,-py/2,deck_top),(-offset,py/2,deck_top),'DistanceY',-40,0,py),
                   ('ShaftOffset',(-offset,0,deck_top),(0,0,deck_top),'DistanceX',-18,16,offset)]
        elif kind=='EncoderHolder':
            pairs=[('SlotRadius',(0,0,deck_top),(14.5,0,deck_top),'DistanceX',7,23,14.5)]
        else:
            pairs=[('Travel',(-7.5,36,6),(7.5,36,6),'DistanceX',0,18,15)]
        for suffix,start,end,kind,x,y,expected_value in pairs:
            line=doc.addObject('Part::Line',name+suffix+'Reference')
            for index,coordinates in enumerate((start,end),1):
                for axis,value in zip(('X','Y','Z'),coordinates):
                    setattr(line,axis+str(index),value)
            if kind=='MotorHolder':
                for field in ('X1','X2'):
                    line.setExpression(field,'-Parameters.MotorShaftToLeftHoleColumnMm')
                line.setExpression('Y1','-Parameters.MotorHolePitchYMm / 2')
                line.setExpression('Y2','Parameters.MotorHolePitchYMm / 2' if suffix=='MountPitchY'
                                   else '-Parameters.MotorHolePitchYMm / 2')
                if suffix=='MountPitchX':
                    line.setExpression('X2','Parameters.MotorHolePitchXMm - Parameters.MotorShaftToLeftHoleColumnMm')
                elif suffix=='ShaftOffset':
                    line.setExpression('X2','0');line.setExpression('Y1','0');line.setExpression('Y2','0')
            elif kind=='EncoderHolder':
                line.setExpression('X2','Parameters.EncoderSlotRadiusMm')
            else:
                line.setExpression('X1','-14 - Parameters.SlideTravelMm / 2')
                line.setExpression('X2','-14 + Parameters.SlideTravelMm / 2')
            for field in ('Z1','Z2'):
                line.setExpression(field,'Parameters.SlideBaseThicknessMm' if kind=='MotorSlideBase'
                                   else ('Parameters.MotorStandHeightMm' if kind=='MotorHolder'
                                         else 'Parameters.EncoderStandHeightMm')+' + Parameters.DeckThicknessMm')
            line.Visibility=False
            doc.recompute()
            dim=doc.addObject('TechDraw::DrawViewDimension',name+suffix)
            dim.Type=kind
            dim.References2D=[(views['Top'],('',))]
            dim.References3D=[(line,('Vertex1','Vertex2'))]
            dim.FormatSpec='%.1f'
            dim.X,dim.Y=x,y
            page.addView(dim)
            dimensions.append(dim)
        if kind in ('MotorHolder','EncoderHolder'):
            radius=(CONFIG['motor_boss_clearance_diameter_mm'] if kind=='MotorHolder'
                    else CONFIG['encoder_boss_clearance_diameter_mm'])/2
            edge=next(f'Edge{i}' for i,e in enumerate(part.Shape.Edges,1)
                      if isinstance(e.Curve,Part.Circle) and abs(e.Curve.Radius-radius)<1e-6)
            dim=doc.addObject('TechDraw::DrawViewDimension',name+'BossDiameter')
            dim.Type='Diameter'
            dim.References2D=[(views['Top'],('',))]
            dim.References3D=[(part,(edge,))]
            dim.FormatSpec='D %.1f'
            dim.X,dim.Y=27,16
            page.addView(dim)
            dimensions.append(dim)
    heading=doc.addObject('TechDraw::DrawViewAnnotation',name+'Title')
    heading.Text=[name.replace('_',' ').upper()+' | mm | '+REV+(' FIT CHECK' if REV=='r06' else ' PHOTO FIT'),
                  'Mounting geometry inferred from references; verify with hardware.']
    heading.TextSize=3.2
    page.addView(heading)
    heading.X,heading.Y=148,201
    feature_notes(doc,page,name,notes)
    page.ViewObject.show()
    doc.recompute(None,True,True)
    app=QtWidgets.QApplication.instance()
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        app.processEvents()
        if all(dim.getRawValue()>0 for dim in dimensions): break
        time.sleep(0.05)
    actual=[dim.getRawValue() for dim in dimensions]
    expected=[bb.XLength,bb.YLength,bb.ZLength]
    if kind=='MotorHolder':expected.extend((CONFIG['motor_hole_pitch_x_mm'],CONFIG['motor_hole_pitch_y_mm'],CONFIG['motor_shaft_to_left_hole_column_mm'],CONFIG['motor_boss_clearance_diameter_mm']))
    elif kind=='EncoderHolder':expected.extend((CONFIG['encoder_slot_radius_mm'],CONFIG['encoder_boss_clearance_diameter_mm']))
    elif kind=='MotorSlideBase':expected.append(CONFIG['slide_travel_mm'])
    assert all(abs(a-e)<1e-6 for a,e in zip(actual,expected)), (name,actual,expected)
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/(name+'_dimensioned.svg')
    TechDrawGui.exportPageAsSvg(page,str(file))
    REPORT['drawings'].append({'svg':str(file.relative_to(ROOT)),
                              'native_dimensions_mm':actual,'orthographic_views':3,
                              'isometric_views':1})
    return page


def housing_seat_drawing(b,part,name,folder,kind):
    """Expose the hidden pocket with native views and dimensions of real edges."""
    doc=b.doc
    depth=CONFIG[kind+'_body_pocket_depth_mm']
    floor=CONFIG['deck_thickness_mm']-depth
    page=doc.addObject('TechDraw::DrawPage',name+'SeatDetail')
    template=doc.addObject('TechDraw::DrawSVGTemplate',name+'SeatTemplate')
    template.Template=App.getResourceDir()+'Mod/TechDraw/Templates/ISO/A4_Landscape_blank.svg'
    page.Template=template
    views={}
    for label,direction,horizontal,x,y,scale in (
        ('Underside',(0,0,-1),(1,0,0),75,135,.9),
        ('Side',(1,0,0),(0,1,0),75,48,.9),
        ('Isometric',(1,-1,-1),(1,1,0),218,145,.7)):
        view=doc.addObject('TechDraw::DrawViewPart',name+'Seat'+label)
        view.Source=[part];view.Direction=V(*direction);view.XDirection=V(*horizontal)
        view.ScaleType='Custom';view.Scale=scale
        view.HardHidden=label=='Side'
        page.addView(view);view.X,view.Y=x,y
        views[label]=view
        caption=doc.addObject('TechDraw::DrawViewAnnotation',name+'Seat'+label+'Caption')
        caption.Text=[label];caption.TextSize=3
        page.addView(caption)
        caption.X,caption.Y=x,107 if label=='Isometric' else (15 if label=='Side' else y-45)
    dims=[];expected=[]
    for label,length,x in (('PocketDepth',depth,-49),('PlateFloor',floor,49)):
        edge=next('Edge'+str(i) for i,e in enumerate(part.Shape.Edges,1)
                  if abs(e.BoundBox.ZLength-length)<1e-6 and
                  e.BoundBox.XLength<1e-6 and e.BoundBox.YLength<1e-6 and
                  e.BoundBox.ZMin>=CONFIG[kind+'_stand_height_mm']-1e-6)
        dim=doc.addObject('TechDraw::DrawViewDimension',name+label)
        dim.Type='DistanceY';dim.References2D=[(views['Side'],('',))]
        dim.References3D=[(part,(edge,))];dim.FormatSpec='%.1f'
        page.addView(dim);dim.X,dim.Y=x,0;dims.append(dim);expected.append(length)
    if kind=='encoder':
        size=CONFIG['encoder_body_pocket_diameter_mm'];typ='Diameter'
        edge=next('Edge'+str(i) for i,e in enumerate(part.Shape.Edges,1)
                  if isinstance(e.Curve,Part.Circle) and abs(e.Curve.Radius-size/2)<1e-6)
    else:
        size=34+2*CONFIG['motor_body_pocket_clearance_per_side_mm'];typ='DistanceY'
        edge=next('Edge'+str(i) for i,e in enumerate(part.Shape.Edges,1)
                  if abs(e.BoundBox.YLength-size)<1e-6 and e.BoundBox.XLength<1e-6
                  and e.BoundBox.ZLength<1e-6)
    dim=doc.addObject('TechDraw::DrawViewDimension',name+'SeatSize')
    dim.Type=typ;dim.References2D=[(views['Underside'],('',))]
    dim.References3D=[(part,(edge,))];dim.FormatSpec=('D ' if typ=='Diameter' else '')+'%.1f'
    page.addView(dim);dim.X,dim.Y=46,0;dims.append(dim);expected.append(size)
    relief=REPORT.get('motor_gearbox_relief') if kind=='motor' else None
    if relief:
        wall='MotorUndersideHousingSeat.Placement.Base.x'
        for suffix,start,end,x,y,want in [
            ('CentreToRecessWall',wall,'-Parameters.MotorShaftToLeftHoleColumnMm',-22,23,relief['screw_centre_to_wall_mm']),
            ('NearSideLip','-20 mm',wall,-27,-23,relief['remaining_near_side_lip_mm'])]:
            line=doc.addObject('Part::Line',name+suffix+'Reference')
            line.setExpression('X1',start);line.setExpression('X2',end)
            for index in (1,2):
                line.setExpression('Y'+str(index),'-Parameters.MotorHolePitchYMm / 2')
                line.setExpression('Z'+str(index),'Parameters.MotorStandHeightMm + Parameters.MotorBodyPocketDepthMm')
            line.Visibility=False;doc.recompute()
            dim=doc.addObject('TechDraw::DrawViewDimension',name+suffix)
            dim.Type='DistanceX';dim.References2D=[(views['Underside'],('',))]
            dim.References3D=[(line,('Vertex1','Vertex2'))];dim.FormatSpec='%.2f'
            page.addView(dim);dim.X,dim.Y=x,y;dims.append(dim);expected.append(want)
    heading=doc.addObject('TechDraw::DrawViewAnnotation',name+'SeatTitle')
    heading.Text=[name.replace('_',' ').upper()+' | UNDERSIDE SEAT | '+REV+(' FIT CHECK' if REV=='r06' else ' PHOTO FIT')]
    heading.TextSize=3.2;page.addView(heading);heading.X,heading.Y=148,201
    detail_notes=[
        ('Round M3 holes; no top head recesses' if kind=='motor' else 'Plain through-slots; no top head recesses'),
        f'Underside pocket depth: {depth:.1f}',
        f'Material over device face: {floor:.1f}',
        'Seat face remains 42 above plywood',
        ('Near-side recess uses measured pad / case gap' if relief else 'Housing profile clearance is photo-derived'),
        ('Encoder top nominally flush with 5 mm boss' if kind=='encoder'
         else 'Motor boss near pad height; retain 3 mm floor'),
        ('#6: sloped access into countersink' if kind=='encoder'
         else 'Support the four flat M4 washer lands'),
        'Print top face down; housing pocket opens up']
    if relief:
        detail_notes.extend(['Pad OD 7.70; pad-to-case gap 2.00 (user)',
                             'Near-side fit clearance 0.60 (assumed)',
                             'Centre to recess wall 6.45; lip width 3.85',
                             'Recess widened 1.55 on near side'])
    feature_notes(doc,page,name+'Seat',detail_notes)
    page.ViewObject.show();doc.recompute(None,True,True)
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        QtWidgets.QApplication.instance().processEvents()
        if all(d.getRawValue()>0 for d in dims):break
        time.sleep(.05)
    actual=[d.getRawValue() for d in dims]
    assert all(abs(a-e)<1e-6 for a,e in zip(actual,expected)),(name,actual,expected)
    path=folder/(name+'_seat_detail.svg')
    TechDrawGui.exportPageAsSvg(page,str(path))
    REPORT['drawings'].append({'svg':str(path.relative_to(ROOT)),
        'native_dimensions_mm':actual,'view_labels':['Underside','Side','Isometric'],
        'detail':'underside housing seat, net plate thickness and pocket depth'})


def reference_assembly(motor,encoder,base):
    b=Builder('LisaAssemblyReference')
    distance=CONFIG['assembly_center_distance_mm']
    pad=CONFIG['assembly_pad_thickness_mm']
    base_t=CONFIG['slide_base_thickness_mm']
    encoder_foot=CONFIG['encoder_foot_thickness_mm']

    def add(name,shape,offset,color,reference=False):
        copy=shape.copy()
        copy.translate(V(*offset))
        obj=b.feature(name,copy)
        obj.ViewObject.ShapeColor=color
        if reference:
            obj.Label=name+' (estimated hardware reference; not printable)'
        path=ROOT/'.build'/('assembly_'+name+'.stl')
        path.parent.mkdir(exist_ok=True)
        MeshPart.meshFromShape(Shape=copy,LinearDeflection=.15,AngularDeflection=.18,
                              Relative=False).write(str(path))
        REPORT['assembly'].append({'name':name,'mesh':str(path.relative_to(ROOT)),
                                  'color':color,'hardware_reference':reference})
        return obj

    add('MotorHolder',motor.Shape,(0,0,base_t+pad),(.12,.47,.59))
    add('MotorSlideBase',base.Shape,(0,0,pad),(.12,.47,.59))
    add('EncoderHolder',encoder.Shape,(-distance,0,pad),(.13,.55,.42))
    face=CONFIG['encoder_stand_height_mm']+CONFIG['encoder_body_pocket_depth_mm']+pad
    assert math.isclose(CONFIG['motor_stand_height_mm']+CONFIG['motor_body_pocket_depth_mm']+base_t,face-pad)
    metal=(.65,.68,.70)
    gearbox=Part.makeBox(50,34,25,V(-14,-17,face-25))
    can=Part.makeCylinder(12,33,V(36,0,face-12.5),V(1,0,0))
    motor_boss=CONFIG['motor_reference_boss_above_mounting_pads_mm']
    # Motor side photo shows bearing collar about level with the raised M3
    # mounting pads; its front is therefore not 3 mm above that mounting plane.
    boss=Part.makeCylinder(7,2+motor_boss,V(0,0,face-2))
    shaft=Part.makeCylinder(3,14,V(0,0,face+motor_boss))
    add('MotorApproximateEnvelope',Part.makeCompound([gearbox,can,boss,shaft]),(0,0,0),metal,True)
    enc=Part.makeCylinder(19,37,V(-distance,0,face-37))
    encoder_boss=CONFIG['encoder_reference_boss_height_mm']
    boss=Part.makeCylinder(10,encoder_boss,V(-distance,0,face))
    shaft=Part.makeCylinder(3,13,V(-distance,0,face+encoder_boss))
    add('EncoderApproximateEnvelope',Part.makeCompound([enc,boss,shaft]),(0,0,0),(.22,.24,.26),True)
    # Axial arrangement is a clearance example; actual hub lengths are unknown.
    groove_bottom=face+16
    for label,cx,radius,flange,hub in (
        ('Motor80TPulley',0,25.2,27.5,13),
        ('Encoder20TPulley',-distance,6.1,8.3,6)):
        pieces=[Part.makeCylinder(hub,8,V(cx,0,face+8)),
                Part.makeCylinder(flange,1,V(cx,0,groove_bottom-1)),
                Part.makeCylinder(radius,6,V(cx,0,groove_bottom)),
                Part.makeCylinder(flange,1,V(cx,0,groove_bottom+6))]
        shape=pieces[0].multiFuse(pieces[1:]).cut(
            Part.makeCylinder(3.05,30,V(cx,0,face+5))).removeSplitter()
        add(label,shape,(0,0,0),metal,True)

    def belt_wire(R,r):
        nx=-(R-r)/distance
        ny=math.sqrt(1-nx*nx)
        a,bp=V(nx*R,ny*R,0),V(nx*R,-ny*R,0)
        c,d=V(-distance+nx*r,-ny*r,0),V(-distance+nx*r,ny*r,0)
        edges=[Part.Arc(a,V(R,0,0),bp).toShape(),Part.makeLine(bp,c),
               Part.Arc(c,V(-distance-r,0,0),d).toShape(),Part.makeLine(d,a)]
        return Part.Wire(edges)

    R,r=80*2/(2*math.pi),20*2/(2*math.pi)
    band=Part.Face(belt_wire(R+1.2,r+1.2)).cut(Part.Face(belt_wire(R,r)))
    belt=band.extrude(V(0,0,6))
    add('BeltApproximateEnvelope',belt,(0,0,groove_bottom),(.07,.08,.09),True)
    if pad>0:
        add('FutureMotorTPUPad',Part.makeBox(100,104,pad,V(-39,-52,0)),(0,0,0),(.34,.34,.37),True)
        add('FutureEncoderTPUPad',Part.makeBox(60,84,pad,V(-30,-42,0)),(-distance,0,0),(.34,.34,.37),True)
    doc=b.doc
    doc.recompute()
    collisions=[]
    for hardware in ('MotorApproximateEnvelope','EncoderApproximateEnvelope',
                     'Motor80TPulley','Encoder20TPulley','BeltApproximateEnvelope'):
        for printed in ('MotorHolder','MotorSlideBase','EncoderHolder'):
            overlap=doc.getObject(hardware).Shape.common(doc.getObject(printed).Shape).Volume
            assert overlap<1e-5,(hardware,printed,overlap)
            collisions.append({'reference':hardware,'printed':printed,'overlap_mm3':overlap})
    REPORT['reference_clearance_checks']=collisions
    motor_envelope=doc.getObject('MotorApproximateEnvelope').Shape
    encoder_envelope=doc.getObject('EncoderApproximateEnvelope').Shape
    REPORT['height_clearances_mm']={
        'device_mounting_faces_above_plywood':face,
        'motor_plate_top_above_plywood':base_t+pad+motor.Shape.BoundBox.ZMax,
        'encoder_plate_top_above_plywood':pad+encoder.Shape.BoundBox.ZMax,
        'encoder_nominal_boss_flush_error':pad+encoder.Shape.BoundBox.ZMax-(face+encoder_boss),
        'motor_nominal_boss_below_plate_top':base_t+pad+motor.Shape.BoundBox.ZMax-(face+motor_boss),
        'motor_reference_body_above_slide_base':motor_envelope.BoundBox.ZMin-(base_t+pad),
        'encoder_reference_body_above_plywood':encoder_envelope.BoundBox.ZMin,
    }
    assert REPORT['height_clearances_mm']['motor_reference_body_above_slide_base']>=10
    assert REPORT['height_clearances_mm']['encoder_reference_body_above_plywood']>=5
    assert abs(REPORT['height_clearances_mm']['encoder_nominal_boss_flush_error'])<1e-6
    # Verify the printed stacks at both travel limits, using the fixed encoder.
    travel_checks=[]
    for shift in (-CONFIG['slide_travel_mm']/2,CONFIG['slide_travel_mm']/2):
        moved=motor.Shape.copy(); moved.translate(V(shift,0,base_t+pad))
        for fixed in ('EncoderHolder',):
            overlap=moved.common(doc.getObject(fixed).Shape).Volume
            assert overlap<1e-5,(shift,fixed,overlap)
            travel_checks.append({'motor_shift_mm':shift,'fixed':fixed,'overlap_mm3':overlap})
    REPORT['travel_clearance_checks']=travel_checks
    REPORT['attachment_screw_stack']={}
    for kind in ('motor','encoder'):
        floor=CONFIG['deck_thickness_mm']-CONFIG[kind+'_body_pocket_depth_mm']
        length=CONFIG[kind+'_attachment_screw_length_mm']
        washer=CONFIG['attachment_washer_thickness_mm_assumed']
        engagement=length-floor-washer
        assert math.isclose(engagement,2.4)
        REPORT['attachment_screw_stack'][kind]={
            'screw_length_mm':length,'washer_mm_assumed':washer,
            'bearing_floor_mm':floor,'nominal_engagement_mm':engagement,
            'usable_thread_depth_verified':False,
            'user_reported_approximate_depth_mm':4 if kind=='motor' else None,
        }
    # Simplified fastening hardware illustrates access and the assembly stack.
    # Threads, actual head shapes and washers are not measured substitutes.
    steel=(.48,.51,.55)
    # Envelope check for exposed M3 heads after removing the top recesses.
    m3_checks=[]
    motor_points=[(x,y) for x in (-CONFIG['motor_shaft_to_left_hole_column_mm'],
                                 CONFIG['motor_hole_pitch_x_mm']-CONFIG['motor_shaft_to_left_hole_column_mm'])
                       for y in (-CONFIG['motor_hole_pitch_y_mm']/2,CONFIG['motor_hole_pitch_y_mm']/2)]
    encoder_points=[(-distance+14.5*math.cos(math.radians(a)),
                     14.5*math.sin(math.radians(a))) for a in (0,120,240)]
    for kind,points,top in (
        ('Motor',motor_points,base_t+pad+motor.Shape.BoundBox.ZMax),
        ('Encoder',encoder_points,pad+encoder.Shape.BoundBox.ZMax)):
        for index,(x,y) in enumerate(points):
            washer=Part.makeCylinder(3.25,.6,V(x,y,top)).cut(
                Part.makeCylinder(1.65,.6,V(x,y,top)))
            head=Part.makeCylinder(2.75,3,V(x,y,top+.6))
            envelope=washer.fuse(head)
            add(kind+'AttachmentM3_'+str(index),envelope,(0,0,0),steel,True)
            shifts=(-7.5,0,7.5) if kind=='Motor' else (0,)
            for shift in shifts:
                moved=envelope.copy();moved.translate(V(shift,0,0))
                for pulley in ('Motor80TPulley','Encoder20TPulley'):
                    reference=doc.getObject(pulley).Shape.copy()
                    if pulley=='Motor80TPulley':reference.translate(V(shift,0,0))
                    overlap=moved.common(reference).Volume
                    assert overlap<1e-5,(kind,index,shift,pulley,overlap)
                    m3_checks.append({'head':kind+str(index),'shift_mm':shift,
                                      'pulley':pulley,'overlap_mm3':overlap})
    REPORT['m3_head_pulley_clearance_checks']=m3_checks
    for i,x in enumerate((-14,36)):
        for j,y in enumerate((-36,36)):
            washer=Part.makeCylinder(4.5,1,V(x,y,pad+11)).cut(
                Part.makeCylinder(2.3,1,V(x,y,pad+11)))
            screw=Part.makeCylinder(2,12,V(x,y,pad)).fuse(
                Part.makeCylinder(3.5,2.5,V(x,y,pad+12)))
            add(f'MotorClampM4_{i}{j}',Part.makeCompound([washer,screw]),(0,0,0),steel,True)
    for label,points,top,length in (
        ('MotorWoodScrew',[(x,y) for x in (-31,53) for y in (-46,46)],pad+6,12.7 if pad==0 else 15.875),
        ('EncoderWoodScrew',[(x-distance,y) for x in (-24,24) for y in (-36,36)],pad+encoder_foot,12.7 if pad==0 else 15.875)):
        for i,(x,y) in enumerate(points):
            depth=(4-2.1)/math.tan(math.radians(41))
            head=Part.makeCone(2.1,4,depth,V(x,y,top-depth))
            shank=Part.makeCylinder(1.75,length-depth,V(x,y,top-length))
            add(label+str(i),head.fuse(shank),(0,0,0),steel,True)
    for obj in doc.Objects:
        if hasattr(obj,'Shape') and not obj.Shape.isNull():
            assert obj.Shape.isValid(),obj.Name
    Gui.activeDocument().activeView().viewAxonometric()
    Gui.activeDocument().activeView().fitAll()
    doc.saveAs(str(ROOT/'motor holder/output/model/motor_encoder_assembly_reference.FCStd'))
    REPORT['assembly_note']='Current assembly has no TPU pads. Motor adjusts; encoder remains fixed. Hardware envelopes, 2 mm belt pitch, 6 mm belt width and pulley axial positions are unverified reference assumptions. Additional encoder gear is not dimensioned or modeled.'


def main():
    if not App.GuiUp:
        raise RuntimeError('Use build_mounts.sh to run inside FreeCAD.')
    App.ParamGet('User parameter:BaseApp/Preferences/Document').RemBool('SaveThumbnail')
    cache=Path(sys.argv[1]); cache.mkdir(parents=True,exist_ok=True)
    measurement=json.loads((SCRIPT_DIR/'motor_hole_measurements.json').read_text())
    for key,value in measurement['nominal_model_mm'].items():assert CONFIG[key]==value,key
    REPORT['motor_hole_measurement_evidence']={
        'json':'scripts/motor_hole_measurements.json','csv':'scripts/motor_hole_measurements.csv',
        'source_photo':measurement['source'],'sha256':measurement['sha256'],
        'measured_mm':measurement['measured_mm'],'nominal_model_mm':measurement['nominal_model_mm'],
        'estimated_uncertainty_mm':measurement['estimated_dimension_uncertainty_mm']}
    motorfolder=ROOT/'motor holder/output'
    encfolder=ROOT/'encoder holder/output'
    mb=Builder('LisaMotorHolder')
    motor=stand(mb,'Motor')
    export_part(motor,'motor_holder_'+REV,motorfolder.parent,True)
    native_drawing(mb,motor,'motor_holder_'+REV,motorfolder/'render',[
        f"Motor mount: M3 x 6; round holes D {CONFIG['motor_attachment_hole_diameter_mm']:.1f}",
        f"Hole centers: {CONFIG['motor_hole_pitch_x_mm']:.1f} x {CONFIG['motor_hole_pitch_y_mm']:.1f} (photo)",
        f"Shaft to near column: {CONFIG['motor_shaft_to_left_hole_column_mm']:.1f} (photo)",
        'Boss opening: diameter 16.0',
        'Underside gearbox seat: depth 4.0, open end',
        'Floor 3.0 + washer 0.6: engagement 2.4',
        'Feet: 5.0 thick; 4 holes diameter 4.6 for M4',
        'Local support at four flat M4 washer lands'])
    housing_seat_drawing(mb,motor,'motor_holder_'+REV,motorfolder/'render','motor')
    base=slide_base(mb)
    export_part(base,'motor_slide_base_'+REV,motorfolder.parent,True)
    native_drawing(mb,base,'motor_slide_base_'+REV,motorfolder/'render',[
        'Motor travel: 15.0 along X; 4 M4 clamps',
        '4 slots: length 19.6 / width 4.6',
        'Clamp centers: 50.0 x 72.0; center X=11',
        'Underside M4 nut channels: 23.6 x 7.4',
        'Nut channel depth: 3.4',
        'Wood hole centers: 84.0 x 92.0',
        '#6 clearance 4.2; head seat 8 / 82 deg',
        'Print countersinks DOWN / nut channels UP'],.85)
    mb.finish(motor)
    mb.doc.saveAs(str(motorfolder/('model/motor_holder_'+REV+'.FCStd')))

    eb=Builder('LisaEncoderHolder')
    encoder=stand(eb,'Encoder')
    export_part(encoder,'encoder_holder_'+REV,encfolder.parent,True)
    native_drawing(eb,encoder,'encoder_holder_'+REV,encfolder/'render',[
        'Fixed encoder: assumed M3 x 8 + washers',
        'Use only the actual mounting-hole group',
        'Slot centers: radius 14.5; length 5.5 / width 3.8',
        'Angles: 0, 30, 120, 150, 240, 270 deg',
        'Boss opening: diameter 21.5',
        'Underside body seat: diameter 39.5 / depth 2',
        'Floor 5.0; nominally flush with bearing boss',
        'Feet 6.0: #6 seat 8 / 82 deg; sloped access'])
    housing_seat_drawing(eb,encoder,'encoder_holder_'+REV,encfolder/'render','encoder')
    eb.finish(encoder)
    eb.doc.saveAs(str(encfolder/('model/encoder_holder_'+REV+'.FCStd')))
    reference_assembly(motor,encoder,base)
    for docname in (mb.doc.Name,eb.doc.Name):
        App.closeDocument(docname)
    for file in (motorfolder/('model/motor_holder_'+REV+'.FCStd'),encfolder/('model/encoder_holder_'+REV+'.FCStd')):
        doc=App.openDocument(str(file))
        for obj in doc.Objects:
            if obj.Name in ('MotorHolder','EncoderHolder','MotorSlideBase'):
                assert obj.Shape.isValid() and len(obj.Shape.Solids)==1
        App.closeDocument(doc.Name)
    REPORT['native_documents_reopened']=True
    (SCRIPT_DIR/'build_report.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    (cache/'freecad_complete').write_text(CONFIG['revision']+'\n')
    print('Completed CAD, STL, STEP, native drawings and reference assembly.')


if __name__=='__main__':
    main()
