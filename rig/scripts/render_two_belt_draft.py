"""Render the authoritative FreeCAD two-belt draft with the existing GPU helper."""
import json
import os
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

SCRIPTS=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPTS))
from render_device import configure_cycles_device
from project_paths import ASSEMBLY_RENDER

ROOT=SCRIPTS.parent
OUT=ASSEMBLY_RENDER
TAG=os.environ.get('LISA_RENDER_TAG','draft')
REPORT_PATH=SCRIPTS/os.environ.get('LISA_ASSEMBLY_REPORT','two_belt_draft_report.json')
REPORT=json.loads(REPORT_PATH.read_text())


def material(name,color,metal=0,roughness=.34):
    mat=bpy.data.materials.new(name)
    mat.use_nodes=True
    bsdf=mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value=(*color,1)
    bsdf.inputs['Metallic'].default_value=metal
    bsdf.inputs['Roughness'].default_value=roughness
    return mat


def main():
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC'
    scene.render.engine='CYCLES'
    scene.cycles.samples=48
    scene.cycles.use_denoising=True
    REPORT['render_device']=configure_cycles_device(scene)
    scene.render.image_settings.file_format='PNG'
    scene.render.film_transparent=True
    scene.view_settings.view_transform='AgX'
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.85,.85,.85,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.45
    objects={}
    for group,items in (('assembly',REPORT['items']),('detail',REPORT['detail_items']),
                        ('exploded',REPORT['exploded_items'])):
        objects[group]=[]
        for entry in items:
            bpy.ops.wm.stl_import(filepath=str(ROOT/entry['mesh']),global_scale=.001)
            obj=bpy.context.object
            obj.name=group+'_'+entry['name']
            if group=='exploded':
                obj.location += Vector(entry['exploded_offset_mm'])*.001
            color=entry['color']
            if group=='detail' and entry['name']=='SteelDShaft100mm':
                color=(.88,.56,.12)
            metal=.65 if not entry['printed'] and 'Belt' not in entry['name'] else .0
            obj.data.materials.append(material(obj.name,color,metal))
            obj.hide_render=True
            objects[group].append(obj)
    bpy.ops.mesh.primitive_cube_add(size=1,location=(.015,0,-.005))
    wood=bpy.context.object
    wood.name='IllustrativePlywood10mm'
    wood.dimensions=(.280,.160,.010)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    wood.data.materials.append(material('Plywood',(.60,.42,.25),0,.68))
    bevel=wood.modifiers.new('Board edge softness','BEVEL')
    bevel.width=.0005
    bevel.segments=3
    target=Vector((.015,0,.06))
    bpy.ops.object.camera_add()
    camera=bpy.context.object
    camera.data.type='ORTHO'
    camera.data.clip_start=.0001
    scene.camera=camera
    for delta,power,size in (((-.06,-.14,.30),4,.20),((.19,.06,.19),3,.18),((-.20,.08,.15),2,.16)):
        bpy.ops.object.light_add(type='AREA',location=target+Vector(delta))
        light=bpy.context.object
        light.data.energy=power
        light.data.size=size
        light.rotation_euler=(target-light.location).to_track_quat('-Z','Y').to_euler()
    views=[
        ('assembly','assembly',(.26,-.47,.32),(.015,0,.056),.355,(1700,1150)),
        ('top','assembly',(.015,0,.45),(.015,0,.06),.320,(1700,1150)),
        ('front','assembly',(.015,-.55,.065),(.015,0,.065),.320,(1700,1050)),
        ('side','assembly',(.50,0,.065),(.015,0,.065),.190,(1100,1050)),
        ('detail','detail',(.025,-.32,.123),(0,0,.076),.155,(1250,1450)),
        ('exploded','exploded',(.19,-.38,.26),(0,0,.060),.260,(1600,1300))]
    projections={}
    for view,group,location,look_at,scale,size in views:
        for group_name,items in objects.items():
            for obj in items:
                obj.hide_render=group_name!=group
        wood.hide_render=group!='assembly'
        camera.location=Vector(location)
        camera.rotation_euler=(Vector(look_at)-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.data.ortho_scale=scale
        scene.render.resolution_x,scene.render.resolution_y=size
        scene.render.resolution_percentage=100
        bpy.context.view_layer.update()
        anchors={}
        for name,point in REPORT['anchors_mm'].items():
            point=Vector(point)
            if view=='detail':
                # Section labels must land on the retained cut faces, not the removed front half.
                section_points={'adapter':(-7,0,124),'upper_collar':(-6,0,102.5),
                                'lower_collar':(-6,0,32.5),'upper_bearing':(-5,0,95.5),
                                'lower_bearing':(-5,0,40),'upper80':(-17,0,77),'lower80':(17,0,61)}
                point=Vector(section_points.get(name,point))
            if view=='exploded':
                offsets={'panel':(0,0,0),'foot':(0,0,0),
                         'upper_plate':(0,0,26),'lower_plate':(0,0,0),
                         'upper_bearing':(0,0,26),'lower_bearing':(0,0,0)}
                point += Vector(offsets.get(name,(0,0,0)))
            uv=world_to_camera_view(scene,camera,point*.001)
            anchors[name]=[round(uv.x*size[0],2),round((1-uv.y)*size[1],2)]
        projections[view]={'size':list(size),'anchors':anchors}
        scene.render.filepath=str(OUT/('two_belt_'+view+'_'+TAG+'_raw.png'))
        # Keep editable assembly/detail/exploded scenes; other views use the same CAD meshes.
        if view in ('assembly','detail','exploded'):
            bpy.ops.wm.save_as_mainfile(filepath=str(OUT/('two_belt_'+view+'_'+TAG+'.blend')))
        bpy.ops.render.render(write_still=True)
    (SCRIPTS/('two_belt_'+TAG+'_projection.json')).write_text(json.dumps(projections,indent=2)+'\n')
    REPORT_PATH.write_text(json.dumps(REPORT,indent=2)+'\n')
    if TAG=='r06':
        release_path=SCRIPTS/'two_belt_r06_report.json'
        release=json.loads(release_path.read_text())
        for items in objects.values():
            for obj in items:obj.hide_render=True
        wood.hide_render=True
        scene.render.resolution_x=1100; scene.render.resolution_y=900
        selected=set(filter(None,os.environ.get('LISA_RENDER_PARTS','').split(',')))
        for name,part in release['parts'].items():
            if selected and name not in selected:continue
            bpy.ops.wm.stl_import(filepath=str(ROOT/part['stl']),global_scale=.001)
            obj=bpy.context.object
            bpy.context.view_layer.update()
            bounds=[obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            low=Vector(tuple(min(v[i] for v in bounds) for i in range(3)))
            high=Vector(tuple(max(v[i] for v in bounds) for i in range(3)))
            target=(low+high)/2
            camera.location=target+Vector((.13,-.19,.16))
            camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
            camera.data.ortho_scale=max(high-low)*1.7
            color=(.95,.37,.06) if name.startswith('gear_') else (.13,.55,.4) if name.startswith('encoder_') else (.07,.42,.61)
            obj.data.materials.append(material(name,color))
            output=ROOT/Path(part['stl']).parent/'output/render'/(name+'_print_preview.png')
            scene.render.filepath=str(output)
            bpy.ops.render.render(write_still=True)
            part['preview_render']=str(output.relative_to(ROOT))
            obj.hide_render=True
        release['render_device']=REPORT['render_device']
        release_path.write_text(json.dumps(release,indent=2)+'\n')
    print('Rendered six CAD draft views on the configured Cycles device.')


if __name__=='__main__':
    main()
