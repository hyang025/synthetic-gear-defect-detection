"""Update moved assembly scene output paths without rendering or editing geometry."""
from pathlib import Path
import hashlib
import json
import sys
from array import array

import bpy

SCRIPTS=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPTS))
from project_paths import ASSEMBLY_RENDER


def geometry_signature():
    digest=hashlib.sha256()
    bpy.context.view_layer.update()
    for obj in sorted(bpy.context.scene.objects,key=lambda x:x.name):
        digest.update(obj.name.encode())
        digest.update(str(tuple(round(v,7) for row in obj.matrix_world for v in row)).encode())
        if obj.type=='MESH':
            coords=array('f',[0])* (len(obj.data.vertices)*3)
            obj.data.vertices.foreach_get('co',coords);digest.update(coords.tobytes())
            indices=array('i',[0])*len(obj.data.loops)
            obj.data.loops.foreach_get('vertex_index',indices);digest.update(indices.tobytes())
            digest.update(str(tuple(p.loop_total for p in obj.data.polygons)).encode())
    return digest.hexdigest()


results=[]
for path in sorted(ASSEMBLY_RENDER.glob('two_belt_*_r06.blend')):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    original=geometry_signature()
    previous=bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath=str(ASSEMBLY_RENDER/Path(previous).name)
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path))
    assert geometry_signature()==original,path
    assert Path(bpy.context.scene.render.filepath).parent==ASSEMBLY_RENDER
    results.append({'scene':str(path.relative_to(ASSEMBLY_RENDER.parents[2])),
                    'previous_output':previous,'current_output':bpy.context.scene.render.filepath,
                    'geometry_signature_unchanged':True})
assert len(results)==3
(SCRIPTS/'saved_scene_relocation_report.json').write_text(json.dumps(results,indent=2)+'\n')
print('Updated three saved scene output paths; scene geometry signatures match.')
