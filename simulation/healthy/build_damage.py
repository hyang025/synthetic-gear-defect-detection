"""Create three ellipsoidal dents and matched-resolution control/damaged meshes.

Run using ./tools/freecad freecadcmd healthy/build_damage.py. Dimensions are mm.
The original healthy CAD and mesh are never modified.
"""

import json
import math
from pathlib import Path

import FreeCAD as App
import MeshPart
import Part

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODEL = PROJECT / "gear" / "model" / "damage_test"
OUT = PROJECT / "outputs" / "damage_test"
MODEL.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
params = json.loads((PROJECT / "gear" / "model_parameters.json").read_text())
s, e = params["supplied"], params["estimated"]
doc = App.openDocument(str(PROJECT / "gear" / "model" / "large_gear.FCStd"))
healthy = doc.getObject("LargeGear").Shape.copy()
source_valid = healthy.isValid()
if not source_valid:
    print("Repairing a derived copy of the source CAD topology...", flush=True)
    fixed = healthy.fix(1e-7, 1e-7, 1e-4)
    print(f"Repair returned {fixed}; solid valid={healthy.isValid()}", flush=True)
    if not healthy.isValid():
        print("Source topology details:", healthy.check(False), flush=True)
        raise RuntimeError("Source topology repair did not succeed; original file is untouched.")
V = App.Vector
t = 0.55
radius = e["tooth_inner_radius"]+(s["outside_diameter"]/2-e["tooth_inner_radius"])*t
nominal_angle = math.radians(e["tooth_sweep_degrees"])*(0.65*t+0.35*t*t)
cutters, dents = [], []

for tooth, size in ((5, 0.1), (17, 0.2), (29, 0.3)):
    theta = nominal_angle+2*math.pi*tooth/s["tooth_count"]
    x, y = radius*math.cos(theta), radius*math.sin(theta)
    crossing = healthy.section(Part.makeLine(V(x, y, 30), V(x, y, -1)))
    if not crossing.Vertexes:
        raise RuntimeError(f"Cannot locate crown on tooth {tooth}")
    point = max((vertex.Point for vertex in crossing.Vertexes), key=lambda p: p.z)
    face = min(healthy.Faces, key=lambda f: f.distToShape(Part.Vertex(point))[0])
    normal = face.normalAt(*face.Surface.parameter(point))
    if normal.z < 0:
        normal = -normal
    normal.normalize()
    scale = App.Matrix()
    scale.A11 = size/2
    scale.A22 = size/2
    scale.A33 = size
    cutter = Part.makeSphere(1).transformGeometry(scale)
    cutter.Placement = App.Placement(point, App.Rotation(V(0, 0, 1), normal))
    cutters.append(cutter)
    dents.append({
        "id": f"dent_{size:.1f}mm", "tooth_index": tooth,
        "diameter_mm": size, "depth_mm": size,
        "center_mm": list(point), "normal": list(normal),
        "nominal_laser_crossing_deg": (-tooth*360/s["tooth_count"]) % 360,
        "profile": "Lower half of an ellipsoid: transverse radii diameter/2, normal radius depth; no raised lip.",
    })

print(f"Source CAD: valid={healthy.isValid()}, solids={len(healthy.Solids)}", flush=True)
damaged = healthy
for dent, cutter in zip(dents, cutters):
    print(f"Cutting {dent['id']}: cutter valid={cutter.isValid()}", flush=True)
    damaged = damaged.cut(cutter, 1e-7).removeSplitter()
    print(f"Result: null={damaged.isNull()}, valid={damaged.isValid()}, solids={len(damaged.Solids)}", flush=True)
if damaged.isNull() or not damaged.isValid() or len(damaged.Solids) != 1:
    raise RuntimeError("Dent subtraction did not produce one valid gear solid.")
if not 0 < healthy.Volume-damaged.Volume < 1:
    raise RuntimeError("Unexpected removed volume from submillimeter dents.")
result = App.newDocument("DamagedLargeGear")
feature = result.addObject("Part::Feature", "DamagedGear")
feature.Shape = damaged
feature.Label = "Large gear - 0.1, 0.2, 0.3 mm diameter x depth dents"
feature.addProperty("App::PropertyString", "DentSpecification", "Damage")
feature.DentSpecification = json.dumps(dents)
result.recompute()
result.saveAs(str(MODEL / "damaged_gear.FCStd"))
damaged.exportStep(str(MODEL / "damaged_gear.step"))


def export_obj(shape, filename):
    offset, triangles_count = 1, 0
    with filename.open("w") as stream:
        stream.write("# Units mm; matched-resolution damage comparison mesh.\n")
        for face in shape.Faces:
            mesh = MeshPart.meshFromShape(Shape=face, LinearDeflection=0.0005,
                                         AngularDeflection=0.025, Relative=False)
            vertices, triangles = mesh.Topology
            normals = [face.normalAt(*face.Surface.parameter(v)) for v in vertices]
            if triangles:
                a, b, c = triangles[0]
                cross = (vertices[b]-vertices[a]).cross(vertices[c]-vertices[a])
                if cross.dot(normals[a]) < 0:
                    normals = [-n for n in normals]
            for v in vertices:
                stream.write(f"v {v.x:.9f} {v.y:.9f} {v.z:.9f}\n")
            for n in normals:
                stream.write(f"vn {n.x:.9f} {n.y:.9f} {n.z:.9f}\n")
            for triangle in triangles:
                stream.write("f "+" ".join(f"{offset+i}//{offset+i}" for i in triangle)+"\n")
            offset += len(vertices)
            triangles_count += len(triangles)
    return {"vertices": offset-1, "triangles": triangles_count}


print("Exporting matched-resolution healthy control...", flush=True)
healthy_stats = export_obj(healthy, MODEL / "healthy_control.obj")
print("Exporting dented gear...", flush=True)
damaged_stats = export_obj(damaged, MODEL / "damaged_gear.obj")
metadata = {
    "units": "mm", "dents": dents,
    "base_topology": {"original_valid": source_valid, "comparison_copy_valid": healthy.isValid(),
                      "repair_working_precision_mm": 1e-7, "repair_max_tolerance_mm": 1e-4,
                      "original_file_modified": False},
    "healthy_mesh": "../../gear/model/damage_test/healthy_control.obj",
    "damaged_mesh": "../../gear/model/damage_test/damaged_gear.obj",
    "meshing": {"linear_deflection_mm": 0.0005, "angular_deflection_rad": 0.025,
                "healthy": healthy_stats, "damaged": damaged_stats},
    "removed_volume_mm3": healthy.Volume-damaged.Volume,
    "notes": [
        "Dent dimensions are diameter x depth, confirmed by the user.",
        "Depth is along each original crown normal, not global Z.",
        "Diameters are nominal tangent-plane openings; curved crowns slightly modify the rim.",
        "The rest of the existing CAD geometry is unchanged; both comparison meshes use the same tighter meshing settings.",
        "Boolean-cut faces are retessellated. Tiny mesh/interpolation differences can remain and are not physical defects.",
        "No plastic deformation, raised lip, or material change is included.",
    ],
}
(OUT / "dents.json").write_text(json.dumps(metadata, indent=2)+"\n")
print(json.dumps(metadata, indent=2), flush=True)
App.closeDocument(result.Name)
App.closeDocument(doc.Name)
