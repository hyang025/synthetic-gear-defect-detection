"""Build the approximate large gear with FreeCAD; dimensions are in mm.

From the Mitsuba folder:
    ./tools/freecad freecadcmd gear/build_gear.py

Edit model_parameters.json and rerun to change the supplied or estimated inputs.
"""

import json
import math
from pathlib import Path

import FreeCAD as App
import MeshPart
import Part


ROOT = Path(__file__).resolve().parent
PARAMS = json.loads((ROOT / "model_parameters.json").read_text())
S = PARAMS["supplied"]
E = PARAMS["estimated"]
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)
V = App.Vector

radius = S["outside_diameter"] / 2
hub_radius = E["rear_hub_diameter"] / 2
back_z = E["rear_hub_projection"]
inner_radius = E["tooth_inner_radius"]
front_z = S["total_thickness"] - E["tooth_height"]

# Revolve the disc, conical tooth-root surface, and rear hub together.
profile_rz = [
    (0, 0),
    (hub_radius - E["hub_edge_chamfer"], 0),
    (hub_radius, E["hub_edge_chamfer"]),
    (hub_radius, back_z - E["hub_root_chamfer"]),
    (hub_radius + E["hub_root_chamfer"], back_z),
    (radius - E["rim_back_chamfer"], back_z),
    (radius, back_z + E["rim_back_chamfer"]),
    (radius, E["outer_root_height"]),
    (inner_radius, front_z),
    (0, front_z),
    (0, 0),
]
profile = Part.makePolygon([V(r, 0, z) for r, z in profile_rz])
body = Part.Face(profile).revolve(V(0, 0, 0), V(0, 0, 1), 360)

# A curved loft estimates the visible tooth envelope. It is not an involute
# or conjugate bevel-gear tooth calculation.
pitch = 2 * math.pi / S["tooth_count"]


def loft_tooth(tip_fraction):
    sections = []
    for station in range(9):
        t = station / 8
        r = inner_radius + (radius - inner_radius) * t
        angle = math.radians(E["tooth_sweep_degrees"]) * (0.65*t + 0.35*t*t)
        root_z = front_z + (E["outer_root_height"]-front_z)*t
        root_half, tip_half = pitch*E["tooth_root_pitch_fraction"]/2, pitch*tip_fraction/2
        corners = [(-root_half, root_z-0.2), (root_half, root_z-0.2),
                   (tip_half, root_z+E["tooth_height"]), (-tip_half, root_z+E["tooth_height"])]
        points = [V(r*math.cos(angle+a), r*math.sin(angle+a), z) for a, z in corners]
        sections.append(Part.makePolygon(points+[points[0]]))
    return Part.makeLoft(sections, True, False)


def crown_width(shape, t):
    """Section the crown perpendicular to its local 3D longitudinal tangent."""
    r = inner_radius + (radius-inner_radius)*t
    sweep = math.radians(E["tooth_sweep_degrees"])
    angle = sweep*(0.65*t+0.35*t*t)
    x, y = r*math.cos(angle), r*math.sin(angle)
    hits = shape.section(Part.makeLine(V(x, y, 30), V(x, y, -1)))
    point = max((v.Point for v in hits.Vertexes), key=lambda p: p.z)
    face = min(shape.Faces, key=lambda f: f.distToShape(Part.Vertex(point))[0])
    normal = face.normalAt(*face.Surface.parameter(point))
    normal.normalize()
    dtheta = sweep*(0.65+0.70*t)
    span = radius-inner_radius
    tangent = V(span*math.cos(angle)-r*math.sin(angle)*dtheta,
                span*math.sin(angle)+r*math.cos(angle)*dtheta,
                E["outer_root_height"]-front_z)
    tangent = tangent-normal*tangent.dot(normal)
    tangent.normalize()
    across = tangent.cross(normal)
    across.normalize()
    corners = [point+across*a+normal*b for a, b in ((-4,-4), (4,-4), (4,4), (-4,4))]
    plane = Part.Face(Part.makePolygon(corners+[corners[0]]))
    section = face.section(plane)
    endpoints = [v.Point for v in section.Vertexes]
    if len(endpoints) != 2:
        raise RuntimeError(f'Expected two crown-section endpoints, got {len(endpoints)}')
    width = (endpoints[1]-endpoints[0]).Length
    return {"radial_fraction": t, "width_mm": width, "center_mm": list(point),
            "endpoints_mm": [list(p) for p in endpoints], "normal": list(normal),
            "along_tangent": list(tangent), "across_tangent": list(across),
            "definition": "Endpoint distance on a crown section perpendicular to its local longitudinal tangent"}


measurement = PARAMS.get("measured")
tip_fraction = E.get("tooth_tip_pitch_fraction", 0.24)
if measurement:
    for iteration in range(6):
        tooth = loft_tooth(tip_fraction)
        width = crown_width(tooth, measurement["radial_fraction"])
        error = width["width_mm"]-measurement["tooth_top_width_mm"]
        print(f"Crown fitting: fraction={tip_fraction:.8f}, perpendicular width={width['width_mm']:.6f} mm", flush=True)
        if abs(error) < 1e-5:
            break
        tip_fraction *= measurement["tooth_top_width_mm"]/width["width_mm"]
    else:
        raise RuntimeError('Could not fit the requested crown width.')
else:
    tooth = loft_tooth(tip_fraction)
    width = crown_width(tooth, 0.5)
teeth = []
for index in range(S["tooth_count"]):
    instance = tooth.copy()
    instance.rotate(V(0, 0, 0), V(0, 0, 1), index * 360 / S["tooth_count"])
    teeth.append(instance)
print("Fusing the body and 35 estimated curved teeth...", flush=True)
gear = body.multiFuse(teeth).removeSplitter()

bore_r = S["bore_diameter"] / 2
height = S["total_thickness"]
tools = [Part.makeCylinder(bore_r, height + 2, V(0, 0, -1))]
chamfer = E["bore_edge_chamfer"]
tools.append(Part.makeCone(bore_r, bore_r + chamfer, chamfer, V(0, 0, front_z - chamfer)))
tools.append(Part.makeCone(bore_r + chamfer, bore_r, chamfer, V(0, 0, 0)))
keyway_reach = bore_r + E["keyway_radial_depth"]
tools.append(Part.makeBox(
    keyway_reach, S["keyway_width"], height + 2,
    V(-keyway_reach, -S["keyway_width"] / 2, -1),
))

for index in range(E["front_hole_count"]):
    angle = math.radians(45 + 360 * index / E["front_hole_count"])
    x = E["front_hole_pitch_circle_diameter"] / 2 * math.cos(angle)
    y = E["front_hole_pitch_circle_diameter"] / 2 * math.sin(angle)
    hole_r = E["front_hole_diameter"] / 2
    tools.append(Part.makeCylinder(hole_r, E["front_hole_depth"] + 0.01,
                                   V(x, y, front_z - E["front_hole_depth"])))
    tools.append(Part.makeCone(hole_r, E["front_hole_entry_diameter"] / 2,
                              E["front_hole_entry_depth"],
                              V(x, y, front_z - E["front_hole_entry_depth"])))
gear = gear.cut(Part.makeCompound(tools)).removeSplitter()
valid_before_repair = gear.isValid()
if not valid_before_repair:
    gear.fix(1e-7, 1e-7, 1e-4)
if not gear.isValid() or len(gear.Solids) != 1:
    raise RuntimeError('Revised gear did not produce one valid solid.')
finished_width = crown_width(gear, measurement["radial_fraction"] if measurement else 0.5)
metadata = {"revision": PARAMS.get("revision"), "fitted_tip_pitch_fraction": tip_fraction,
            "target_width_mm": measurement["tooth_top_width_mm"] if measurement else None,
            "crown_section": finished_width, "valid_before_repair": valid_before_repair,
            "valid_final": gear.isValid(), "solids": len(gear.Solids),
            "repair_max_tolerance_mm": 0.0001 if not valid_before_repair else None}
(OUT / "build_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

doc = App.newDocument("LargeGear")
part = doc.addObject("Part::Feature", "LargeGear")
part.Label = "Large gear - nominal 0.7 mm mid-tooth crown"
part.Shape = gear
for group, values in (("Supplied", S), ("Estimated", E)):
    for name, value in values.items():
        property_name = "".join(word.capitalize() for word in name.split("_"))
        kind = "App::PropertyInteger" if isinstance(value, int) else "App::PropertyFloat"
        part.addProperty(kind, property_name, group,
                         "Input recorded from model_parameters.json; edit that file and rebuild.")
        setattr(part, property_name, value)
        part.setEditorMode(property_name, 1)
part.addProperty("App::PropertyStringList", "ModelNotes", "Source")
part.ModelNotes = PARAMS["notes"]
part.addProperty("App::PropertyFloat", "MidToothTopWidth", "Measured")
part.MidToothTopWidth = finished_width["width_mm"]
part.addProperty("App::PropertyFloat", "DerivedTipPitchFraction", "Derived")
part.DerivedTipPitchFraction = tip_fraction
doc.recompute()
doc.saveAs(str(OUT / "large_gear.FCStd"))
gear.exportStep(str(OUT / "large_gear.step"))

# Split vertices at CAD face boundaries, but use each face's surface normals
# within the face. This preserves sharp edges without faceting smooth flanks.
print("Exporting the Mitsuba mesh...", flush=True)
offset = 1
with (OUT / "large_gear.obj").open("w") as stream:
    stream.write("# Approximate large gear. Coordinates are millimeters.\n")
    stream.write("# See ../model_parameters.json for supplied and estimated inputs.\n")
    for face in gear.Faces:
        mesh = MeshPart.meshFromShape(Shape=face, LinearDeflection=0.005,
                                     AngularDeflection=0.08, Relative=False)
        vertices, triangles = mesh.Topology
        normals = [face.normalAt(*face.Surface.parameter(v)) for v in vertices]
        # Align surface normals to the exported triangle winding.
        if triangles:
            a, b, c = triangles[0]
            geometric = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a])
            if geometric.dot(normals[a]) < 0:
                normals = [-n for n in normals]
        for v in vertices:
            stream.write(f"v {v.x:.8f} {v.y:.8f} {v.z:.8f}\n")
        for n in normals:
            stream.write(f"vn {n.x:.8f} {n.y:.8f} {n.z:.8f}\n")
        for triangle in triangles:
            stream.write("f " + " ".join(f"{offset + i}//{offset + i}" for i in triangle) + "\n")
        offset += len(vertices)
print(f"Saved CAD, STEP, and OBJ in {OUT}", flush=True)
App.closeDocument(doc.Name)
