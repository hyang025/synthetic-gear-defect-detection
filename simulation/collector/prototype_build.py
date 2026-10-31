"""Build three additional gears with spherical crown, edge, and flank cuts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import FreeCAD as App
import MeshPart
import Part

PROJECT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT / "collector/prototype.json"
V = App.Vector


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(vector):
    result = V(vector)
    result.normalize()
    return result


def crown_section(shape, parameters, tooth, fraction):
    supplied, estimated = parameters["supplied"], parameters["estimated"]
    span = supplied["outside_diameter"] / 2 - estimated["tooth_inner_radius"]
    radius = estimated["tooth_inner_radius"] + fraction * span
    sweep = math.radians(estimated["tooth_sweep_degrees"])
    angle = tooth * 2 * math.pi / supplied["tooth_count"] + sweep * (0.65*fraction + 0.35*fraction**2)
    x, y = radius*math.cos(angle), radius*math.sin(angle)
    intersection = shape.section(Part.makeLine(V(x, y, 30), V(x, y, -1)))
    point = max((vertex.Point for vertex in intersection.Vertexes), key=lambda p: p.z)
    face = min(shape.Faces, key=lambda f: f.distToShape(Part.Vertex(point))[0])
    normal = normalized(face.normalAt(*face.Surface.parameter(point)))
    derivative = sweep * (0.65 + 0.70*fraction)
    tangent = V(span*math.cos(angle)-radius*math.sin(angle)*derivative,
                span*math.sin(angle)+radius*math.cos(angle)*derivative,
                estimated["outer_root_height"]-(supplied["total_thickness"]-estimated["tooth_height"]))
    tangent = normalized(tangent - normal*tangent.dot(normal))
    across = normalized(normal.cross(tangent))
    corners = [point+across*a+normal*b for a, b in ((-4, -4), (4, -4), (4, 4), (-4, 4))]
    section = face.section(Part.Face(Part.makePolygon(corners+[corners[0]])))
    endpoints = sorted((vertex.Point for vertex in section.Vertexes), key=lambda p: (p-point).dot(across))
    if len(endpoints) != 2:
        raise RuntimeError("The crown section requires two edge endpoints.")
    return point, normal, tangent, across, endpoints


def surface_point(shape, parameters, defect):
    point, normal, tangent, across, edges = crown_section(
        shape, parameters, defect["tooth"], defect["radial_fraction"])
    if defect["kind"] == "top_edge":
        point = edges[0 if defect["side"] < 0 else 1]
    elif defect["kind"] == "flank":
        edge = edges[0 if defect["side"] < 0 else 1]
        candidate = edge + across*(defect["side"]*0.03) - normal*0.06
        candidates = []
        for face in shape.Faces:
            distance, pairs, _ = face.distToShape(Part.Vertex(candidate))
            if distance > 0.15:
                continue
            projected = pairs[0][0]
            n = normalized(face.normalAt(*face.Surface.parameter(projected)))
            if abs(n.dot(normal)) < 0.85 and n.dot(across)*defect["side"] > 0:
                candidates.append((distance, projected, n))
        if not candidates:
            raise RuntimeError(f"Could not locate the requested flank for {defect['id']}.")
        _, point, normal = min(candidates, key=lambda entry: entry[0])
    return point, normal, tangent, across


def export_obj(shape, destination):
    offset, triangles_total = 1, 0
    with destination.open("w") as stream:
        stream.write("# Units mm; spherical geometric damage.\n")
        for face in shape.Faces:
            mesh = MeshPart.meshFromShape(Shape=face, LinearDeflection=0.0005,
                                         AngularDeflection=0.025, Relative=False)
            vertices, triangles = mesh.Topology
            normals = [face.normalAt(*face.Surface.parameter(p)) for p in vertices]
            if triangles:
                a, b, c = triangles[0]
                if (vertices[b]-vertices[a]).cross(vertices[c]-vertices[a]).dot(normals[a]) < 0:
                    normals = [-n for n in normals]
            for p in vertices:
                stream.write(f"v {p.x:.9f} {p.y:.9f} {p.z:.9f}\n")
            for n in normals:
                stream.write(f"vn {n.x:.9f} {n.y:.9f} {n.z:.9f}\n")
            for triangle in triangles:
                stream.write("f "+" ".join(f"{offset+i}//{offset+i}" for i in triangle)+"\n")
            offset += len(vertices)
            triangles_total += len(triangles)
    return {"vertices": offset-1, "triangles": triangles_total}


def main():
    config = json.loads(CONFIG.read_text())
    parameters = json.loads((PROJECT/"gear/model_parameters.json").read_text())
    source = PROJECT/config["cad_source"]
    output = PROJECT/config["model_output"]
    if output.exists():
        raise RuntimeError(f"An existing model set occupies {output}.")
    output.mkdir(parents=True)
    doc = App.openDocument(str(source))
    original = doc.getObject("LargeGear").Shape.copy()
    all_records = []
    for gear in config["gears"]:
        print(f"Building {gear['id']}...", flush=True)
        cutters, records = [], []
        for spec in gear["defects"]:
            point, normal, tangent, across = surface_point(original, parameters, spec)
            lateral_radius, depth = spec["diameter_mm"]/2, spec["depth_mm"]
            sphere_radius = (lateral_radius**2+depth**2)/(2*depth)
            center = point + normal*(sphere_radius-depth)
            cutters.append(Part.makeSphere(sphere_radius, center))
            records.append({**spec, "surface_point_mm": list(point), "surface_normal": list(normal),
                            "along_tangent": list(tangent), "across_tangent": list(across),
                            "sphere_center_mm": list(center), "sphere_radius_mm": sphere_radius,
                            "footprint_definition": "Nominal spherical-cap diameter on the local tangent plane; tooth boundaries crop the cap"})
        damaged = original.copy()
        for record, cutter in zip(records, cutters):
            damaged = damaged.cut(cutter, 1e-7).removeSplitter()
            probe = V(*record["surface_point_mm"])-V(*record["surface_normal"])*record["depth_mm"]*0.5
            if damaged.isInside(probe, 1e-7, True):
                raise RuntimeError(f"The spherical subtraction missed {record['id']}.")
        if not damaged.isValid():
            damaged.fix(1e-7, 1e-7, 1e-4)
        if not damaged.isValid() or len(damaged.Solids) != 1:
            raise RuntimeError(f"{gear['id']} requires one valid solid.")
        folder = output/gear["id"]
        folder.mkdir()
        mesh_path = folder/"damaged_gear.obj"
        counts = export_obj(damaged, mesh_path)
        damaged.exportStep(str(folder/"damaged_gear.step"))
        saved = App.newDocument(gear["id"])
        part = saved.addObject("Part::Feature", "DamagedGear")
        part.Shape = damaged
        saved.recompute()
        saved.saveAs(str(folder/"damaged_gear.FCStd"))
        App.closeDocument(saved.Name)
        record = {"id": gear["id"], "defects": records, "mesh": str(mesh_path.relative_to(PROJECT)),
                  "mesh_sha256": digest(mesh_path), "mesh_counts": counts,
                  "valid_solid": True, "removed_volume_mm3": original.Volume-damaged.Volume}
        (folder/"defects.json").write_text(json.dumps(record, indent=2)+"\n")
        all_records.append(record)
        print(f"Saved {gear['id']}: {counts}", flush=True)
    App.closeDocument(doc.Name)
    metadata = {"gears": all_records, "source_cad": config["cad_source"], "source_sha256": digest(source),
                "config_sha256": digest(CONFIG), "builder_sha256": digest(Path(__file__))}
    (output/"models.json").write_text(json.dumps(metadata, indent=2)+"\n")


def match_healthy_mesh():
    config = json.loads(CONFIG.read_text())
    output = PROJECT/config["model_output"]
    metadata_path = output/"models.json"
    metadata = json.loads(metadata_path.read_text())
    if (PROJECT/config["output"]/"manifest.json").exists():
        raise RuntimeError("Mesh refinement must precede prototype captures.")
    for record in metadata["gears"]:
        folder = output/record["id"]
        doc = App.openDocument(str(folder/"damaged_gear.FCStd"))
        mesh_path = folder/"damaged_gear.obj"
        record["mesh_counts"] = export_obj(doc.getObject("DamagedGear").Shape, mesh_path)
        record["mesh_sha256"] = digest(mesh_path)
        record["meshing"] = {"linear_deflection_mm": 0.0005, "angular_deflection_rad": 0.025}
        (folder/"defects.json").write_text(json.dumps(record, indent=2)+"\n")
        App.closeDocument(doc.Name)
        print(f"Matched healthy tessellation settings for {record['id']}", flush=True)
    metadata["builder_sha256"] = digest(Path(__file__))
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")


def inspect_cut_locations():
    config = json.loads(CONFIG.read_text())
    metadata = json.loads((PROJECT/config["model_output"]/"models.json").read_text())
    doc = App.openDocument(str(PROJECT/config["cad_source"]))
    shape = doc.getObject("LargeGear").Shape
    for gear in metadata["gears"]:
        damaged_doc = App.openDocument(str(PROJECT/config["model_output"]/gear["id"]/"damaged_gear.FCStd"))
        damaged = damaged_doc.getObject("DamagedGear").Shape
        for defect in gear["defects"]:
            point, normal = V(*defect["surface_point_mm"]), V(*defect["surface_normal"])
            print(defect["id"], "inward inside:", [shape.isInside(point-normal*h, 1e-7, True)
                  for h in (0.001, defect["depth_mm"]*0.5, defect["depth_mm"])], flush=True)
            print(defect["id"], "damaged inside:", [damaged.isInside(point-normal*h, 1e-7, True)
                  for h in (0.001, defect["depth_mm"]*0.5, defect["depth_mm"]*0.99)], flush=True)
        App.closeDocument(damaged_doc.Name)
    App.closeDocument(doc.Name)


def repair_cut_locations():
    config = json.loads(CONFIG.read_text())
    output = PROJECT/config["model_output"]
    metadata_path = output/"models.json"
    metadata = json.loads(metadata_path.read_text())
    captures_path = PROJECT/config["output"]/"manifest.json"
    captures = json.loads(captures_path.read_text()) if captures_path.exists() else {"frames": []}
    source_doc = App.openDocument(str(PROJECT/config["cad_source"]))
    original = source_doc.getObject("LargeGear").Shape
    parameters = json.loads((PROJECT/"gear/model_parameters.json").read_text())
    for gear in metadata["gears"]:
        folder = output/gear["id"]
        doc = App.openDocument(str(folder/"damaged_gear.FCStd"))
        part = doc.getObject("DamagedGear")
        damaged = part.Shape.copy()
        changed = False
        for defect in gear["defects"]:
            probe = V(*defect["surface_point_mm"])-V(*defect["surface_normal"])*defect["depth_mm"]*0.5
            if not damaged.isInside(probe, 1e-7, True):
                continue
            if any(row["case"] == gear["id"] for row in captures["frames"]):
                raise RuntimeError("Geometry corrections must precede damaged captures.")
            sphere = Part.makeSphere(defect["sphere_radius_mm"], V(*defect["sphere_center_mm"]))
            repaired = damaged.cut(sphere, 1e-7).removeSplitter()
            print(defect["id"], "correction status", repaired.isInside(probe, 1e-7, True), repaired.isValid(), len(repaired.Solids), flush=True)
            if repaired.isInside(probe, 1e-7, True):
                requested_fraction = defect["radial_fraction"]
                for offset in (0.001, 0.005, 0.01):
                    corrected = {**defect, "radial_fraction": requested_fraction+offset}
                    point, normal, tangent, across = surface_point(original, parameters, corrected)
                    center = point+normal*(defect["sphere_radius_mm"]-defect["depth_mm"])
                    sphere = Part.makeSphere(defect["sphere_radius_mm"], center)
                    probe = point-normal*defect["depth_mm"]*0.5
                    repaired = damaged.cut(sphere, 1e-7).removeSplitter()
                    print(defect["id"], "radial adjustment", offset, repaired.isInside(probe, 1e-7, True), repaired.isValid(), flush=True)
                    if not repaired.isInside(probe, 1e-7, True) and repaired.isValid():
                        defect.update(radial_fraction=corrected["radial_fraction"], requested_radial_fraction=requested_fraction,
                                      surface_point_mm=list(point), surface_normal=list(normal), along_tangent=list(tangent),
                                      across_tangent=list(across), sphere_center_mm=list(center))
                        break
            if repaired.isInside(probe, 1e-7, True) or not repaired.isValid() or len(repaired.Solids) != 1:
                raise RuntimeError(f"Correction failed for {defect['id']}.")
            damaged = repaired
            changed = True
            print(f"Corrected spherical subtraction for {defect['id']}", flush=True)
        if changed:
            part.Shape = damaged
            doc.recompute()
            doc.saveAs(str(folder/"damaged_gear.FCStd"))
            damaged.exportStep(str(folder/"damaged_gear.step"))
            mesh_path = folder/"damaged_gear.obj"
            gear["mesh_counts"] = export_obj(damaged, mesh_path)
            gear["mesh_sha256"] = digest(mesh_path)
            gear["removed_volume_mm3"] = original.Volume-damaged.Volume
            gear["cut_verification"] = "Mid-depth probes lie in the removed spherical cavities"
            (folder/"defects.json").write_text(json.dumps(gear, indent=2)+"\n")
        App.closeDocument(doc.Name)
    App.closeDocument(source_doc.Name)
    metadata["builder_sha256"] = digest(Path(__file__))
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")


def rebuild_gear_one():
    config = json.loads(CONFIG.read_text())
    output = PROJECT/config["model_output"]
    metadata_path = output/"models.json"
    metadata = json.loads(metadata_path.read_text())
    parameters = json.loads((PROJECT/"gear/model_parameters.json").read_text())
    source_doc = App.openDocument(str(PROJECT/config["cad_source"]))
    original = source_doc.getObject("LargeGear").Shape
    damaged, records = original.copy(), []
    for spec in config["gears"][0]["defects"]:
        point, normal, tangent, across = surface_point(original, parameters, spec)
        radius = ((spec["diameter_mm"]/2)**2+spec["depth_mm"]**2)/(2*spec["depth_mm"])
        center = point+normal*(radius-spec["depth_mm"])
        damaged = damaged.cut(Part.makeSphere(radius, center), 1e-7).removeSplitter()
        probe = point-normal*spec["depth_mm"]*0.5
        print(spec["id"], "removed mid-depth:", not damaged.isInside(probe, 1e-7, True), flush=True)
        if damaged.isInside(probe, 1e-7, True):
            raise RuntimeError(f"The rebuilt cavity missed {spec['id']}.")
        records.append({**spec, "surface_point_mm": list(point), "surface_normal": list(normal),
                        "along_tangent": list(tangent), "across_tangent": list(across),
                        "sphere_center_mm": list(center), "sphere_radius_mm": radius,
                        "footprint_definition": "Nominal spherical-cap diameter on the local tangent plane; tooth boundaries crop the cap"})
    if not damaged.isValid() or len(damaged.Solids) != 1:
        raise RuntimeError("The rebuilt gear requires one valid solid.")
    folder = output/"gear_1"
    saved = App.openDocument(str(folder/"damaged_gear.FCStd"))
    saved.getObject("DamagedGear").Shape = damaged
    saved.recompute()
    saved.saveAs(str(folder/"damaged_gear.FCStd"))
    damaged.exportStep(str(folder/"damaged_gear.step"))
    mesh_path = folder/"damaged_gear.obj"
    metadata["gears"][0].update(defects=records, mesh_counts=export_obj(damaged, mesh_path),
                                mesh_sha256=digest(mesh_path), removed_volume_mm3=original.Volume-damaged.Volume,
                                cut_verification="All five mid-depth probes lie inside their removed cavities")
    (folder/"defects.json").write_text(json.dumps(metadata["gears"][0], indent=2)+"\n")
    metadata["config_sha256"], metadata["builder_sha256"] = digest(CONFIG), digest(Path(__file__))
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    App.closeDocument(saved.Name)
    App.closeDocument(source_doc.Name)


if __name__ == "__main__" or (len(sys.argv) > 1 and Path(sys.argv[1]).resolve() == Path(__file__).resolve()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--match-healthy-mesh", action="store_true")
    parser.add_argument("--inspect-cuts", action="store_true")
    parser.add_argument("--repair-cuts", action="store_true")
    parser.add_argument("--rebuild-gear-one", action="store_true")
    args, _ = parser.parse_known_args()
    CONFIG = args.config.resolve()
    if args.match_healthy_mesh:
        match_healthy_mesh()
    elif args.inspect_cuts:
        inspect_cut_locations()
    elif args.repair_cuts:
        repair_cut_locations()
    elif args.rebuild_gear_one:
        rebuild_gear_one()
    else:
        main()
