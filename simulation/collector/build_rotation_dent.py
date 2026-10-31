"""Locally refine and geometrically deform a crown of the saved healthy mesh."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import mitsuba as mi

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT/"gear/model/damage_test/healthy_control.obj"
MODEL = PROJECT/"gear/model/collector_dent_0p10_depth_0p025"
REFERENCE = PROJECT/"outputs/collector_rotation_25mm/manifest.json"
DIAMETER, DEPTH, EDGE = 0.1, 0.025, 0.001
TOOTH, TEETH = 7, 35


def main():
    if (MODEL/"damaged_gear.obj").exists():
        raise SystemExit(f"An existing model occupies {MODEL}.")
    vertices, normals, faces = [], [], []
    for line in SOURCE.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "vn":
            normals.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f":
            entries = [item.split("/") for item in fields[1:]]
            if len(entries) != 3 or any(item[0] != item[2] for item in entries):
                raise RuntimeError("The source requires matching triangular position/normal indices.")
            faces.append([int(item[0])-1 for item in entries])
    vertices, normals = np.asarray(vertices), np.asarray(normals)
    faces = np.asarray(faces, dtype=np.int32)
    source_counts = {"vertices": len(vertices), "triangles": len(faces)}
    reference = json.loads(REFERENCE.read_text())["geometry"]
    angle = math.radians(TOOTH*360/TEETH)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    nominal = rotation@np.asarray(reference["patch_m"])
    normal = rotation@np.asarray(reference["normal"])
    mi.set_variant("llvm_ad_rgb")
    scene = mi.load_dict({"type": "scene", "gear": {"type": "obj", "filename": str(SOURCE),
                         "to_world": mi.ScalarTransform4f.scale(0.001)}})
    hit = scene.ray_intersect(mi.Ray3f((nominal+0.001*normal).tolist(), (-normal).tolist()))
    if not bool(hit.is_valid()[0]):
        raise RuntimeError("The selected crown center could not be located.")
    center = np.array([float(hit.p[i][0]) for i in range(3)])*1000
    normal = np.array([float(hit.sh_frame.n[i][0]) for i in range(3)])
    normal /= np.linalg.norm(normal)
    u = rotation@np.asarray(reference["line_tangent"])
    u -= normal*np.dot(u, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    basis = np.column_stack((u, v, normal))
    radius = DIAMETER/2
    support = radius+2*EDGE
    rounds = []
    for iteration in range(40):
        local = (vertices-center)@basis
        tri = local[faces]
        lo, hi = tri.min(axis=1), tri.max(axis=1)
        near = ((lo[:, 0] <= support) & (hi[:, 0] >= -support)
                & (lo[:, 1] <= support) & (hi[:, 1] >= -support)
                & (lo[:, 2] < 0.01) & (hi[:, 2] > -0.01)
                & ((normals[faces].mean(axis=1)@normal) > 0.85))
        candidates = faces[near]
        xyz = vertices[candidates]
        length2 = np.sum((xyz-np.roll(xyz, -1, axis=1))**2, axis=2)
        longest = length2.argmax(axis=1)
        keep = length2.max(axis=1) > EDGE**2
        candidates, longest = candidates[keep], longest[keep]
        if len(candidates) == 0:
            break
        edges = np.sort(np.column_stack((candidates[np.arange(len(candidates)), longest],
                                         candidates[np.arange(len(candidates)), (longest+1) % 3])), axis=1)
        edges = np.unique(edges, axis=0)
        offset = len(vertices)
        midpoint = {tuple(edge): offset+i for i, edge in enumerate(edges)}
        vertices = np.vstack((vertices, vertices[edges].mean(axis=1)))
        # Linear normal interpolation retains the original field during refinement.
        normals = np.vstack((normals, normals[edges].mean(axis=1)))
        refined = []
        for a, b, c in faces:
            nodes = [a, b, c]
            mids = [midpoint.get(tuple(sorted(pair))) for pair in ((a, b), (b, c), (c, a))]
            count = sum(m is not None for m in mids)
            if count == 0:
                refined.append(nodes)
            elif count == 3:
                ab, bc, ca = mids
                refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            else:
                # Rotate to the single split edge, or the two consecutive split edges.
                k = next(i for i in range(3) if mids[i] is not None and
                         (count == 1 or mids[(i+1) % 3] is not None))
                a, b, c = nodes[k:]+nodes[:k]
                ab, bc, ca = mids[k:]+mids[:k]
                if count == 1:
                    refined.extend(((a, ab, c), (ab, b, c)))
                else:
                    refined.extend(((b, bc, ab), (a, ab, c), (ab, bc, c)))
        faces = np.asarray(refined, dtype=np.int32)
        rounds.append({"round": iteration+1, "split_edges": len(edges), "triangles": len(faces)})
        print(f"Refinement {iteration+1}: {len(edges)} edges; {len(faces)} triangles", flush=True)
    else:
        raise RuntimeError("Local mesh refinement reached its iteration limit.")
    local = (vertices-center)@basis
    rho2 = np.sum(local[:, :2]**2, axis=1)/radius**2
    affected = (rho2 < 1) & (abs(local[:, 2]) < 0.01) & ((normals@normal) > 0.85)
    displacement = np.zeros(len(vertices))
    displacement[affected] = -DEPTH*(1-rho2[affected])**2
    deformed = vertices+displacement[:, None]*normal
    output_normals = normals.copy()
    gradient = (4*DEPTH/radius**2*(1-rho2[affected]))[:, None]*(local[affected, 0, None]*u+local[affected, 1, None]*v)
    output_normals[affected] -= (normals[affected]@normal)[:, None]*gradient
    output_normals[affected] /= np.linalg.norm(output_normals[affected], axis=1)[:, None]
    MODEL.mkdir(parents=True, exist_ok=True)
    mesh_path = MODEL/"damaged_gear.obj"
    with mesh_path.open("w") as stream:
        stream.write("# Units mm; geometric quartic dent on tooth 7.\n")
        for point in deformed:
            stream.write("v "+" ".join(f"{value:.9f}" for value in point)+"\n")
        for n in output_normals:
            stream.write("vn "+" ".join(f"{value:.9f}" for value in n)+"\n")
        for triangle in faces+1:
            stream.write("f "+" ".join(f"{i}//{i}" for i in triangle)+"\n")
    np.savez_compressed(MODEL/"deformation.npz", base_vertices_mm=vertices,
                        vertices_mm=deformed, faces=faces, displacement_mm=displacement,
                        center_mm=center, basis=basis)
    metadata = {
        "diameter_mm": DIAMETER, "depth_mm": DEPTH, "tooth_index": TOOTH,
        "nominal_laser_crossing_deg": (-TOOTH*360/TEETH) % 360,
        "center_mm": center.tolist(), "normal": normal.tolist(),
        "line_tangent": u.tolist(), "line_across": v.tolist(),
        "profile": "Normal displacement = -depth * (1 - r^2 / radius^2)^2 inside the circular rim",
        "local_maximum_edge_mm": EDGE, "maximum_vertex_depth_mm": float(-displacement.min()),
        "deformed_vertices": int(affected.sum()), "source_counts": source_counts,
        "result_counts": {"vertices": len(vertices), "triangles": len(faces)},
        "refinement": "Conforming shared-edge subdivision of the original triangles",
        "normal_model": "Original interpolated normals transformed by the deformation Jacobian",
        "source_mesh": str(SOURCE.relative_to(PROJECT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "refinement_rounds": rounds,
    }
    (MODEL/"dent.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "refinement_rounds"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
