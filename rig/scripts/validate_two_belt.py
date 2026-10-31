"""Validate the r06 meshes and CAD drawings; estimate PLA usage per piece."""
import csv
import json
import os
from pathlib import Path

import cairosvg
import numpy as np
import trimesh

from validate_and_estimate import estimate

SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent


def main():
    path=SCRIPTS/'two_belt_r06_report.json'
    report=json.loads(path.read_text())
    selected=set(filter(None,os.environ.get('LISA_VALIDATE_PARTS','').split(',')))
    grams=cost=0
    for name,part in report['parts'].items():
        if selected and name not in selected:
            assert part.get('mesh_validation') and part.get('material_estimate'),name
            continue
        mesh=trimesh.load_mesh(ROOT/part['stl'],process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent,name
        assert mesh.volume>0 and len(mesh.split())==1,name
        assert np.isclose(mesh.bounds[0,2],0,atol=1e-5),(name,mesh.bounds)
        assert np.allclose(mesh.extents,part['bounds_mm'],atol=.025),name
        assert np.all(mesh.extents+6<=180),name
        assert abs(mesh.volume-part['volume_mm3'])/part['volume_mm3']<.005,name
        part['mesh_validation']={'watertight':True,'winding_consistent':True,'connected_bodies':1,
            'positive_volume':True,'within_a1_mini_with_3mm_brim':True,
            'mesh_volume_mm3':round(float(mesh.volume),3),'lowest_z_mm':float(mesh.bounds[0,2])}
        cfg={**report['parameters'],'infill_fraction':part['infill_fraction']}
        part['material_estimate']=estimate(mesh,part,cfg)
        part['material_estimate']['method']=f"CAD volume x estimated shells/{int(part['infill_fraction']*100)}% infill x PLA density x 1.05 waste"
        print(name,part['material_estimate']['estimated_filament_g'],'g',flush=True)
    for drawing in report['drawings']:
        if selected and not any(Path(drawing['svg']).name.startswith(name+'_') for name in selected):
            continue
        svg=ROOT/drawing['svg']
        source=svg.read_text()
        assert all(label in source for label in drawing.get('view_labels',('Top','Front','Side','Isometric'))),svg
        assert 'r06 FIT CHECK' in source,svg
        assert len(drawing['native_dimensions_mm'])>=3
        cairosvg.svg2png(url=str(svg),write_to=str(svg.with_suffix('.png')),output_width=1800,background_color='white')
        cairosvg.svg2pdf(url=str(svg),write_to=str(svg.with_suffix('.pdf')),background_color='white')
        assert svg.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    grams=sum(p['material_estimate']['estimated_filament_g']*p['default_quantity'] for p in report['parts'].values())
    cost=sum(p['material_estimate']['estimated_filament_cost_usd']*p['default_quantity'] for p in report['parts'].values())
    report['assembly_material_estimate']={'estimated_filament_g':round(grams,1),
        'estimated_filament_cost_usd':round(cost,2),'printed_piece_quantity':len(report['parts']),
        'assumed_pla_price_usd_kg':20,'assumed_density_g_cm3':1.24,'waste_fraction':.05,
        'excludes':'Slicer-generated supports, TPU feet, purchased hardware, electricity and labor',
        'planning_uncertainty':'Approximately +/-20%; confirm with the slicer and actual spool price',
        'slicer_estimate':False}
    with (SCRIPTS/'two_belt_print_estimates.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['Part','Quantity','Estimated PLA g','Estimated filament cost USD','Infill percent','Support','Print orientation'])
        for name,part in report['parts'].items():
            e=part['material_estimate']
            writer.writerow([name,part['default_quantity'],e['estimated_filament_g'],e['estimated_filament_cost_usd'],
                             int(part['infill_fraction']*100),part['support'],part['print_orientation']])
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Validated {len(report["parts"])} STLs and native drawings. PLA:',round(grams,1),'g; USD',round(cost,2))


if __name__=='__main__':main()
