"""Check exported meshes/drawings and estimate shells plus 40% infill.

This is a geometric planning estimate, not a slicer/toolpath prediction.
"""
import json
from pathlib import Path

import cairosvg
import numpy as np
from scipy import ndimage
import trimesh

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT=SCRIPT_DIR.parent


def estimate(mesh,part,config):
    # Finite-grid approximation of horizontal perimeter walls and vertical skins.
    pitch=.6
    grid=mesh.voxelized(pitch).fill()
    filled=np.pad(grid.matrix.astype(bool),8)
    shell=np.zeros_like(filled)
    for z in range(filled.shape[2]):
        cross_section=filled[:,:,z]
        if cross_section.any():
            distance=ndimage.distance_transform_edt(cross_section)*pitch-pitch/2
            shell[:,:,z]=cross_section & (distance<=config['wall_thickness_mm'])
    skin_layers=round(config['top_bottom_shell_mm']/pitch)
    internal=ndimage.binary_erosion(filled,structure=np.ones((1,1,2*skin_layers+1),bool))
    shell |= filled & ~internal
    fraction=shell.sum()/filled.sum()
    effective_fraction=fraction+(1-fraction)*config['infill_fraction']
    solid_mass=part['volume_mm3']/1000*config['material_density_g_cm3_assumed']
    mass=solid_mass*effective_fraction*(1+config['waste_allowance_fraction'])
    return {
        'estimated_filament_g':round(mass,1),
        'estimated_filament_cost_usd':round(mass*config['filament_price_usd_kg_assumed']/1000,2),
        'solid_mass_g_without_waste':round(solid_mass,1),
        'estimated_shell_fraction':round(float(fraction),4),
        'voxel_pitch_mm':pitch,
        'method':'CAD volume × (estimated shell fraction + 40% of interior) × assumed density × 1.05 waste',
        'slicer_estimate':False,
    }


def main():
    report=json.loads((SCRIPT_DIR/'build_report.json').read_text())
    cfg=report['parameters']
    total_mass,total_cost=0.0,0.0
    for name,part in report['parts'].items():
        mesh=trimesh.load_mesh(ROOT/part['stl'],process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent,name
        assert mesh.volume>0 and len(mesh.split())==1,name
        assert np.isclose(mesh.bounds[0,2],0,atol=1e-5),(name,mesh.bounds)
        assert np.allclose(mesh.extents,part['bounds_mm'],atol=.02),name
        assert np.all(mesh.extents<=180),name
        assert abs(mesh.volume-part['volume_mm3'])/part['volume_mm3']<.005,name
        part['mesh_validation']={
            'watertight':True,'winding_consistent':True,'connected_bodies':1,
            'positive_volume':True,'within_a1_mini_envelope':True,
            'mesh_volume_mm3':round(float(mesh.volume),3),
            'bounds_mm':mesh.extents.tolist(),'lowest_z_mm':float(mesh.bounds[0,2]),
            'encoder_mounting_pattern_user_reported_exact':name.startswith('encoder_holder'),
            'motor_corrected_hole_pattern_fit_verified':False,
            'housing_seat_fit_verified':False,
        }
        part['material_estimate']=estimate(mesh,part,cfg)
        total_mass+=part['material_estimate']['estimated_filament_g']*part['default_quantity']
        total_cost+=part['material_estimate']['estimated_filament_cost_usd']*part['default_quantity']
        print(name,part['material_estimate']['estimated_filament_g'],'g',flush=True)
    for drawing in report['drawings']:
        svg=ROOT/drawing['svg']
        text=svg.read_text()
        assert all(s in text for s in drawing.get('view_labels',('Top','Front','Side','Isometric'))),svg
        assert cfg['revision'].split('_')[0]+' PHOTO FIT' in text,svg
        cairosvg.svg2png(url=str(svg),write_to=str(svg.with_suffix('.png')),output_width=1800,
                        background_color='white')
        cairosvg.svg2pdf(url=str(svg),write_to=str(svg.with_suffix('.pdf')),background_color='white')
        assert svg.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    report['assembly_material_estimate']={
        'estimated_filament_g':round(total_mass,1),
        'estimated_filament_cost_usd':round(total_cost,2),
        'includes':'1 shortened adjustable motor holder, 1 motor slide base, 1 fixed encoder holder; no spacers',
        'excludes':'slicer-generated supports, future TPU pads, hardware, electricity and labor',
        'assumed_pla_price_usd_kg':cfg['filament_price_usd_kg_assumed'],
        'assumed_density_g_cm3':cfg['material_density_g_cm3_assumed'],
        'waste_fraction':cfg['waste_allowance_fraction'],
        'planning_uncertainty':'Allow roughly 20% variation; slice the actual files for a toolpath estimate.',
    }
    (SCRIPT_DIR/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['Part,Default quantity,Estimated PLA g,Estimated filament cost USD,Solid volume mm3']
    for name,part in report['parts'].items():
        e=part['material_estimate']
        lines.append(f"{name},{part['default_quantity']},{e['estimated_filament_g']},{e['estimated_filament_cost_usd']},{part['volume_mm3']:.1f}")
    (SCRIPT_DIR/'print_estimates.csv').write_text('\n'.join(lines)+'\n')
    # Keep the requested per-part estimates visible in the ordinary part notes.
    for folder in (ROOT,ROOT/'motor holder',ROOT/'encoder holder'):
        selected=[(name,part) for name,part in report['parts'].items()
                  if folder==ROOT or str(Path(part['stl']).parts[0])==folder.name]
        subtotal_mass=sum(part['material_estimate']['estimated_filament_g']*part['default_quantity']
                          for _,part in selected)
        subtotal_cost=sum(part['material_estimate']['estimated_filament_cost_usd']*part['default_quantity']
                          for _,part in selected)
        text=['<!-- generated-estimates-start -->','',
              '## Latest material estimate','',
              '| Part | Default quantity | PLA estimate | Filament cost |',
              '| --- | ---: | ---: | ---: |']
        for name,part in selected:
            e=part['material_estimate']
            text.append(f"| {name} | {part['default_quantity']} | {e['estimated_filament_g']:.1f} g | USD {e['estimated_filament_cost_usd']:.2f} |")
        text.extend(['',f'Default subtotal: **{subtotal_mass:.1f} g / USD {subtotal_cost:.2f}**.',
                     '', 'Assumed PLA USD 20/kg, density 1.24 g/cm³, and 5% waste. Geometric',
                     'shell/40% infill estimate; allow roughly 20% variation.',
                     'TPU pads and hardware are excluded from the subtotal.',
                     '', '<!-- generated-estimates-end -->'])
        file=folder/'readme.md'
        original=file.read_text()
        start=original.find('<!-- generated-estimates-start -->')
        if start>=0:
            end=original.index('<!-- generated-estimates-end -->',start)+len('<!-- generated-estimates-end -->')
            original=original[:start]+original[end:]
        file.write_text(original.rstrip()+'\n\n'+'\n'.join(text)+'\n')
    print('All STL and native drawing checks passed. Default assembly:',round(total_mass,1),'g;',round(total_cost,2),'USD')


if __name__=='__main__':main()
