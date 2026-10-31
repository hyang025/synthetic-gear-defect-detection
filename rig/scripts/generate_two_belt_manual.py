"""Lay out the r06 manual from authoritative CAD drawings and CAD renders."""
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[3]/'.cache/matplotlib'))
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import generate_manual as layout

SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent
REPORT=json.loads((SCRIPTS/'two_belt_r06_report.json').read_text())
ASSEMBLY=json.loads((SCRIPTS/'two_belt_r06_assembly_report.json').read_text())
BOM=json.loads((SCRIPTS/'two_belt_parts.json').read_text())
OUT=ROOT/'manual.pdf'
PREVIEW=ROOT.parents[1]/'.cache/lisa-manual-r06'
PREVIEW.mkdir(parents=True,exist_ok=True)


def main():
    count=0
    actual={p['assembly_object'] for p in REPORT['parts'].values()}
    assert actual=={p['cad_name'] for p in BOM['printed']}
    with (SCRIPTS/'two_belt_parts.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['Category','Component','Quantity','Specification','Source / status','Supplier','SKU','URL','Unit price USD'])
        for category in ('core','reused','fasteners','printed'):
            for item in BOM[category]:
                writer.writerow([category,item['name'],item['quantity'],item['specification'],item['status'],
                    item.get('supplier',''),item.get('sku',''),item.get('url',''),item.get('unit_price_usd','')])
    with PdfPages(OUT,metadata={'Title':'Lisa r06 two-belt printable fit-test set',
                               'Author':'3dprint_generator','Subject':'CAD drawings, print orientation, hardware, assembly and estimates'}) as pdf:
        def save(fig):
            nonlocal count
            count+=1
            fig.text(.045,.035,'LISA / R06 TWO-BELT FIT SET / dimensions in mm',fontsize=8,color=layout.MUTED)
            fig.text(.955,.035,f'{count:02d}',fontsize=9,color=layout.MUTED,ha='right')
            fig.canvas.draw(); renderer=fig.canvas.get_renderer()
            for text in fig.texts:
                box=text.get_window_extent(renderer)
                assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.x1 and box.y1<=fig.bbox.y1,(count,text.get_text())
            for axes in fig.axes:
                for table in axes.tables:
                    for cell in table.get_celld().values():
                        box=cell.get_text().get_window_extent(renderer); bound=cell.get_window_extent(renderer)
                        assert box.x0>=bound.x0 and box.x1<=bound.x1 and box.y0>=bound.y0 and box.y1<=bound.y1,(count,cell.get_text().get_text())
            pdf.savefig(fig)
            fig.savefig(PREVIEW/f'page_{count:02d}.png',dpi=140,facecolor='white')
            plt.close(fig)

        for key in ('assembly','views','detail','exploded'):
            fig=plt.figure(figsize=(11.69,8.27),facecolor='white')
            layout.picture(fig,ROOT/ASSEMBLY['renders'][key],[.01,.057,.98,.94])
            save(fig)

        fig=layout.page('Assemble the bearing sandwich on the bench',
            'Two printed frame pieces  •  Fit both closed belts before lowering the upper bearing bridge')
        layout.picture(fig,ROOT/'assembly/output/render/two_belt_exploded_r06_raw.png',[.02,.565,.54,.31])
        fig.text(.045,.555,'Gusseted body stays fixed; upper bridge lowers along the shaft.',fontsize=9,color=layout.ACCENT)
        layout.table(fig,[['Frame body','1'],['Upper bridge','1'],['Bearing blocks','2'],
            ['M4 x 16 bearing screws','8'],['M4 x 25 bridge screws','4'],['Plain M4 nuts','12'],
            ['6 mm shaft','1'],['Shaft collars','2'],['80T pulleys','2'],['Closed GT2 belts','2']],
            ['Frame hardware / components','Quantity'],[.59,.55,.365,.32],[.79,.21],size=8.5)
        steps=[
            '1. Remove every support under the lower deck and inside the round bores. Before board mounting, slide the lower bearing in through an open X side and bolt it underneath the integral deck with four M4 x 16 screws and four nuts. Driver access is from below.',
            '2. On the bench, attach the upper bearing above its bridge with four M4 x 16 screws and four nuts. Leave the bearing fasteners just loose enough to align with the shaft; omit extra bearing washers.',
            '3. With the bridge off, lower the bare shaft and both 80T pulleys into the lower bearing. Loop both closed belts around their pulleys now. Leave the upper collar, gear and adapter off so the upper bearing can pass over the bare shaft end.',
            '4. Lower the bridge and upper bearing vertically over the shaft. Guide the front pin into the round socket and the rear pin into the slot. Seat both flanges on their shoulders. Do not use bolts to pull a misaligned bridge down.',
            '5. Insert four M4 x 25 screws from the outside front and rear. Hold the four M4 nuts with a 7 mm spanner through the open X sides. Start all four screws, then tighten evenly; align and secure the bearing mounts while checking free shaft rotation.',
            '6. Fit the lower collar from below and the upper collar from above; set light contact against the bearing inner rings without preload. Fit the gear adapter, then mount the frame to plywood. Align pulleys and tension both belts using the device slides.'
        ]
        y=.515
        for step in steps:y=layout.paragraph(fig,.045,y,step,size=9.5,leading=.022)-.014
        layout.paragraph(fig,.045,y-.005,
            'No adhesive. Locator fit assumes 0.25 mm radial clearance and 0.25 mm end clearance. CAD tool checks use a 5 mm driver stem and a 3 mm thick, 14 mm OD wrench head; check your tools and printed fit before tightening.',size=9,leading=.022)
        save(fig)

        fig=layout.page('Purchased parts: goBILDA first','R06 fit set  •  Individual quantities  •  Blue part numbers open the supplier pages')
        rows=[[i['name'],str(i['quantity']),i['specification'],i['sku'],f"${i['unit_price_usd']:.2f}",f"${i['quantity']*i['unit_price_usd']:.2f}"] for i in BOM['core']]
        tab=layout.table(fig,rows,['Component','Quantity','Specification','goBILDA part number','Each','Subtotal'],[.045,.555,.91,.29],[.19,.08,.33,.22,.09,.09],size=10)
        for row,item in enumerate(BOM['core'],1):
            text=tab[row,3].get_text(); text.set_url(item['url']); text.set_color(layout.ACCENT)
        subtotal=sum(i['quantity']*i['unit_price_usd'] for i in BOM['core'])
        fig.text(.045,.505,f'Shaft, bearing blocks and collars: USD {subtotal:.2f}',fontsize=17,color=layout.ACCENT,weight='bold')
        layout.paragraph(fig,.045,.46,'Each bearing block includes its bearing. Each collar includes its M4 pinch bolt. The blocks and collars use manufacturer STEP; the steel shaft also uses the supplier CAD in this assembly. The quoted subtotal excludes other fasteners, belt, filament, shipping and tax. Prices checked '+BOM['prices_checked']+'.',size=11,leading=.028)
        layout.table(fig,[[i['name'],str(i['quantity']),('Both belts on hand (user report)' if i['name']=='Timing belt' else 'Reuse existing component')] for i in BOM['reused']],
            ['Component','Quantity','Source'],[.045,.125,.91,.215],[.26,.09,.65],size=9)
        layout.paragraph(fig,.045,.085,'Required belts: closed-loop 250 mm pitch length, 6 mm width, 2 mm pitch, 125 teeth. User reports components are on hand; check actual specifications and fit.',size=9,leading=.022)
        save(fig)

        fig=layout.page('Complete screw, nut and washer list','Quantities cover the whole r06 assembly. Use stock already available where it matches the listed specification.')
        rows=[[i['name'],str(i['quantity']),i['specification'],i.get('sku','Existing / generic')] for i in BOM['fasteners']]
        tab=layout.table(fig,rows,['Location / component','Quantity','Specification','goBILDA part number'],[.045,.275,.91,.57],[.255,.08,.445,.22],size=9)
        for row,item in enumerate(BOM['fasteners'],1):
            if item.get('url'):
                text=tab[row,3].get_text(); text.set_url(item['url']); text.set_color(layout.ACCENT)
        y=layout.paragraph(fig,.045,.245,'Both slides use M4 x 12 screws with 1 mm washers on 5 mm feet, above 6 mm bases. Captive nuts have full nominal 3.2 mm engagement and the screw tips reach the plywood surface. The selected goBILDA washer is confirmed 1 mm thick in supplier CAD.',size=10,leading=.024)
        y=layout.paragraph(fig,.045,y-.012,'Use plain M4 nuts at the bearing plates and omit extra washers in that stack. Longer upper bearing bolts can contact the upper pulley. The 20 M4 nuts cover 8 bearings, 4 bridge joints and 8 slide clamps. Twelve #6 x 1/2 inch wood screws secure the two bases and the frame feet.',size=10,leading=.024)
        layout.paragraph(fig,.045,y-.012,'Motor M3 x 6 and encoder M3 x 8 give about 2.4 mm engagement with 0.6 mm washers; verify actual thread depths. Encoder M3 x 10 plus a rigid 2 mm spacer is the alternative. Gear cap: one M3 x 16 screw and one M3 nut in the retained hex pocket.',size=10,leading=.024)
        save(fig)

        fig=layout.page('Print settings and PLA estimate','Bambu Lab A1 mini  •  0.4 mm stainless steel nozzle  •  All 8 STLs exported in printing orientation')
        y=layout.paragraph(fig,.045,.835,'PLA: 0.20 mm layer height; 6 walls (about 2.5–2.7 mm); 6 top and bottom layers (1.2 mm). Use 40% gyroid for structural parts and 100% infill for both gear-adapter pieces. An optional 3 mm brim fits within the A1 mini envelope.',size=11,leading=.027)
        layout.paragraph(fig,.045,y-.012,'Support the underside of the integral lower bearing deck and the eight flat holder washer lands. Add local supports in horizontal M4/M3 bores and at the clamp nut-pocket roof as needed. Remove all support before assembly. Part pages show exact orientations.',size=10,leading=.025)
        rows=[]
        for name,part in REPORT['parts'].items():
            e=part['material_estimate']
            rows.append([name.removesuffix('_r06').replace('_',' '),str(part['default_quantity']),f"{e['estimated_filament_g']:.1f} g",f"${e['estimated_filament_cost_usd']:.2f}",str(int(part['infill_fraction']*100))+'%'])
        layout.table(fig,rows,['Printed component','Quantity','PLA','Filament cost','Infill'],[.045,.22,.91,.44],[.47,.085,.145,.16,.14],size=9)
        total=REPORT['assembly_material_estimate']
        fig.text(.045,.175,f"Complete print set: approximately {total['estimated_filament_g']:.1f} g PLA / USD {total['estimated_filament_cost_usd']:.2f}",fontsize=15,color=layout.ACCENT,weight='bold')
        layout.paragraph(fig,.045,.13,'Geometric estimate, not a slicer result: PLA USD 20/kg, density 1.24 g/cm³, 5% waste. Shells estimated on a 0.6 mm voxel grid; allow approximately ±20%. Slicer-generated support, TPU, hardware, electricity and labor are excluded.',size=9,leading=.022)
        save(fig)

        for name,part in REPORT['parts'].items():
            fig=layout.page(name.replace('_',' ').title(),'Print one  •  Native FreeCAD TechDraw: three orthographic views + isometric  •  Drawing shows local assembly orientation')
            layout.picture(fig,(ROOT/part['drawings']).with_suffix('.png'),[.01,.115,.78,.765])
            layout.picture(fig,ROOT/part['preview_render'],[.795,.57,.195,.27])
            fig.text(.805,.55,'PRINT ORIENTATION',fontsize=10,color=layout.ACCENT,weight='bold')
            y=layout.paragraph(fig,.805,.515,part['print_orientation'],size=9,leading=.024)
            e=part['material_estimate']
            y=layout.paragraph(fig,.805,y-.018,f"PLA: ~{e['estimated_filament_g']:.1f} g\nFilament: ~USD {e['estimated_filament_cost_usd']:.2f}\nInfill: {int(part['infill_fraction']*100)}%",size=10,leading=.030)
            y=layout.paragraph(fig,.805,y-.018,part['support'],size=9,leading=.024)
            if name.startswith('gear_'):
                layout.paragraph(fig,.805,y-.018,'Photo-derived gear fit: print these two small pieces first and check against the actual gear and shaft.',size=9,leading=.024)
            fig.text(.045,.075,'STL: '+part['stl'],fontsize=8.5,color=layout.MUTED)
            save(fig)
            if part.get('seat_detail'):
                fig=layout.page('Motor gearbox seat: near-side relief',
                    'User dimensions locate the gearbox edge; 0.60 mm clearance is the selected printing allowance')
                layout.picture(fig,(ROOT/part['seat_detail']).with_suffix('.png'),[.01,.235,.98,.65])
                layout.paragraph(fig,.045,.195,
                    'The raised circular mounting pad is 7.70 mm outside diameter, with 2.00 mm from its edge to the gearbox edge (user measurements). The gearbox edge is therefore 5.85 mm from the screw centre. The revised pocket wall is 6.45 mm from that centre, providing 0.60 mm nominal clearance.',size=10,leading=.025)
                layout.paragraph(fig,.045,.105,
                    'The near-side recess is widened by 1.55 mm, leaving a 3.85 mm lip. Pocket depth remains 4 mm and the screw-bearing plate remains 3 mm. Dry-fit the revised holder against the actual gearbox before tightening its screws.',size=10,leading=.025)
                save(fig)

        fig=layout.page('Assembly: preserve alignment, then tension both belts','Motor 20T → gear 80T reduces speed; gear 80T → encoder 20T restores motor RPM at the encoder')
        steps=[
            '1. Remove supports from the eight holder washer lands, horizontal round fastener bores and clamp nut pocket. Check the motor holes and encoder mounting group. Fit the motor with four M3 x 6 screws and encoder with three M3 x 8 screws, using seven thin washers. Retain case fasteners.',
            '2. Before mounting the frame to plywood, fit the lower bearing under the integral lower deck and prefit the upper bearing to its removable bridge. Use eight M4 x 16 screws and eight plain nuts. Follow the illustrated bearing-sandwich page; leave bearing fasteners loose enough for alignment.',
            '3. With the upper bridge removed, install the bare shaft, both 80T pulleys and both belts. Lower the bridge and its bearing along the bare shaft onto the locating pins and seating shoulders. Start four M4 x 25 screws from front/rear with four plain nuts accessed through the open sides; tighten evenly after alignment.',
            '4. Fit the lower collar beneath its bearing and the upper collar above its bearing. Face the raised collar surfaces toward the bearing inner rings. Bring them into light contact and tighten without forcing axial preload. Check that the shaft rotates freely.',
            '5. Insert captive slide nuts, then fix both slide bases and the integrated frame feet to plywood with twelve #6 x 1/2 inch flat-head wood screws. Fit the holders to their bases using eight M4 x 12 clamps and eight 1 mm washers. The lower bearing must already be installed because its screw access is from below.',
            '6. Align the 20T motor pulley with the lower belt plane at 61 mm and the 20T encoder pulley with the upper plane at 77 mm. Nominal centers are 72.47 mm; each slide has 15 mm travel. Set only the tension needed for reliable tracking, and tighten the slide clamps evenly.',
            '7. Assemble and fit-check the two-piece gear adapter as on the next page. Turn the mechanism by hand through several revolutions, checking shaft grip, pulley/fastener clearance and belt tracking. Evaluate vibration, stiffness and motor heat in a low-load trial before increasing load.'
        ]
        y=.835
        for step in steps:y=layout.paragraph(fig,.045,y,step,size=10,leading=.024)-.018
        layout.paragraph(fig,.045,y-.01,'Keep the internal frame, bearing mounts and slides rigid. If vibration isolation is needed later, the preferred trial is TPU feet beneath the whole plywood board; any mating gear should share the same support. No TPU or soft bolt washers are included in this release.',size=10,leading=.024)
        save(fig)

        fig=layout.page('Fit the gear adapter before printing the large parts','Two pieces allow assembly around the existing gear  •  Gear bore and key dimensions are inferred from photographs')
        for name,x,label in (('gear_adapter_sleeve_r06',.045,'KEYED SLEEVE + SUPPORT FLANGE'),('gear_adapter_clamp_cap_r06',.535,'REMOVABLE CLAMP CAP')):
            layout.picture(fig,ROOT/REPORT['parts'][name]['preview_render'],[x,.46,.42,.365])
            fig.text(x,.835,label,fontsize=12,color=layout.ACCENT,weight='bold')
        y=layout.paragraph(fig,.045,.425,'1. Check the sleeve slides onto the actual 6 mm D shaft and that its 11.9 mm outside diameter and 2.8 mm key fit the gear. The reference bore is approximately 12 mm. Do not force the gear onto a tight print; adjust the source fit dimensions if necessary.',size=11,leading=.029)
        y=layout.paragraph(fig,.045,y-.018,'2. Slide the sleeve onto the shaft with its flange beneath the gear. Slide the gear over the keyed sleeve, then fit the separate cap over the exposed upper sleeve neck. The cap rests above the gear and can be positioned to match its actual thickness.',size=11,leading=.029)
        y=layout.paragraph(fig,.045,y-.018,'3. Insert the M3 nut from the right outer face into the hex pocket. Install the M3 x 16 screw from the left and tighten gradually until the cap compresses the split sleeve onto the shaft. Check grip against both twisting and axial movement by hand. Do not crush the plastic or close the split fully to compensate for a poor fit.',size=11,leading=.029)
        layout.paragraph(fig,.045,y-.018,'Both pieces use 100% infill. Shaft bore: 6.3 mm, flat at Y=2.65; splits: sleeve 1.2 mm, cap 2.2 mm. The cap has a round 3.4 mm M3 passage and full hex nut pocket; use and remove supports as needed. Clearances are design assumptions. Gear torque, axial load and long-term PLA clamp durability remain unverified.',size=10,leading=.026)
        save(fig)

        fig=layout.page('Dimension sources and limits of this fit set','The encoder pattern previously fitted; new motor centers and the new adapter still need physical fit checks')
        layout.picture(fig,ROOT/'motor holder/output/render/reference/motor_mount_measurement_reference_r05.png',[.045,.31,.91,.54])
        layout.paragraph(fig,.045,.275,'The leveled motor shaft-face photograph gives a 33 x 18 mm M3 hole pattern, with the shaft 9.7 mm from the nearer column. Nominal model holes are 3.8 mm round clearance holes; estimated photo uncertainty is about ±0.3 mm. This release preserves the corrected r05 mounting interface.',size=11,leading=.029)
        layout.paragraph(fig,.045,.165,'Encoder mounting slots retain the user-confirmed pattern. Gear bore/key, device body envelopes and pulley axial dimensions are still reference estimates. Valid solids, watertight meshes, hole/fastener clearances, collar access and slide travel have been checked; physical stiffness, vibration behavior and a working load limit have not been established.',size=10,leading=.026)
        save(fig)

    data=OUT.read_bytes()
    assert data.startswith(b'%PDF') and len(re.findall(rb'/Type\s*/Page\b',data))==count
    assert count==11+len(REPORT['parts'])+sum(bool(p.get('seat_detail')) for p in REPORT['parts'].values()),count
    REPORT['manual']={'pdf':'manual.pdf','pages':count,'revision':'r06','preview_directory':str(PREVIEW),
                      'contains':'Current r06 CAD assembly, hardware quantities, print settings/costs, eight native part drawings, motor recess relief detail, illustrated bearing sandwich, assembly, adapter fit and dimension provenance'}
    (SCRIPTS/'two_belt_r06_report.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print('Created r06 manual:',count,'pages; bounds and table text checks passed')


if __name__=='__main__':main()
