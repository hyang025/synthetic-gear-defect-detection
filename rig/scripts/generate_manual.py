"""Create the project PDF from final CAD drawings, renders and build estimates.

Uses the existing Matplotlib/Pillow toolchain. The dimensioned views are native
FreeCAD TechDraw exports; this script only lays out the manual pages.
"""
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[3]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from PIL import Image

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT=SCRIPT_DIR.parent
REPORT=json.loads((SCRIPT_DIR/'build_report.json').read_text())
REV=REPORT['revision'].split('_')[0]
PARTS=REPORT['parts']
OUT=ROOT/'manual.pdf'
PREVIEW=ROOT.parent.parent/'.cache/lisa-manual'
PREVIEW.mkdir(parents=True,exist_ok=True)
INK='#213443'; ACCENT='#127185'; MUTED='#586674'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})


def page(title, subtitle=''):
    fig=plt.figure(figsize=(11.69,8.27),facecolor='white')
    fig.text(.045,.942,title,fontsize=21,color=INK,weight='bold')
    if subtitle:fig.text(.045,.904,subtitle,fontsize=10,color=MUTED)
    fig.add_artist(plt.Line2D([.045,.955],[.886,.886],transform=fig.transFigure,color=ACCENT,lw=1.5))
    return fig


def paragraph(fig,x,y,text,size=10,color=INK,leading=.025):
    lines=[]
    renderer=fig.canvas.get_renderer()
    font=FontProperties(family='DejaVu Sans',size=size)
    available=(.955-x)*fig.get_figwidth()*fig.dpi
    for original in text.split('\n'):
        line=''
        for word in original.split():
            candidate=(line+' '+word).strip()
            if line and renderer.get_text_width_height_descent(candidate,font,False)[0]>available:
                lines.append(line); line=word
            else:line=candidate
        lines.append(line)
    for line in lines:
        fig.text(x,y,line,fontsize=size,color=color,va='top')
        y-=leading
    return y


def picture(fig,path,rect):
    assert path.exists(),path
    ax=fig.add_axes(rect)
    with Image.open(path) as im:ax.imshow(im.copy())
    ax.set_axis_off()
    return ax


def table(fig,rows,labels,rect,widths,size=10):
    ax=fig.add_axes(rect); ax.axis('off')
    renderer=fig.canvas.get_renderer()
    font=FontProperties(family='DejaVu Sans',size=size)
    wrapped=[]
    for row in rows:
        new=[]
        for value,width in zip(row,widths):
            available=rect[2]*width*fig.get_figwidth()*fig.dpi*.86
            lines=[]; line=''
            for word in value.split():
                candidate=(line+' '+word).strip()
                if line and renderer.get_text_width_height_descent(candidate,font,False)[0]>available:
                    lines.append(line); line=word
                else:line=candidate
            new.append('\n'.join(lines+[line]))
        wrapped.append(new)
    t=ax.table(cellText=wrapped,colLabels=labels,colWidths=widths,loc='center',cellLoc='left',bbox=[0,0,1,1])
    t.auto_set_font_size(False); t.set_fontsize(size)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#d5dfe3'); cell.set_linewidth(.6)
        cell.PAD=.06
        if r==0:
            cell.set_facecolor(ACCENT); cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#eef4f5' if r%2 else 'white')
    return t


def main():
    count=0
    draft_path=SCRIPT_DIR/'two_belt_draft_report.json'
    draft=json.loads(draft_path.read_text()) if draft_path.exists() else None
    if draft and 'renders' not in draft:
        draft=None
    draft_pages=0
    section='draft' if draft else 'legacy'
    bom=json.loads((SCRIPT_DIR/'two_belt_parts.json').read_text()) if draft else None
    if bom:
        actual_printed={item['name'] for item in draft['items'] if item['printed']}
        assert actual_printed=={item['cad_name'] for item in bom['printed']}
        columns=['Category','Component','Quantity','Specification','Source / status','Supplier','SKU','URL','Unit price USD']
        with (SCRIPT_DIR/'two_belt_parts.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(columns)
            for category in ('core','reused','fasteners','printed'):
                for item in bom[category]:
                    assert isinstance(item['quantity'],int) and item['quantity']>0
                    writer.writerow([category,item['name'],item['quantity'],item['specification'],item['status'],
                                     item.get('supplier',''),item.get('sku',''),item.get('url',''),
                                     f"{item['unit_price_usd']:.2f}" if 'unit_price_usd' in item else ''])
    with PdfPages(OUT,metadata={'Title':('Lisa two-belt draft and goBILDA parts' if draft else 'Lisa motor and encoder mounts — '+REV),
                   'Author':'3dprint_generator','Subject':'CAD drawings, hardware, assembly and print recommendations'}) as pdf:
        def save(fig):
            nonlocal count
            count+=1
            footer=('LISA  /  TWO-BELT VISUAL DRAFT  /  dimensions in mm' if section=='draft' else
                    'LISA  /  '+REV.upper()+' EARLIER MOUNT FIT-TEST  /  original single-belt layout' if draft else
                    'LISA  /  '+REV.upper()+'  /  dimensions in mm')
            fig.text(.045,.035,footer,fontsize=8,color=MUTED)
            fig.text(.955,.035,f'{count:02d}',fontsize=9,color=MUTED,ha='right')
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            pagebox=fig.bbox
            for annotation in fig.texts:
                box=annotation.get_window_extent(renderer)
                assert box.x0>=0 and box.y0>=0 and box.x1<=pagebox.x1 and box.y1<=pagebox.y1, (count,annotation.get_text())
            for axes in fig.axes:
                for tab in axes.tables:
                    for cell in tab.get_celld().values():
                        box=cell.get_text().get_window_extent(renderer)
                        bounds=cell.get_window_extent(renderer)
                        assert box.x0>=bounds.x0 and box.x1<=bounds.x1 and box.y0>=bounds.y0 and box.y1<=bounds.y1, (count,cell.get_text().get_text())
            pdf.savefig(fig)
            fig.savefig(PREVIEW/f'page_{count:02d}.png',dpi=140,facecolor='white')
            plt.close(fig)

        if draft:
            for key in ('assembly','views','detail','exploded'):
                fig=plt.figure(figsize=(11.69,8.27),facecolor='white')
                picture(fig,ROOT/draft['renders'][key],[.01,.057,.98,.94])
                save(fig)
                draft_pages+=1
            fig=page('goBILDA parts for the two-belt draft',
                     'Prices checked '+bom['prices_checked']+'  •  USD  •  Blue part numbers link to the supplier pages')
            rows=[[item['name'],str(item['quantity']),item['specification'],item['sku'],
                   f"${item['unit_price_usd']:.2f}",f"${item['unit_price_usd']*item['quantity']:.2f}"] for item in bom['core']]
            tab=table(fig,rows,['Component','Quantity','Specification','goBILDA part number','Each','Subtotal'],
                      [.045,.555,.91,.29],[.19,.08,.33,.22,.09,.09],size=10)
            for row,item in enumerate(bom['core'],1):
                text=tab[row,3].get_text(); text.set_url(item['url']); text.set_color(ACCENT)
            subtotal=sum(item['quantity']*item['unit_price_usd'] for item in bom['core'])
            fig.text(.045,.505,f'Shaft, bearing blocks and collars: USD {subtotal:.2f}',fontsize=17,color=ACCENT,weight='bold')
            paragraph(fig,.045,.46,'This is the hardware selection for the pictured layout. The two bearing blocks and two collars already use manufacturer CAD in the draft. Each block includes its bearing, and each collar includes its pinch bolt. No extra loose bearings or collar bolts are needed.',size=11,leading=.028)
            paragraph(fig,.045,.35,'The subtotal covers these three component types only. Belt, other screws/nuts/washers, filament, shipping and tax are additional. Reuse your motor, encoder, gear and four pulleys.',size=11,leading=.028)
            fig.text(.045,.255,'ONE MATCHING BELT TO REPLACE THE MISSING ONE',fontsize=12,color=ACCENT,weight='bold')
            table(fig,[['Replacement timing belt','1','Closed-loop GT2 / 2GT; 250 mm pitch length x 6 mm width; 2 mm pitch; 125 teeth']],
                  ['Component','Quantity','Specification'],[.045,.125,.91,.10],[.25,.09,.66],size=10)
            note=fig.text(.045,.085,'Exact 250 mm belt not found in the checked goBILDA catalog; source a matching belt elsewhere.',fontsize=10,color=ACCENT)
            note.set_url('https://www.gobilda.com/2mm-gt2-timing-belts/')
            save(fig)
            draft_pages+=1

            fig=page('Two-belt draft: complete fastening list',
                     'Quantity means individual pieces for the whole assembly. Check existing stock before ordering.')
            rows=[[item['name'],str(item['quantity']),item['specification'],item.get('sku','Existing / generic')]
                  for item in bom['fasteners']]
            tab=table(fig,rows,['Location / component','Quantity','Screw, nut or washer','goBILDA part number'],
                      [.045,.275,.91,.57],[.255,.08,.445,.22],size=9)
            for row,item in enumerate(bom['fasteners'],1):
                if item.get('url'):
                    text=tab[row,3].get_text(); text.set_url(item['url']); text.set_color(ACCENT)
            y=paragraph(fig,.045,.245,'The 28 M4 nuts cover 8 bearing bolts, 12 frame bolts, 4 motor clamps and 4 encoder clamps. Use plain hex nuts for the draft stack. Bearing bolts have little spare length: adding washers or using thicker locknuts changes engagement. Longer upper bearing bolts can hit the upper pulley.',size=10,leading=.024)
            y=paragraph(fig,.045,y-.012,'Motor M3 x 6 and encoder M3 x 8 each give about 2.4 mm engagement with the specified washers. If M3 x 8 is unavailable, encoder M3 x 10 plus a rigid 2 mm spacer is the earlier documented alternative. Verify actual encoder threads and depth.',size=10,leading=.024)
            paragraph(fig,.045,y-.012,'Encoder clamp length and adapter nut retention remain unresolved. The goBILDA M4 washer is a candidate: confirm its thickness against the approximately 1 mm washer budget. Actual supplier screw-head/nut envelopes need a final fit check before the printable release.',size=10,leading=.024)
            save(fig)
            draft_pages+=1

            fig=page('Printed parts and components to reuse',
                     'Current two-belt layout  •  11 printed pieces  •  PLA structural parts  •  No TPU pads yet')
            printed_details=['Broad side walls','15 mm adjustment','Face height 58 mm','15 mm adjustment',
                             '8 mm panel','8 mm panel','6 mm base + flange','6 mm base + flange',
                             'Bearing underneath','Bearing above','Keyed bore + clamp']
            table(fig,[[item['name'],str(item['quantity']),detail] for item,detail in zip(bom['printed'],printed_details)],
                  ['Printed component','Quantity','Function / feature'],[.045,.22,.465,.625],[.45,.14,.41],size=9)
            reused_details=['Existing device','Existing device','Motor and encoder','Both on gear shaft',
                            'One existing; one to buy','Existing keyed gear','About 10 mm thick']
            table(fig,[[item['name'],str(item['quantity']),detail] for item,detail in zip(bom['reused'],reused_details)],
                  ['Other component','Quantity','Reuse / source'],[.54,.435,.415,.41],[.36,.15,.49],size=9)
            y=paragraph(fig,.54,.40,'PLA starting settings: 0.20 mm layers, 6 wall loops (about 2.5–2.7 mm), 6 top/bottom layers (1.2 mm), 40% gyroid.',size=10,leading=.027)
            y=paragraph(fig,.54,y-.012,'Print the flat panels on their broad faces and plates with flanges upward. These are draft intentions; final hole roofs, supports and orientations still need a printability check.',size=10,leading=.027)
            paragraph(fig,.54,y-.012,'The 11-piece draft has no released STL set or final filament/cost estimate. Existing r05 files and their estimates later in this manual belong to the earlier three-piece fit test.',size=10,leading=.027)
            fig.text(.045,.16,'Parts table: scripts/two_belt_parts.csv',fontsize=11,color=ACCENT,weight='bold')
            save(fig)
            draft_pages+=1

            fig=page('Two-belt draft: assembly and checks',
                     'Proposed assembly sequence  •  The gear shaft is fixed; both device mounts slide for tension')
            steps=[
                '1. Bench-mount the lower bearing block underneath its lower plate and the upper block above its upper plate. Use eight M4 x 16 screws and eight plain M4 nuts. Leave the bolts loose for alignment.',
                '2. Join the foot rails, rear panel and plates using the M4 x 25 screws and plain nuts. Keep the front panel off while installing the shaft, pulleys and collars.',
                '3. Feed the 100 mm steel shaft through both bearings, capturing both 80T pulleys between them. Put both belts around their central pulleys before closing the frame. Align the bearings with the shaft in place, then tighten their mounting bolts.',
                '4. Fit the lower collar below the lower bearing and the upper collar above the upper bearing. Face each raised collar surface toward the bearing inner ring. Bring them into light contact, tighten, and check free rotation without forcing axial preload.',
                '5. Install the front panel to finish the 12 frame joints. Mount the fixed frame and both slide bases to plywood with 12 #6 x 1/2 inch flat-head wood screws: four on each base and four on the frame feet. The draft has no TPU pads.',
                '6. Fit motor 20T to the lower belt and encoder 20T to the upper belt. Each stage has 72.47 mm nominal shaft spacing and 15 mm adjustment travel. Align pulley grooves before setting modest belt tension and tightening both sets of slide clamps.',
                '7. Fit the existing gear using the printed adapter only after its key fit and clamp retention are finalized. Turn the whole mechanism by hand to check clearance, tracking and free rotation before evaluating loaded stiffness and vibration.'
            ]
            y=.835
            for step in steps:
                y=paragraph(fig,.045,y,step,size=10,leading=.024)-.016
            y=paragraph(fig,.045,y-.005,'Speed: motor 20T → lower 80T gives one-quarter motor RPM at the gear. Upper 80T → encoder 20T gives four encoder turns per gear turn, matching motor RPM. The encoder alone gains four times the counts at the gear; belt compliance and the remaining mechanical errors still contribute.',size=10,leading=.024)
            paragraph(fig,.045,y-.012,'Before a printable release: confirm pulley hub/flange heights, shaft engagement, the gear key dimensions, encoder clamp length, adapter nut retention and loaded stiffness. The later r05 pages are historical fit-test references with the original pulley arrangement.',size=10,leading=.024)
            save(fig)
            draft_pages+=1
            section='legacy'

        fig=page('Earlier r05 motor + encoder fit-test set' if draft else 'Lisa motor + encoder mounts',
                 f'{REV.upper()}  •  Original single-belt arrangement  •  Retained mounting reference' if draft else
                 f'{REV.upper()}  •  Motor adjusts for belt tension  •  Encoder remains fixed')
        picture(fig,ROOT/'motor holder/output/render/motor_encoder_assembly_reference.png',[.025,.18,.65,.69])
        fig.text(.70,.81,'EARLIER FIT-TEST SET' if draft else 'BUILD THIS SET',fontsize=12,color=ACCENT,weight='bold')
        paragraph(fig,.70,.765,'1 moving motor holder\n1 motor slide base\n1 fixed encoder holder',size=12,leading=.036)
        paragraph(fig,.70,.57,'Motor: M3 × 6 mm\nEncoder: M3 × 8 mm\nM4 × 12 mm clamps\n#6 flat-head wood screws',size=11,leading=.030)
        paragraph(fig,.70,.43,'Both shafts point upward. The motor carries the 80T pulley and the encoder the 20T pulley. The motor can stays exposed.',size=10)
        paragraph(fig,.045,.17,f'Earlier fit-test hardware is on page {draft_pages+3}; parts on pages {draft_pages+5}–{draft_pages+7}; underside seats on page {draft_pages+8}. Motor holes are corrected to 33 × 18 mm centers and 9.7 mm shaft offset. Page {draft_pages+14} shows the measurement evidence. Encoder and slide-base geometry are unchanged from r04. No TPU pads in this build.',size=10)
        save(fig)

        fig=page('Assembly and adjustment','Teal: shortened motor holder + fixed slide base. Green: encoder holder directly on plywood.')
        picture(fig,ROOT/'motor holder/output/render/motor_encoder_assembly_reference_reverse.png',[.015,.19,.67,.69])
        fig.text(.70,.81,'MOTOR SLIDE',fontsize=12,color=ACCENT,weight='bold')
        paragraph(fig,.70,.765,'15 mm total travel along the shaft-center line. At the illustrated 72.5 mm nominal spacing, travel covers approximately 65–80 mm.',size=11,leading=.030)
        paragraph(fig,.70,.58,'Loosen the four M4 clamps, move the motor to remove belt slack, then tighten all four evenly. The #6 screws holding the base to plywood stay in place.',size=11,leading=.030)
        paragraph(fig,.70,.38,'Both device mounting faces: 42 mm above plywood. Estimated body clearances: motor 11 mm above its slide base; encoder 5 mm above plywood.',size=10)
        paragraph(fig,.045,.16,'Hardware bodies, belt and pulleys are reference envelopes. New gear photos show an approximately Ø12 keyed bore. The gear is absent from r05, and the pulley occupies the encoder shaft: a supported-shaft assembly is still needed. Everything must stay above plywood; the motor may be raised to the new belt height.',size=10)
        save(fig)

        fig=page('Earlier r05 fit-test: screws and washers' if draft else 'Screws and washers: current build','Approximately 10 mm plywood. Encoder mounts directly to wood; no spacers or TPU pads in this build.')
        rows=[
            ['Motor → holder','4','M3 × 6 mm, round/pan/button/socket head','4 M3 flat washers; ~0.6 mm thick'],
            ['Encoder → holder*','3','M3 × 8 mm, round/pan/button/socket head','3 M3 flat washers; ~0.6 mm thick'],
            ['Encoder alternative*','3','M3 × 10 mm instead of M3 × 8','3 rigid 2 mm spacers + 3 thin washers'],
            ['Motor feet → slide base','4','M4 × 12 mm, button/socket head','4 M4 washers, ~1 mm thick; OD ≤10 mm'],
            ['Under motor slide base','4','M4 hex nuts, 7 mm across flats','~3.2 mm thick; captive underside channels'],
            ['Motor slide base → plywood','4','#6 × 1/2 in FLAT-HEAD WOOD screws','12.7 mm long; ~6.7 mm into wood'],
            ['Encoder feet → plywood','4','#6 × 1/2 in FLAT-HEAD WOOD screws','12.7 mm long; ~6.7 mm into wood'],
        ]
        table(fig,rows,['Location','Quantity','Screw / nut specification','Washer / length budget'],[.045,.42,.91,.425],[.25,.075,.35,.325],size=9)
        paragraph(fig,.045,.37,'Use eight #6 × 1/2 inch flat-head WOOD screws total: four for the motor base and four for the encoder. #6 is the imperial wood-screw size, separate from M3 and M4 machine screws. The wood screws need no washers or nuts here.',size=10)
        paragraph(fig,.045,.255,'M3 lengths are measured under the head. Motor: 6 − 3 mm plate − 0.6 mm washer = 2.4 mm engagement. Encoder: 8 − 5 − 0.6 = 2.4 mm. If using M3 × 10 on the encoder, add a 2 mm rigid spacer beneath each washer; longer screws alone may bottom out. All top screw-head recesses are removed.',size=10)
        paragraph(fig,.045,.15,'* Encoder quantity assumes a three-hole mounting group. Use its actual mounting holes and leave case-retention screws intact. Encoder M3 threads and usable depth still need a fit check. Wood-screw lengths include the flat head; the listed wood engagement includes the point. Head fit and seat depth alter engagement slightly.',size=9,leading=.022)
        save(fig)

        fig=page('Later TPU pads and screw length','90A TPU is available. Pads will be designed later; they are not included in this STL set or cost estimate.')
        rows=[
            ['Motor slide base','4','6.0 mm','2–3 mm','#6 × 5/8 in (15.875 mm)','6.9–7.9 mm'],
            ['Encoder feet','4','6.0 mm','2–3 mm','#6 × 5/8 in (15.875 mm)','6.9–7.9 mm'],
        ]
        table(fig,rows,['Fastened stack','Quantity','Rigid thickness','TPU pad','Wood screw','Length in wood*'],[.045,.65,.91,.18],[.25,.075,.14,.10,.255,.18],size=10)
        y=paragraph(fig,.045,.58,'Use eight #6 × 1/2 inch screws until pads are added. With 2–3 mm pads under the motor base and encoder feet, change to eight #6 × 5/8 inch screws. The M4 clamps connect printed parts only, so their M4 × 12 length stays the same.',size=12,leading=.033)
        y=paragraph(fig,.045,y-.025,'Use the compressed pad thickness when checking screw length and pulley alignment. Equal 2.5 mm pads raise both mounting faces from 42 to 44.5 mm before compression. Independent pads may compress differently; align the belt grooves after tightening.',size=11,leading=.030)
        y=paragraph(fig,.045,y-.025,'Wood-screw length − (6 mm fastening land + compressed pad thickness) = approximate length in wood. The motor base and encoder screw feet both have 6 mm lands. Retain the intended purchase without piercing the plywood underside.',size=11,leading=.030)
        paragraph(fig,.045,.18,'* Includes the screw point. Actual plywood is only approximately 10 mm thick. The printed #6 seats assume 4.2 mm clearance, an 8 mm head opening and an 82° countersink. Verify the actual head seats correctly and check underside clearance.',size=10)
        save(fig)

        partinfo=[
            ('motor_holder','Motor holder','DEFAULT: print one','M3 × 6 + thin washers. Round Ø3.8 holes, 33 × 18 mm centers. Underside gearbox seat, 3 mm net plate. Print top-down; support four M4 washer lands.'),
            ('motor_slide_base','Motor slide base','DEFAULT: print one','Insert four M4 nuts underneath before mounting. Four #6 × 1/2 in wood screws. Print countersinks down, nut channels up.'),
            ('encoder_holder','Fixed encoder holder','DEFAULT: print one','M3 × 8 + thin washers. Underside Ø39.5 seat, 5 mm net plate. Four #6 × 1/2 in wood screws. Print top-down.'),
        ]
        for stem,title,status,notes in partinfo:
            name=stem+'_'+REV; part=PARTS[name]
            fig=page(title,status+'  •  Render and three orthographic views + isometric, all from CAD')
            draw=ROOT/Path(part['stl']).parent/'output/render'/(name+'_dimensioned.png')
            picture(fig,draw,[.008,.105,.775,.765])
            picture(fig,ROOT/part['preview_render'],[.78,.49,.215,.34])
            e=part['material_estimate']
            paragraph(fig,.795,.47,f"PLA: ~{e['estimated_filament_g']:.1f} g\nFilament: ~USD {e['estimated_filament_cost_usd']:.2f}",size=11,leading=.032)
            paragraph(fig,.795,.365,notes,size=9,leading=.025)
            fig.text(.045,.075,'STL: '+part['stl'],fontsize=9,color=MUTED)
            save(fig)

        fig=page('Underside seating pockets','CAD-derived underside renders. Both device seating faces remain 42 mm above plywood.')
        for stem,x,title in (('motor_holder',.045,'MOTOR'),('encoder_holder',.525,'ENCODER')):
            part=PARTS[stem+'_'+REV]
            picture(fig,ROOT/part['underside_preview_render'],[x,.32,.43,.52])
            fig.text(x,.835,title,fontsize=12,color=ACCENT,weight='bold')
        paragraph(fig,.045,.285,'Motor: open-ended gearbox pocket, 4 mm deep, with 0.6 mm nominal side clearance. The top retains 3 mm of material above the screw pads. Its bearing collar appears level with those pads, so the motor top is not flush with the collar.',size=10)
        paragraph(fig,.045,.17,'Encoder: Ø39.5 mm pocket, 2 mm deep, leaving 5 mm above the housing face. The top is nominally flush with the photo-estimated 5 mm boss. New housing profiles and exact boss heights still need a fit check. Native seat-detail PDFs include dimensions of the actual pocket edges.',size=10)
        save(fig)

        fig=page('Earlier r05 fit-test: print settings and estimate' if draft else 'Print settings and material estimate','Bambu Lab A1 mini  •  0.4 mm stainless steel nozzle  •  PLA structural parts')
        total=REPORT['assembly_material_estimate']
        settings=[
            ['Layer height','0.20 mm'],['Wall loops / thickness','6 walls; approximately 2.5–2.7 mm'],
            ['Top / bottom shells','6 layers each; 1.2 mm'],['Infill','40% gyroid'],
            ['Orientation','Holders top-down; base nut channels upward'],
            ['Support / brim','Motor: local support at 4 M4 washer lands; optional 3 mm brim'],
        ]
        table(fig,settings,['Setting','Recommendation'],[.045,.49,.51,.36],[.37,.63],size=11)
        rows=[]
        for name,part in PARTS.items():
            e=part['material_estimate']; label=name.removesuffix('_'+REV).replace('_',' ')
            rows.append([label,str(part['default_quantity']),f"{e['estimated_filament_g']:.1f} g",f"USD {e['estimated_filament_cost_usd']:.2f}"])
        table(fig,rows,['Part','Quantity','PLA','Filament cost'],[.59,.49,.365,.36],[.43,.16,.17,.24],size=8)
        fig.text(.045,.405,f"Default set: approximately {total['estimated_filament_g']:.1f} g PLA / USD {total['estimated_filament_cost_usd']:.2f}",fontsize=17,color=ACCENT,weight='bold')
        paragraph(fig,.045,.345,'Planning estimate, not a slicer result. Assumed PLA price: USD 20/kg; density: 1.24 g/cm³; waste: 5%. A 0.6 mm voxel approximation estimates 2.52 mm walls and 1.2 mm skins, then applies 40% infill to the remaining interior. Allow about ±20%; actual slicer settings and spool price determine the final estimate.',size=11,leading=.030)
        paragraph(fig,.045,.19,'Total excludes future TPU pads, hardware and slicer-generated supports. Paint support only beneath the four motor M4 washer lands, inside their Ø11 driver chambers. Keep support off the housing seats and M3 openings. Encoder #6 access tapers continuously; the inverted base has upward-opening nut channels. Inspect the sliced toolpaths before printing.',size=10)
        save(fig)

        fig=page('Assembly sequence and final fit check','Keep the encoder fixed. Adjust only the motor carriage when setting belt tension.')
        steps=[
            '1. Print in the supplied STL orientations. Paint local supports at the four motor M4 washer lands. Remove supports/brim and inspect the plain M3 openings and seating pockets.',
            '2. Seat each housing in its underside pocket. Motor: M3 × 6 with thin washers. Encoder: M3 × 8, or M3 × 10 plus 2 mm rigid spacers and thin washers. Start by hand; check clamping and thread depth. Retain encoder case screws.',
            '3. Seat four M4 nuts in the underside channels of the motor slide base. Set the motor carriage near the middle of its travel and loosely fit four M4 × 12 clamps with 1 mm washers. Check full nut engagement and that ends stay within the base.',
            '4. Lay out the two shaft centers around 72.5 mm apart on the plywood. Point the motor can away from the encoder. Test the actual belt and encoder gear clearance before marking screw positions.',
            '5. Attach the motor slide base and the encoder holder directly to plywood, using four #6 × 1/2 in flat-head wood screws for each. Size pilot holes to the actual wood screws and plywood; keep screw points within the board.',
            '6. Fit 80T to the motor and 20T to the encoder. Align the belt grooves axially. Install the belt, slide the motor just enough to remove slack, and tighten all four M4 clamps evenly.',
            f'7. Turn the shafts by hand to check clearances, pulley alignment and gear motion, then evaluate vibration and motor heat in use. When TPU pads are added, switch the wood screws as shown on page {draft_pages+4} and realign the grooves.',
        ]
        y=.84
        for step in steps:y=paragraph(fig,.045,y,step,size=11,leading=.028)-.019
        save(fig)

        fig=page('Files, sources and validation','FreeCAD is authoritative for the printed geometry. Blender provides the supporting renders.')
        y=paragraph(fig,.045,.835,'manual.pdf is the combined visual reference in the Lisa project folder. Print-ready STLs are directly in each part folder. Editable FreeCAD documents and STEP copies remain in output/model/. Native TechDraw SVG/PDF/PNG drawings, seat details and previews remain in output/render/. The complete '+REV+' ZIP includes this manual and the regeneration sources.',size=12,leading=.032)
        y=paragraph(fig,.045,y-.035,'Regenerate from the repository root: projects/lisa/scripts/build_mounts.sh. Project scripts, mount_parameters.json, build_report.json and print_estimates.csv are in lisa/scripts/. Edit the parameters there and rebuild to keep solids, meshes, drawings, renders, estimates and this manual consistent.',size=11,leading=.030)
        y=paragraph(fig,.045,y-.035,'Validation: all STLs are single watertight bodies with consistent winding, positive volume, Z=0 at the bed and dimensions within the A1 mini envelope. Native CAD reopens and drawing dimensions agree with CAD. Seating floors are checked in the solids. Reference hardware clears the prints, exposed M3 heads clear the example pulleys, and motor travel clears the fixed encoder.',size=11,leading=.030)
        y=paragraph(fig,.045,y-.035,'Fit evidence: the user confirms the encoder pattern fits and reports the motor holes were misplaced. The leveled motor face photo R0004472.JPG gives approximately 33.06 × 17.86 mm pitch and 9.71 mm shaft offset. This build uses 33 × 18 mm, 9.7 mm offset and round Ø3.8 holes; physical motor fit is unverified. Seating heights remain 42 mm. Pocket profiles, exact boss heights, encoder thread depth, actual hubs and the extra gear still need fit checks.',size=10,leading=.027)
        paragraph(fig,.045,.13,'Revisions r01–r04 are preserved under archive/lisa/. This '+REV+' corrects the motor hole pattern and replaces its long M3 slots with round holes. Encoder and slide-base geometry are retained.',size=10)
        save(fig)
        for stem,title in (('motor_holder','Motor underside seat: CAD detail'),
                           ('encoder_holder','Encoder underside seat: CAD detail')):
            part=PARTS[stem+'_'+REV]
            fig=page(title,'Native FreeCAD underside, hidden-edge side and isometric views; actual pocket and floor dimensions.')
            detail=ROOT/Path(part['stl']).parent/'output/render'/(stem+'_'+REV+'_seat_detail.png')
            picture(fig,detail,[.02,.08,.96,.79])
            save(fig)
        measurement=json.loads((SCRIPT_DIR/'motor_hole_measurements.json').read_text())
        fig=page('Motor holes: photo measurement evidence','R0004472.JPG  •  Two metric ruler edges  •  Bearing collar used for shaft-axis center')
        picture(fig,ROOT/'motor holder/output/render/motor_mount_measurement_reference_r05.png',[.025,.27,.62,.60])
        rows=[['Column pitch','36.0','33.06','33.0'],
              ['Row pitch','19.0','17.86','18.0'],
              ['Shaft → near column','9.0','9.71','9.7']]
        table(fig,rows,['Dimension','r04','Photo','r05'],[.665,.51,.29,.29],[.43,.19,.19,.19],size=8)
        paragraph(fig,.665,.47,'All values in mm.\n\nr05: four round Ø3.8 M3 clearance holes. Centers relative to shaft: X = −9.7 / +23.3; Y = ±9.0.',size=9,leading=.025)
        paragraph(fig,.045,.24,'Yellow dots mark 58 metric ruler ticks. Blue crosses mark four bore-mouth centers and the bearing-collar center. A planar perspective correction uses both ruler directions; the raised shaft tip is not used as the shaft-axis reference.',size=11,leading=.030)
        paragraph(fig,.045,.13,f"Photo dimensions are estimated to about ±{measurement['estimated_dimension_uncertainty_mm']:.1f} mm. Ruler-fit RMS is {measurement['ruler_reprojection_rms_px']:.2f} pixels; it does not include ruler height, lens distortion or feature-selection error. Scripts and JSON/CSV evidence are in lisa/scripts/. A physical motor fit check is still needed.",size=10,leading=.027)
        save(fig)
        concept_path=SCRIPT_DIR/'shaft_concept_report.json'
        concept_pages=0
        if concept_path.exists() and not draft:
            concept=json.loads(concept_path.read_text())
            if 'renders' in concept:
                for key in ('assembly','cutaway','loadpath','collars'):
                    if key not in concept['renders']:
                        continue
                    fig=plt.figure(figsize=(11.69,8.27),facecolor='white')
                    picture(fig,ROOT/concept['renders'][key],[.01,.057,.98,.94])
                    save(fig)
                    concept_pages+=1
                if 'collars' in concept['renders']:
                    fig=page('Separated bearings, retention and board clearance',
                             'Pulley between bearings; broad panels carry belt pull along X. New geometry remains a concept.')
                    rows=[['Steel shaft','1','6 mm D shaft, 100 mm long; length still illustrative'],
                          ['Single-bearing blocks','2','goBILDA 1603-0032-0006; one bearing in each block'],
                          ['Shaft collars','2','goBILDA 2910-0919-0006; supplied M4 pinch bolts'],
                          ['Flexible coupling','1','6 mm / 6 mm reference part; 25 mm long in supplier CAD'],
                          ['Bearing block screws','8','M4 x 20 mm reference geometry; four on each block'],
                          ['Bearing block nuts','8','M4 hex nuts'],
                          ['Upper crossplate screws','4','M4 x 20 mm reference geometry'],
                          ['Upper crossplate nuts','4','M4 hex nuts in side-access pockets'],
                          ['Pedestal wood screws','4','#6 x 1/2 inch reference screws; new pedestal only']]
                    table(fig,rows,['Component','Quantity','Specification / status'],[.045,.49,.91,.36],[.28,.10,.62],size=9)
                    y=paragraph(fig,.045,.46,'Bearing centres are 87 and 125 mm above plywood: a 38 mm span. The belt plane is at 114 mm, inside that span. In ideal belt-only statics, the lower bearing carries 0.29 F and the upper 0.71 F. F is the unknown resultant belt pull. These ratios exclude gear loads, frame flexibility and dynamics; they do not establish stiffness or a load rating.',size=10,leading=.027)
                    y=paragraph(fig,.045,y-.015,'Fit one collar below the lower bearing and one above the upper bearing. Point each raised face toward its bearing inner ring. Slide into light contact, tighten the pinch bolts using the access holes, and check free rotation with minimal axial play. The lower metal block is flipped so its bearing face is flush on the collar side. Do not force the collars together to preload the bearings.',size=10,leading=.027)
                    paragraph(fig,.045,y-.015,'The upper crossplate is removable; its four screws join it to both broad panels. Align both bearing blocks with the shaft inserted before tightening their eight mounting screws. The encoder remains above plywood. Moving it below is an available alternative, requiring a shaft/coupling opening and board elevation; roughly 50 mm below-board clearance is a layout allowance only.',size=10,leading=.027)
                    save(fig)
                    concept_pages+=1
    data=OUT.read_bytes()
    assert data.startswith(b'%PDF')
    assert len(re.findall(rb'/Type\s*/Page\b',data))==count
    assert count==14+concept_pages+draft_pages,count
    REPORT['manual']={'pdf':'manual.pdf','pages':count,'revision':REV,
                      'contains':('Current two-belt visual draft, four-view layout, shaft section, exploded bearing/frame assembly, linked goBILDA parts selection, complete numeric fastening/printed/reused quantities and proposed assembly; earlier r05 fit-test drawings, hardware, estimates and photo evidence' if draft else
                                  'r05 drawings, hardware, estimates and photo evidence; separated-bearing assembly/cutaway, belt load path, collar contact and proposed shaft hardware/assembly instructions when available')}
    if bom:
        REPORT['manual']['draft_pages']=draft_pages
        REPORT['manual']['parts_list']='scripts/two_belt_parts.csv'
        REPORT['manual']['gobilda_core_subtotal_usd']=round(subtotal,2)
        REPORT['manual']['prices_checked']=bom['prices_checked']
    (SCRIPT_DIR/'build_report.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(f'Created {OUT.relative_to(ROOT)}: {count} pages; page previews in {PREVIEW}')


if __name__=='__main__':main()
