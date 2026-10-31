"""Annotate real CAD renders; no dimension inference or generated imagery."""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from project_paths import ASSEMBLY_RENDER

SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent
OUT=ASSEMBLY_RENDER
TAG=os.environ.get('LISA_RENDER_TAG','draft')
REPORT_PATH=SCRIPTS/os.environ.get('LISA_ASSEMBLY_REPORT','two_belt_draft_report.json')
PROJ=json.loads((SCRIPTS/('two_belt_'+TAG+'_projection.json')).read_text())
REPORT=json.loads(REPORT_PATH.read_text())
INK='#213747'
MUTED='#566977'
ACCENT='#087a9c'
FONTS=Path('/usr/share/fonts/truetype/dejavu')


def font(size,bold=False):
    return ImageFont.truetype(str(FONTS/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),size)


def render_page(group,title,subtitle,labels,footer):
    if TAG=='r06':
        title=title.replace('DRAFT','R06 FIT SET')
        labels=[(anchor,side,y,heading,('M3 x 16 + captive nut;\nremovable sleeve clamp' if anchor=='adapter' else detail))
                for anchor,side,y,heading,detail in labels]
        footer=[line.replace('CAD draft for layout review. Pulley axial heights, gear key fit, shaft grip and strength need final checks.',
                            'Printable fit set: check actual pulley heights, gear key fit, shaft grip and loaded stiffness before use.') for line in footer]
    canvas=Image.new('RGB',(2400,1750),'white')
    raw=Image.open(OUT/('two_belt_'+group+'_'+TAG+'_raw.png')).convert('RGBA')
    offset=((canvas.width-raw.width)//2,180)
    canvas.paste(raw,offset,raw)
    draw=ImageDraw.Draw(canvas)
    draw.text((64,40),title,font=font(52,True),fill=INK)
    draw.text((66,112),subtitle,font=font(27),fill=MUTED)
    draw.line([(65,159),(2335,159)],fill=ACCENT,width=3)
    for anchor,side,y,heading,detail in labels:
        x,ay=PROJ[group]['anchors'][anchor]
        point=(round(x+offset[0]),round(ay+offset[1]))
        tx=64 if side=='left' else 1960
        end=(420 if side=='left' else 1925,y+20)
        elbow=(end[0]+(28 if side=='left' else -28),point[1])
        draw.line([point,elbow,end],fill='#8396a2',width=3)
        draw.ellipse((point[0]-5,point[1]-5,point[0]+5,point[1]+5),fill=ACCENT)
        draw.rectangle((tx-8,y-5,tx+377,y+125),fill='white')
        draw.text((tx,y),heading,font=font(29,True),fill=INK)
        for i,line in enumerate(detail.split('\n')):
            assert draw.textlength(line,font=font(24))<=375,(group,line)
            draw.text((tx,y+42+32*i),line,font=font(24),fill=MUTED)
    draw.line([(65,1617),(2335,1617)],fill='#dce4e8',width=2)
    for i,line in enumerate(footer):
        assert draw.textlength(line,font=font(26))<2270,(group,line)
        draw.text((65,1640+39*i),line,font=font(26),fill=MUTED)
    path=OUT/('two_belt_'+group+'_'+TAG+'.png')
    canvas.save(path)
    return str(path.relative_to(ROOT))


def drawing_page():
    canvas=Image.new('RGB',(2800,2000),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((65,35),'LISA | TWO-BELT LAYOUT — '+('R06 FIT SET' if TAG=='r06' else 'CAD DRAFT'),font=font(53,True),fill=INK)
    draw.text((67,107),'Nominal dimensions in mm. Four views of the same FreeCAD assembly.',font=font(30),fill=MUTED)
    rects={'top':(40,190,1370,810),'assembly':(1420,190,2760,810),
           'front':(40,1015,1370,1810),'side':(1420,1015,2760,1810)}
    placements={}
    titles={'top':'TOP | 250 mm belts; separate tension adjustments',
            'assembly':'ISOMETRIC | exposed motor, separate gear shaft',
            'front':'FRONT | belt heights: 61 / 77 above plywood',
            'side':'SIDE | integral frame and removable upper bridge'}
    for view,rect in rects.items():
        draw.text((rect[0]+15,rect[1]-48),titles[view],font=font(29,True),fill=INK)
        raw=Image.open(OUT/('two_belt_'+view+'_'+TAG+'_raw.png')).convert('RGBA')
        scale=min((rect[2]-rect[0])/raw.width,(rect[3]-rect[1])/raw.height)
        size=(round(raw.width*scale),round(raw.height*scale))
        offset=(round((rect[0]+rect[2]-size[0])/2),rect[1])
        raw=raw.resize(size,Image.Resampling.LANCZOS)
        canvas.paste(raw,offset,raw)
        placements[view]=(offset,scale)
    def projected(view,anchor):
        offset,scale=placements[view]
        p=PROJ[view]['anchors'][anchor]
        return p[0]*scale+offset[0],p[1]*scale+offset[1]
    # Dimensions attach to actual projected shaft axes, not photo estimates.
    axes=[projected('top',a) for a in ('axis_encoder','axis_gear','axis_motor')]
    y=872
    for a,b in zip(axes,axes[1:]):
        draw.line([(a[0],a[1]),(a[0],y+15)],fill='#7a8e9b',width=2)
        draw.line([(b[0],b[1]),(b[0],y+15)],fill='#7a8e9b',width=2)
        draw.line([(a[0],y),(b[0],y)],fill=ACCENT,width=3)
        for x in (a[0],b[0]):
            draw.line([(x-7,y-8),(x+7,y+8)],fill=ACCENT,width=3)
        text=f"{REPORT['shaft_spacing_mm']:.2f}"
        tw=draw.textlength(text,font=font(27,True))
        tx=(a[0]+b[0]-tw)/2
        draw.rectangle((tx-8,y-39,tx+tw+8,y-5),fill='white')
        draw.text((tx,y-40),text,font=font(27,True),fill=ACCENT)
    draw.text((1450,858),'Gear: 1/4 motor RPM  |  Encoder: motor RPM',font=font(29,True),fill=ACCENT)
    draw.text((1450,904),'Two 80T pulleys on one 100 mm steel shaft',font=font(28),fill=MUTED)
    draw.line([(65,1852),(2735,1852)],fill='#dce4e8',width=2)
    draw.text((65,1876),'Bearing centres: 40 / 95.5 above plywood (55.5 span). Steel shaft: 28–128 above plywood.',
              font=font(29),fill=MUTED)
    draw.text((65,1920),('R06 printable fit set: pulley axial dimensions, gear key fit and loaded stiffness need physical checks.' if TAG=='r06' else
                        'Visual draft: pulley axial dimensions, gear key fit, final screw engagement and loaded stiffness are provisional.'),
              font=font(28),fill=MUTED)
    path=OUT/('two_belt_views_'+TAG+'.png')
    canvas.save(path)
    return str(path.relative_to(ROOT))


def main():
    assembly=render_page('assembly','LISA | TWO-BELT DRIVE DRAFT',
        'Motor → supported gear shaft → encoder. Two 250 mm x 6 mm GT2 belts; both devices adjustable.',[
        ('gear','left',260,'Existing gear','Above the upper bearing;\nprinted keyed adapter'),
        ('upper80','left',555,'Upper 80T → 20T','4x encoder speed\nrelative to the gear'),
        ('encoder','left',850,'Encoder + 20T','Runs at motor RPM;\nface raised by 16 mm'),
        ('panel','left',1270,'Integral shaft frame','Twelve local gussets;\nremovable upper bridge'),
        ('upper_collar','right',260,'Exposed upper collar','Above the bearing,\noutside the plastic plate'),
        ('lower80','right',555,'Lower 80T','Both large pulleys\nare between bearings'),
        ('motor20','right',850,'Motor + 20T','Lower belt reduces\ngear speed to 1/4'),
        ('motor_slide','right',1270,'Independent slides','15 mm motor travel;\n15 mm encoder travel')],
        ['Motor 20T → lower 80T: gear at 1/4 motor RPM. Upper 80T → encoder 20T: encoder at motor RPM.',
         'CAD draft for layout review. Pulley axial heights, gear key fit, shaft grip and strength need final checks.'])
    detail=render_page('detail','LISA | BEARINGS AND COLLARS — DRAFT',
        'Actual CAD section: front halves removed. Gold shaft is kept whole. Both collars remain outside the plates.',[
        ('adapter','left',255,'Gear adapter clamp','M3 x 16 reference;\nretention to finalize'),
        ('upper_collar','left',465,'Upper shaft collar','Accessible above\nthe upper bearing'),
        ('upper80','left',795,'Upper 80T pulley','77 mm belt centre\nEncoder speed boost'),
        ('lower_collar','left',1310,'Lower shaft collar','Front window gives\naccess to the pinch bolt'),
        ('shaft_top','right',255,'100 mm steel shaft','6 mm D profile\n28–128 above plywood'),
        ('upper_bearing','right',465,'Upper bearing','On the removable\nupper bridge'),
        ('lower80','right',970,'Lower 80T pulley','61 mm belt centre\nMotor speed reduction'),
        ('lower_bearing','right',1310,'Lower bearing','Under the integral deck;\n55.5 mm bearing span')],
        ['Bearing housings bolt to the plates. Collars clamp to the shaft and lightly contact the bearing inner rings.',
         'Bearing blocks and collars use manufacturer STEP geometry. Pulley and gear envelopes remain provisional.'])
    exploded=render_page('exploded','LISA | BENCH ASSEMBLY — DRAFT',
        'Fit the lower bearing before board mounting. Load both pulleys and belts, then lower the upper bridge.',[
        ('upper_bearing','left',310,'Upper bearing + bridge','Prefit four M4 x 16;\nlower over bare shaft'),
        ('lower_bearing','left',925,'Lower bearing + deck','Slide bearing in sideways;\nfour M4 x 16 screws'),
        ('upper_plate','right',545,'Removable upper bridge','Four M4 x 25 side bolts;\nseats and pins locate it'),
        ('panel','right',925,'One rigid frame body','Twelve gussets replace\nold screw-joint ledges'),
        ('foot','right',1280,'Integrated feet','Four #6 x 1/2 screws;\nmount after bearing fit')],
        ['Exploded view shows only the stationary frame and bearings. Shaft, belts and devices omitted for clarity.',
         'Add collars and gear after the bridge is seated. Start all four side bolts before tightening evenly.'])
    REPORT['renders']={'assembly':assembly,'detail':detail,'exploded':exploded,'views':drawing_page()}
    REPORT_PATH.write_text(json.dumps(REPORT,indent=2)+'\n')
    print('Created three labeled draft renders and four-view CAD layout.')


if __name__=='__main__':
    main()
