"""Check the bearing installation sequence and assumed hand-tool envelopes in CAD."""
import FreeCAD as App
import Part

V=App.Vector


def check(layout):
    shapes=layout.SHAPES
    body=shapes['FrameMainBody'];bridge=shapes['FrameUpperPlate']
    records=[]

    def clear(label,tool,obstacles):
        maximum=0
        for name,solid in obstacles:
            overlap=tool.common(solid).Volume
            assert overlap<1e-4,(label,name,overlap)
            maximum=max(maximum,overlap)
        records.append({'check':label,'maximum_overlap_mm3':round(maximum,8)})

    fixed=[('body',body),('upper bridge',bridge)]
    if any('gusset' in e.get('role','').lower() for e in getattr(layout,'REPORT',{}).get('items',[]) if e['name']=='FrameMainBody'):
        # Check the old full-width rectangular screw flanges are absent, while
        # each of the twelve new local ribs and both bridge seats remain solid.
        for side in (-1,1):
            for z in (14,57.5):
                assert not body.isInside(V(0,side*36,z),1e-7,False),('obsolete ledge',side,z)
            for x in (-24,24):
                for kind,z in (('foot',10),('lower deck',40),('bridge seat',64)):
                    assert body.isInside(V(x,side*36,z),1e-7,False),(kind,side,x)
                    records.append({'check':f'{kind} gusset at X{x}/side{side}','material_present':True})
            assert body.isInside(V(0,side*36,68),1e-7,False),('bridge seat',side)
    clear('Assembled frame parts',bridge,[('body',body)])
    lower=[s for n,s in shapes.items() if n.startswith('LowerBearing') and not n.startswith('LowerBearingMount')]
    upper=[s for n,s in shapes.items() if n.startswith('UpperBearing')]
    # Side access is checked at 1 mm increments, with the bridge removed.
    lower_block=Part.makeCompound(lower)
    for dx in range(0,86):
        clear(f'Lower bearing insertion X+{dx}',layout.moved(lower_block,dx),[('body',body)])
    stack=Part.makeCompound([shapes[n] for n in ('SteelDShaft100mm','LowerShaft80T','UpperShaft80T')])
    for dz in range(0,101):
        clear(f'Shaft and pulley stack lowered from Z+{dz}',layout.moved(stack,0,0,dz),[('body',body)])
    top=Part.makeCompound([bridge]+upper)
    for dz in range(0,31):
        clear(f'Upper bearing bridge lowered from Z+{dz}',layout.moved(top,0,0,dz),
              [('body',body),('shaft and pulleys',stack)])
    # Lower bearing screws must be fitted before the body is screwed to plywood.
    for x in (-16,16):
        for y in (-16,16):
            driver=Part.makeCylinder(2.5,49.4,V(x,y,-16),V(0,0,1))
            clear(f'Lower bearing driver {x}/{y}',driver,[('body',body)])
            for label,z in (('Lower',49.6),('Upper',82.9)):
                head=Part.makeCylinder(7,3,V(x,y,z))
                handle=Part.makeBox(36,7,3,V(x if x>0 else x-36,y-3.5,z))
                clear(f'{label} bearing nut wrench with pulleys fitted {x}/{y}',head.fuse(handle),fixed+
                      [(n,shapes[n]) for n in ('LowerShaft80T','UpperShaft80T')])
    for x in (-24,24):
        for front in (True,False):
            y=-90 if front else 52.1
            driver=Part.makeCylinder(2.5,37.9,V(x,y,78),V(0,1,0))
            clear(f'Bridge screw driver {x}/{front}',driver,fixed)
            # 14 mm outside-diameter open-end wrench head, 3 mm thick.
            # A full disk conservatively encloses the open jaws around the nut.
            cy=-31.9 if front else 28.9
            head=Part.makeCylinder(7,3,V(x,cy,78),V(0,1,0))
            handle=Part.makeBox(36,3,7,V(x if x>0 else x-36,cy,74.5))
            clear(f'Bridge nut wrench {x}/{front}',head.fuse(handle),fixed+
                  [(n,shapes[n]) for n in ('LowerShaft80T','UpperShaft80T')])
    return {'passed':True,'checks':records,
            'sequence':'Bench-fit lower bearing before board mounting; prefit upper bearing to bridge; load bare shaft and both pulleys with both belts; lower bridge onto pins and seats; fit four side bolts; add collars and gear.',
            'tool_assumptions':'5 mm diameter straight driver stem; 7 mm AF open-end wrench with 14 mm OD head, 3 mm thick. Verify actual tools.',
            'sampling_mm':1,
            'limits':'CAD insertion paths and tool envelopes only; support removal, printed fit, actual pulley grip and hand assembly remain unverified.'}


if __name__=='__main__':
    import json
    import sys
    from pathlib import Path
    scripts=Path(__file__).resolve().parent
    sys.path.insert(0,str(scripts))
    import generate_two_belt_draft as layout
    path=scripts/'two_belt_r06_assembly_report.json'
    report=json.loads(path.read_text())
    doc=App.openDocument(str(scripts.parent/report['cad_source']))
    layout.SHAPES={e['name']:doc.getObject(e['cad_object']).Shape for e in report['items']}
    layout.REPORT=report
    result=check(layout)
    report['validation']['frame_assembly']=result
    path.write_text(json.dumps(report,indent=2)+'\n')
    (scripts/'frame_assembly_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Frame installation and tool-access checks passed:',len(result['checks']),flush=True)
