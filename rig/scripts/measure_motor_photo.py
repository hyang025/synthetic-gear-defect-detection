"""Calibrate the leveled motor photo against both metric ruler edges.

This is reference-image analysis, separate from authoritative FreeCAD drawings.
Selected bore-mouth centers were inspected against local circle detection.
The shaft axis comes from the bearing collar, rather than its elevated tip.
Reproduce with the repository .venv Python; never recompress the source JPEG.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT=SCRIPT_DIR.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent.parent/'.cache/matplotlib'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

SOURCE=ROOT/'motor holder/scan/motor_reference_shaft_face_leveled_02.JPG'


def main():
    image=cv2.imread(str(SOURCE));assert image is not None
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    Y,X=np.indices(gray.shape)
    refs=[]
    # Follow the white tick strokes, intersect them with each ruler edge,
    # then identify their integral millimeter index. Missing/merged ticks are
    # skipped; no assumption is made that all components are consecutive.
    masks=[('top',(X>955)&(X<3330)&(Y>923-.110*X-190)&(Y<923-.110*X-15)),
           ('left',(Y>825)&(Y<2180)&(X>925+.110*Y-175)&(X<925+.110*Y-10))]
    for name,mask in masks:
        count,labels,stats,_=cv2.connectedComponentsWithStats(((gray>140)&mask).astype(np.uint8))
        for i,(x,y,w,h,area) in enumerate(stats):
            if area<=150:continue
            if not ((name=='top' and h>90 and w<70) or (name=='left' and w>90 and h<45)):continue
            ys,xs=np.where(labels==i);points=np.column_stack((xs,ys)).astype(float)
            center=points.mean(axis=0)
            _,_,vectors=np.linalg.svd(points-center,full_matrices=False);direction=vectors[0]
            if name=='top':
                t=(923-.110*center[0]-center[1])/(direction[1]+.110*direction[0])
            else:
                t=(925+.110*center[1]-center[0])/(direction[0]-.110*direction[1])
            pixel=center+t*direction
            if name=='top':index=round((pixel[0]-1014.36)/37.05);mm=[index,0]
            else:index=round((pixel[1]-821.02)/36.7);mm=[0,index]
            refs.append({'edge':name,'millimeter_index':index,'plane_mm':mm,'pixel':pixel.tolist()})
    world=np.array([r['plane_mm'] for r in refs],float)
    pixels=np.array([r['pixel'] for r in refs],float)
    H,_=cv2.findHomography(world,pixels,0);inverse=np.linalg.inv(H)
    projected=cv2.perspectiveTransform(world[None],H)[0]
    rms=float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1))))
    assert len(refs)>=50 and rms<3,(len(refs),rms)

    # Use the outer bearing collar on the mounting face. Rays locate its
    # outward dark-to-light edge, then a robust ellipse suppresses reflections.
    initial_center=np.array([1555.,1222.]);axes=np.array([266.,242.])
    angle=np.linspace(0,2*np.pi,360,endpoint=False);radius=np.linspace(.88,1.10,160)
    samples=initial_center[None,None,:]+radius[:,None,None]*np.column_stack(
        (axes[0]*np.cos(angle),axes[1]*np.sin(angle)))[None,:,:]
    smooth=cv2.GaussianBlur(gray,(5,5),1.2)
    values=cv2.remap(smooth,samples[:,:,0].astype('float32'),samples[:,:,1].astype('float32'),cv2.INTER_LINEAR)
    gradient=np.gradient(values.astype(float),axis=0);indices=np.argmax(gradient,axis=0)
    edge_points=samples[indices,np.arange(360)]
    edge_points=edge_points[gradient[indices,np.arange(360)]>5]
    def residual(z):
        cx,cy,ax,ay,theta=z;co,si=np.cos(theta),np.sin(theta)
        dx,dy=edge_points[:,0]-cx,edge_points[:,1]-cy
        return ((co*dx+si*dy)/ax)**2+((-si*dx+co*dy)/ay)**2-1
    ellipse=least_squares(residual,[*initial_center,*axes,0],loss='soft_l1',f_scale=.02,
        bounds=([1500,1170,240,220,-.5],[1600,1270,290,270,.5])).x
    names=['near_top','far_top','near_bottom','far_bottom','shaft_axis']
    centers=np.array([[1166.5,928.5],[2390.5,796.5],[1236.5,1592.5],[2450.5,1456.5],ellipse[:2]])
    plane=cv2.perspectiveTransform(centers[None],inverse)[0]
    pitch_x=float(((plane[1]-plane[0])+(plane[3]-plane[2]))[0]/2)
    pitch_y=float(((plane[2]-plane[0])+(plane[3]-plane[1]))[1]/2)
    near_column=(plane[0]+plane[2])/2
    offset=float((plane[4]-near_column)[0])
    row_offset=float(plane[4,1]-plane[:4,1].mean())
    assert abs(row_offset)<.3,row_offset
    measured={'motor_hole_pitch_x_mm':pitch_x,'motor_hole_pitch_y_mm':pitch_y,
              'motor_shaft_to_left_hole_column_mm':offset}
    nominal={'motor_hole_pitch_x_mm':33.,'motor_hole_pitch_y_mm':18.,
             'motor_shaft_to_left_hole_column_mm':9.7}
    previous={'motor_hole_pitch_x_mm':36.,'motor_hole_pitch_y_mm':19.,
              'motor_shaft_to_left_hole_column_mm':9.}
    relative=plane-plane[4]
    relative[:,1]*=-1
    target=np.array([[-9.7,9],[23.3,9],[-9.7,-9],[23.3,-9]])
    center_errors=np.linalg.norm(relative[:4]-target,axis=1)
    assert center_errors.max()<.3,center_errors
    report={'source':str(SOURCE.relative_to(ROOT)),'original_inbox_name':'R0004472.JPG',
        'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'image_size_px':[image.shape[1],image.shape[0]],
        'method':'58 metric ticks on two orthogonal ruler edges; planar homography; inspected bore-mouth centers; robust outer bearing collar ellipse',
        'ruler_tick_count':len(refs),'ruler_reprojection_rms_px':rms,
        'ruler_reference_points':refs,'plane_to_pixel_homography':H.tolist(),
        'selected_centers':[{'name':name,'pixel':point.tolist(),'plane_mm':metric.tolist(),
                             'shaft_relative_mm':rel.tolist()} for name,point,metric,rel in zip(names,centers,plane,relative)],
        'bearing_collar_ellipse':ellipse.tolist(),'measured_mm':measured,'nominal_model_mm':nominal,
        'previous_r04_mm':previous,'nominal_center_residuals_mm':center_errors.tolist(),
        'estimated_dimension_uncertainty_mm':.3,
        'uncertainty_note':'Approximate photo estimate, not a metrology certification. Tick fit RMS does not bound ruler height, lens distortion, bore-center selection or collar projection error.',
        'hardware_fit_verified':False}
    (SCRIPT_DIR/'motor_hole_measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    with (SCRIPT_DIR/'motor_hole_measurements.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['Dimension','r04 mm','Measured photo mm','r05 nominal mm','Estimated uncertainty mm'])
        for key in measured:writer.writerow([key,previous[key],round(measured[key],3),nominal[key],.3])
    fig,ax=plt.subplots(figsize=(12,8));ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    for name,p in zip(names,centers):
        ax.plot(*p,'+',color='#00d7ff',ms=14,mew=2)
        ax.annotate(name.replace('_',' '),p,xytext=(12,12),textcoords='offset points',color='#00d7ff',fontsize=10,
                    bbox={'facecolor':'#18232b','alpha':.85,'edgecolor':'none'})
    for r in refs:ax.plot(*r['pixel'],'.',color='#eaff66',ms=3)
    ax.set_xlim(875,2780);ax.set_ylim(1940,500);ax.set_axis_off()
    fig.suptitle('Motor reference measurement — R0004472.JPG',fontsize=15)
    fig.text(.5,.035,'Photo estimates: pitch 33.06 × 17.86 mm; shaft offset 9.71 mm. Model: 33 × 18 mm / 9.7 mm.\n'
             '58 ruler ticks, two-axis calibration. Photo estimates ≈±0.3 mm; FreeCAD drawings remain authoritative.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.09,1,.94])
    output=ROOT/'motor holder/output/render/motor_mount_measurement_reference_r05.png'
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=150);plt.close(fig)
    print(json.dumps({'measured_mm':measured,'nominal_mm':nominal,'ruler_ticks':len(refs),
                      'rms_px':rms,'center_residuals_mm':center_errors.tolist()},indent=2))


if __name__=='__main__':main()
