"""Crop a pavement ROI without retiming/resizing, preserving original video and audio."""
from __future__ import annotations
import argparse,subprocess
from pathlib import Path
import cv2,numpy as np
from common import digest,now,write_json,save_image


def video_info(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot decode: {path}')
    result=dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    result['video_duration_seconds']=result['frames']/result['fps']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--ffmpeg',required=True)
    p.add_argument('--crop',default='0,0,406,406',help='left,top,width,height in source pixels')
    args=p.parse_args()
    source,output,exe=Path(args.input).resolve(),Path(args.output).resolve(),Path(args.ffmpeg).resolve()
    left,top,width,height=map(int,args.crop.split(','))
    if source==output or output.exists():raise ValueError('Output must be new and different from original')
    if not exe.is_file():raise ValueError('FFmpeg executable missing')
    info=video_info(source)
    if (min(left,top)<0 or min(width,height)<=0 or width%2 or height%2 or
        left+width>info['width'] or top+height>info['height']):raise ValueError('Invalid source crop')
    original_sha=digest(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.stem+'.encoding.mp4')
    if partial.exists():raise ValueError('Partial output already exists')
    command=[str(exe),'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(source),
         '-map','0:v:0','-map','0:a?','-vf',f'crop={width}:{height}:{left}:{top}',
         '-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p',
         '-fps_mode','passthrough','-c:a','copy','-map_metadata','-1','-movflags','+faststart',str(partial)]
    print('Encoding full video; original timing and audio preserved',flush=True)
    subprocess.run(command,check=True)
    after=video_info(partial)
    assert (after['width'],after['height'])==(width,height)
    assert after['frames']==info['frames'],(after['frames'],info['frames'])
    assert abs(after['fps']-info['fps'])<.001
    # Decode EVERY output frame rather than relying on container metadata alone.
    cap=cv2.VideoCapture(str(partial));decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(height,width)
        decoded+=1
    cap.release()
    assert decoded==info['frames'],(decoded,info['frames'])
    assert digest(source)==original_sha,'Original input changed'
    partial.replace(output)
    # Source/edited comparison at the same frame indices; no detection boxes baked in.
    source_cap=cv2.VideoCapture(str(source));edited_cap=cv2.VideoCapture(str(output))
    indices=np.linspace(0,info['frames']-1,12,dtype=int)
    sheet=np.zeros((4*276,3*520,3),np.uint8)
    psnr=[]
    for i,index in enumerate(indices):
        source_cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,original=source_cap.read()
        edited_cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok2,edited=edited_cap.read()
        assert ok and ok2
        region=original[top:top+height,left:left+width]
        psnr.append(float(cv2.PSNR(region,edited)))
        pair=np.hstack([cv2.resize(region,(256,256)),cv2.resize(edited,(256,256))])
        y,x=(i//3)*276+20,(i%3)*520
        sheet[y:y+256,x:x+512]=pair
        cv2.putText(sheet,f'{index/info["fps"]:.1f}s  source ROI | encoded ROI',(x+3,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    source_cap.release();edited_cap.release()
    save_image(output.parent/'source-vs-prepared.jpg',sheet)
    manifest=dict(at=now(),source=str(source),source_sha256=original_sha,source_unchanged=True,
         output=str(output),output_sha256=digest(output),before=info,after=after,
         crop_xywh=[left,top,width,height],retained_source_area_fraction=width*height/(info['width']*info['height']),
         resampling=False,contrast_or_sharpening=False,stabilization=False,trimmed=False,
         frame_interpolation=False,audio='original track copied when present',
         encoding='H.264 libx264 CRF16 medium, yuv420p, faststart',all_frames_decoded=decoded,
         sample_crop_psnr_mean=float(np.mean(psnr)),sample_crop_psnr_min=min(psnr),
         model_preprocessing='unchanged: detector scales ROI to 960 content then adds 16px border; input992',
         original_frame_mapping='edited frame n equals source frame n, xy offset [0,0]',
         scope='derived pavement ROI video; not full-frame coverage or model accuracy validation',ffmpeg_command=command)
    write_json(output.parent/'preparation.json',manifest)
    print(manifest,flush=True)


if __name__=='__main__':main()
