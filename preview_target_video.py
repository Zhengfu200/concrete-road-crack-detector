"""Open the prepared real pavement video with the currently selected unvalidated candidate."""
import subprocess,sys
from common import ROOT,read_json

if __name__=='__main__':
    video=ROOT/'outputs/road-video-prepared-20261027/实际路面视频_检测适配版.mp4'
    if not video.is_file():raise SystemExit(f'处理视频不存在：{video}')
    config=read_json(ROOT/'configs/preview-improved.json')
    if not config or not config.get('preview_args'):raise SystemExit('缺少当前模型预览配置')
    raise SystemExit(subprocess.call([sys.executable,*config['preview_args'],'--source',str(video)],cwd=ROOT))
