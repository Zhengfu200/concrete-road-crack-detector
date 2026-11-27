"""Launch the selected, explicitly unvalidated preview candidate."""
import subprocess
import sys

from common import ROOT, read_json


if __name__ == '__main__':
    config = read_json(ROOT/'configs/preview-improved.json')
    if not config:
        raise SystemExit('优化对照尚未完成，请先等待训练和验证结束。')
    raise SystemExit(subprocess.call([sys.executable, *config['preview_args']], cwd=ROOT))
