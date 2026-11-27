"""Local review UI for connected V2; source files remain read-only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import socket
import sys
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from urllib.error import URLError
import webbrowser

from common import ROOT, digest, read_json
from connected_annotations import (WORK, annotation_path, default_objects, freeze_rules,
                                   frozen, save_annotation)


class ExclusiveHTTPServer(HTTPServer):
    # Windows SO_REUSEADDR otherwise permits two review writers on the same port.
    allow_reuse_address = False

    def server_bind(self):
        if sys.platform == 'win32':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def handler_for(work, token, port):
    work = Path(work)
    rows = {r['id']: r for r in read_json(work/'source-index.json')['rows']}

    class Handler(BaseHTTPRequestHandler):
        def send(self, content, kind='application/json; charset=utf-8', status=200):
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False).encode()
            elif isinstance(content, str):
                content = content.encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def allowed_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

        def do_GET(self):
            if not self.allowed_host():
                return self.send({'error': '仅限本机访问'}, status=403)
            req = urlparse(self.path)
            if req.path == '/':
                html = (ROOT/'ui/connected-review.html').read_text(encoding='utf-8')
                return self.send(html.replace('__SESSION_TOKEN__', token), 'text/html; charset=utf-8')
            if req.path == '/api/state':
                pilot = read_json(work/'pilot.json')['items']
                is_frozen = frozen(work)
                visible = list(rows) if is_frozen else [x['id'] for x in pilot]
                cats = {x['id']: x['category'] for x in pilot}
                states = []
                for image_id in visible:
                    row = rows[image_id]
                    rec = read_json(annotation_path(work, image_id), {})
                    states.append({'id': image_id, 'split': row['split'], 'category': cats.get(image_id, 'full'),
                                   'status': rec.get('status', 'unseen'), 'reviewer_type': rec.get('reviewer_type')})
                return self.send({'items': states, 'frozen': is_frozen, 'workspace': str(work.resolve()),
                                  'rules': (work/'rules.md').read_text(encoding='utf-8'),
                                  'total_source': len(rows), 'pilot_ids': [x['id'] for x in pilot]})
            image_id = parse_qs(req.query).get('id', [''])[0]
            pilot_ids = {x['id'] for x in read_json(work/'pilot.json')['items']}
            if image_id not in rows or (image_id not in pilot_ids and not frozen(work)):
                return self.send({'error': '此阶段图片不可用'}, status=404)
            row = rows[image_id]
            if req.path == '/api/item':
                rec = read_json(annotation_path(work, image_id), {'revision': 0, 'status': 'draft',
                    'objects': default_objects(row), 'excluded': [], 'reason': '', 'background_confirmed': False})
                return self.send({'id': image_id, 'split': row['split'], 'polygons': row['polygons'], 'record': rec})
            if req.path == '/api/image':
                path = Path(row['image'])
                return self.send(path.read_bytes(), 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg')
            return self.send({'error': '入口不存在'}, status=404)

        def do_POST(self):
            if not self.allowed_host() or self.headers.get('X-Review-Token') != token:
                return self.send({'error': '会话校验失败，请从本机复核页面操作'}, status=403)
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}:
                return self.send({'error': '来源不匹配'}, status=403)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2_000_000:
                    raise ValueError('请求大小无效')
                value = json.loads(self.rfile.read(size))
                if self.path == '/api/save':
                    result = save_annotation(work, value['id'], value['annotation'], value['reviewer'], 'human')
                elif self.path == '/api/freeze':
                    result = freeze_rules(work, value['reviewer'])
                else:
                    raise ValueError('入口不存在')
                self.send(result)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.send({'error': str(exc)}, status=400)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=WORK)
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--no-browser', action='store_true')
    args = p.parse_args()
    if not read_json(args.work/'pilot.json'):
        raise SystemExit('缺少试标清单')
    url = f'http://127.0.0.1:{args.port}'
    try:
        with urlopen(url+'/api/state', timeout=3) as response:
            existing = json.load(response)
    except (URLError, OSError, ValueError):
        existing = None
    if existing is not None:
        if existing.get('workspace') != str(args.work.resolve()):
            raise SystemExit('端口已被其他程序或复核工作区占用')
        if not args.no_browser:
            webbrowser.open(url)
        print(f'复核入口已运行：{url}', flush=True)
        return
    try:
        server = ExclusiveHTTPServer(('127.0.0.1', args.port), handler_for(args.work, secrets.token_urlsafe(32), args.port))
    except OSError:
        try:
            with urlopen(url+'/api/state', timeout=3) as response:
                state = json.load(response)
            if state.get('workspace') != str(args.work.resolve()):
                raise ValueError('端口已被其他程序或复核工作区占用')
        except Exception as exc:
            raise SystemExit(f'复核入口无法启动：{exc}')
        if not args.no_browser:
            webbrowser.open(url)
        print(f'复核入口已运行：{url}', flush=True)
        return
    print(f'裂缝整体标注复核：{url}；Ctrl+C关闭。仅写入独立V2复核工作区。', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
