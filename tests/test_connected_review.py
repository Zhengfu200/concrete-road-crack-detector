import json
from http.server import HTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from test_connected_annotations import workspace
from connected_review import handler_for


def test_local_review_api_locks_held_out_data_and_validates_write_session(workspace, monkeypatch, capsys):
    work, rows = workspace
    server = HTTPServer(('127.0.0.1',0), lambda *a: None)
    port = server.server_address[1]
    server.RequestHandlerClass = handler_for(work, 'test-session', port)
    thread = Thread(target=server.serve_forever, daemon=True);thread.start()
    base = f'http://127.0.0.1:{port}'
    try:
        with urlopen(base+'/api/state') as response:
            state = json.load(response)
        assert not state['frozen'] and len(state['items'])==50
        import connected_review
        monkeypatch.setattr(connected_review, 'ExclusiveHTTPServer', lambda *a: pytest.fail('Already-running workspace must reuse its server'))
        monkeypatch.setattr('sys.argv', ['connected_review.py','--work',str(work),'--port',str(port),'--no-browser'])
        connected_review.main()
        assert '已运行' in capsys.readouterr().out
        with pytest.raises(HTTPError) as exc:
            urlopen(base+'/api/item?id='+rows[50]['id'])
        assert exc.value.code == 404
        body=json.dumps({'reviewer':'test human'}).encode()
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base+'/api/freeze',data=body,headers={'Content-Type':'application/json'}))
        assert exc.value.code == 403
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base+'/api/freeze',data=body,headers={'Content-Type':'application/json',
                    'X-Review-Token':'test-session','Origin':base}))
        assert exc.value.code == 400
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base+'/api/freeze',data=body,headers={'Content-Type':'application/json',
                    'X-Review-Token':'test-session','Origin':'https://unrelated.invalid'}))
        assert exc.value.code == 403
    finally:
        server.shutdown();thread.join(timeout=5);server.server_close()
