"""Exercise the cell's actual HTTP handler without installing or starting a tunnel."""
import ast
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

CELL = Path(__file__).parents[1] / '.agents/skills/colab-runtime-bridge/scripts/cell.py'

@pytest.fixture
def bridge():
    tree = ast.parse(CELL.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    namespace = {'TOKEN': 'test-secret'}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(CELL), 'exec'), namespace)
    server = namespace['ThreadingHTTPServer'](('127.0.0.1', 0), namespace['Handler'])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join()


def test_auth_and_health(bridge):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(bridge)
    assert error.value.code == 403
    request = urllib.request.Request(bridge, headers={'Authorization': 'Bearer test-secret'})
    with urllib.request.urlopen(request) as response:
        assert json.load(response)['status'] == 'ready'


@pytest.mark.parametrize('body,code', [({'cmd': 'printf bridge-ok'}, 200), ({'cmd': ''}, 400), ({'cmd': 'true', 'timeout': 3600}, 400), ([], 400)])
def test_commands(bridge, body, code):
    request = urllib.request.Request(bridge, data=json.dumps(body).encode(), headers={'Authorization': 'Bearer test-secret'})
    if code != 200:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == code
    else:
        with urllib.request.urlopen(request) as response:
            assert json.load(response)['stdout'] == 'bridge-ok'
