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


@pytest.mark.parametrize('mode', ['installed', 'download', 'apt_fallback', 'unsupported'])
def test_cloudflared_bootstrap_paths_without_network(monkeypatch, tmp_path, mode):
    from types import SimpleNamespace
    import subprocess
    tree = ast.parse(CELL.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(CELL), 'exec'), ns)
    content = tmp_path / 'content'
    content.mkdir()
    (tmp_path / 'etc/apt/sources.list.d').mkdir(parents=True)
    def sandbox_path(path):
        return tmp_path / str(path).lstrip('/')
    ns['pathlib'] = SimpleNamespace(Path=sandbox_path)
    ns['platform'] = SimpleNamespace(machine=lambda: 'unsupported' if mode == 'unsupported' else 'x86_64')
    installed = False
    ns['shutil'] = SimpleNamespace(which=lambda name: '/usr/bin/cloudflared' if installed else None)
    calls = []
    def run(args, **kwargs):
        nonlocal installed
        calls.append(args)
        if args[0] == 'curl':
            if 'github.com' in args[-1] and mode == 'apt_fallback':
                raise subprocess.CalledProcessError(22, args)
            Path(args[args.index('-o') + 1]).write_bytes(b'fixture')
        if args[:2] == ['apt-get', 'install']:
            installed = True
        return SimpleNamespace(returncode=0)
    ns['subprocess'] = SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError)
    if mode == 'installed':
        (content / 'cloudflared').write_bytes(b'existing')
    if mode == 'unsupported':
        with pytest.raises(RuntimeError, match='Unsupported runtime architecture'):
            ns['ensure_cloudflared']()
        assert calls == []
        return
    result = ns['ensure_cloudflared']()
    assert not (content / 'cloudflared.download').exists()
    if mode == 'apt_fallback':
        assert result == sandbox_path('/usr/bin/cloudflared')
        repo = (tmp_path / 'etc/apt/sources.list.d/cloudflared.list').read_text()
        assert 'signed-by=/usr/share/keyrings/cloudflare-main.gpg' in repo
        assert 'https://pkg.cloudflare.com/cloudflared any main' in repo
        assert any(args[:2] == ['apt-get', 'install'] for args in calls)
    else:
        assert result == content / 'cloudflared'
        assert not any(args[0] == 'apt-get' for args in calls)
        assert sum(args[0] == 'curl' for args in calls) == (mode == 'download')
