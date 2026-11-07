"""Paste this entire file into one Colab cell. Rerun after runtime replacement."""
import hmac
import json
import pathlib
import re
import secrets
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Stop only resources created by this cell, never unrelated runtime processes.
if '_tebdil_bridge' in globals():
    old_server, old_tunnel = _tebdil_bridge
    old_server.shutdown()
    old_server.server_close()
    if old_tunnel is not None:
        old_tunnel.terminate()
        try:
            old_tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            old_tunnel.kill()
            old_tunnel.wait()

TOKEN = secrets.token_urlsafe(32)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, data):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def authorized(self):
        if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + TOKEN):
            self.reply(403, {'error': 'unauthorized'})
            return False
        return True

    def do_GET(self):
        if self.authorized():
            self.reply(200, {'status': 'ready', 'protocol': 1})

    def do_POST(self):
        if not self.authorized():
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 65536:
                raise ValueError('body must be 1..65536 bytes')
            self.connection.settimeout(15)
            body = json.loads(self.rfile.read(length))
            cmd = body['cmd']
            timeout = body.get('timeout', 20)
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError('cmd must be a nonempty string')
            if not isinstance(timeout, (int, float)) or not 0 < timeout <= 30:
                raise ValueError('timeout must be 0..30 seconds; launch long jobs in background')
        except (ValueError, KeyError, TypeError, TimeoutError) as exc:
            self.reply(400, {'error': str(exc)})
            return
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            self.reply(200, {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        except subprocess.TimeoutExpired:
            self.reply(408, {'error': 'command timed out; inspect processes before retrying'})

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
_tebdil_bridge = (server, None)
binary = pathlib.Path('/content/cloudflared')
if not binary.exists():
    temporary = binary.with_suffix('.download')
    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', temporary)
    temporary.chmod(0o700)
    temporary.replace(binary)

ready = False
for attempt in range(3):
    log_path = pathlib.Path('/content/tebdil-tunnel.log')
    with log_path.open('w') as log:
        tunnel = subprocess.Popen([str(binary), 'tunnel', '--url', f'http://127.0.0.1:{server.server_port}', '--no-autoupdate'], stdout=log, stderr=log)
    _tebdil_bridge = (server, tunnel)
    for _ in range(60):
        if tunnel.poll() is not None:
            break
        time.sleep(1)
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', log_path.read_text())
        if not match:
            continue
        url = match.group(0)
        try:
            request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + TOKEN})
            with urllib.request.urlopen(request, timeout=3) as response:
                ready = json.load(response).get('status') == 'ready'
        except (OSError, ValueError):
            continue
        if ready:
            break
    if ready:
        print('TEBDIL bridge ready. Share privately with your agent:')
        print(json.dumps({'url': url, 'token': TOKEN, 'protocol': 1}))
        break
    tunnel.terminate()
    try:
        tunnel.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tunnel.kill()
        tunnel.wait()
else:
    server.shutdown()
    server.server_close()
    raise RuntimeError('Tunnel failed after 3 attempts. Rerun cell. Log: ' + log_path.read_text()[-3000:])
