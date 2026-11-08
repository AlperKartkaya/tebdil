"""Paste this entire file into one Google Colab cell. Rerun after runtime replacement."""
import hmac
import json
import pathlib
import platform
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Stop only resources created by an earlier run of this cell.
if '_tebdil_bridge' in globals():
    old_server, old_tunnel = _tebdil_bridge
    old_server.shutdown()
    old_server.server_close()
    if old_tunnel is not None and old_tunnel.poll() is None:
        old_tunnel.terminate()
        try:
            old_tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            old_tunnel.kill()
            old_tunnel.wait()


def ensure_cloudflared():
    """Use an installed binary, download with retries, or fall back to official APT."""
    binary = pathlib.Path('/content/cloudflared')
    candidates = [binary]
    system_binary = shutil.which('cloudflared')
    if system_binary:
        candidates.append(pathlib.Path(system_binary))

    for candidate in candidates:
        if candidate.is_file():
            try:
                subprocess.run([str(candidate), '--version'], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10)
                return candidate
            except (OSError, subprocess.SubprocessError):
                pass

    architecture = platform.machine().lower()
    if architecture in ('x86_64', 'amd64'):
        filename = 'cloudflared-linux-amd64'
    elif architecture in ('aarch64', 'arm64'):
        filename = 'cloudflared-linux-arm64'
    else:
        raise RuntimeError(f'Unsupported runtime architecture: {architecture}')

    temporary = binary.with_suffix('.download')
    url = f'https://github.com/cloudflare/cloudflared/releases/latest/download/{filename}'
    try:
        subprocess.run([
            'curl', '-fL', '--retry', '4', '--retry-all-errors',
            '--retry-delay', '3', '--connect-timeout', '20',
            '--max-time', '120', '-o', str(temporary), url
        ], check=True, timeout=650)
        temporary.chmod(0o700)
        subprocess.run([str(temporary), '--version'], check=True,
                       stdout=subprocess.DEVNULL, timeout=10)
        temporary.replace(binary)
        return binary
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'GitHub download failed ({exc}); trying the official Cloudflare APT repository...')
    finally:
        temporary.unlink(missing_ok=True)

    # Google Colab normally runs as root on a Debian/Ubuntu-based image.
    key_dir = pathlib.Path('/usr/share/keyrings')
    key_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'curl', '-fL', '--retry', '4', '--retry-delay', '3',
        '--connect-timeout', '20', '--max-time', '120',
        '-o', str(key_dir / 'cloudflare-main.gpg'),
        'https://pkg.cloudflare.com/cloudflare-main.gpg'
    ], check=True, timeout=550)
    pathlib.Path('/etc/apt/sources.list.d/cloudflared.list').write_text(
        'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] '
        'https://pkg.cloudflare.com/cloudflared any main\n'
    )
    subprocess.run(['apt-get', 'update', '-qq'], check=True, timeout=240)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'cloudflared'],
                   check=True, timeout=240)
    installed = shutil.which('cloudflared')
    if not installed:
        raise RuntimeError('APT finished but cloudflared was not found in PATH')
    return pathlib.Path(installed)


# Install before opening the server: a failed download will not leak a server.
binary = ensure_cloudflared()
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
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    text=True, timeout=timeout)
            self.reply(200, {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
        except subprocess.TimeoutExpired:
            self.reply(408, {'error': 'command timed out; inspect processes before retrying'})


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
_tebdil_bridge = (server, None)

ready = False
log_path = pathlib.Path('/content/tebdil-tunnel.log')
for attempt in range(3):
    with log_path.open('w') as log:
        tunnel = subprocess.Popen([
            str(binary), 'tunnel', '--url',
            f'http://127.0.0.1:{server.server_port}', '--no-autoupdate'
        ], stdout=log, stderr=log)
    _tebdil_bridge = (server, tunnel)
    for _ in range(60):
        if tunnel.poll() is not None:
            break
        time.sleep(1)
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com',
                          log_path.read_text())
        if not match:
            continue
        url = match.group(0)
        try:
            request = urllib.request.Request(
                url, headers={'Authorization': 'Bearer ' + TOKEN})
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
    if tunnel.poll() is None:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()
else:
    server.shutdown()
    server.server_close()
    _tebdil_bridge = (server, None)
    raise RuntimeError('Tunnel failed after 3 attempts. Log: ' +
                       log_path.read_text()[-3000:])
