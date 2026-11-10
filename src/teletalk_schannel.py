"""Opt-in verified Windows Schannel fallback for public official Teletalk HTTPS.

No shell, redirects, HTTP URLs, cookies, TLS bypass, or third-party hosts.
Invoked only when Requests fails with an SSL exception and only after explicitly
setting TELERETAIN_SCHANNEL_FALLBACK=1. Response is requests-like for OfficialFetcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile

from requests.structures import CaseInsensitiveDict

ALLOWED = {"www.teletalk.com.bd", "teletalk.com.bd"}
FORBIDDEN_PATH_PARTS = ('/login', '/logout', '/signup', '/sign-up', '/register',
                        '/purchase', '/checkout', '/payment', '/cart',
                        '/my-account', '/api/', '/online-recharge', '/order')
BLOCKED_SUFFIXES = ('.pdf', '.zip', '.png', '.jpg', '.jpeg', '.svg', '.gif',
                    '.webp', '.exe', '.css', '.js')

class SchannelTransportError(RuntimeError):
    pass

@dataclass
class CurlResponse:
    status_code: int
    headers: CaseInsensitiveDict
    body: bytes
    encoding: str = "utf-8"

    def iter_content(self, chunk_size=32_768):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start:start + chunk_size]

    def close(self):
        pass


def _parse_headers(raw: bytes):
    # A proxy may precede the origin response with HTTP/1.1 200 Connection established.
    # The *last* HTTP response block is authoritative.
    sections = re.split(rb"\r?\n\r?\n", raw)
    valid = [s for s in sections if re.match(rb"^HTTP/(?:1\.[01]|2|3)(?:\.0)?\s+\d{3}(?:\s|$)", s)]
    if not valid:
        raise SchannelTransportError("No valid HTTP response status in verified curl headers")
    lines = valid[-1].decode("iso-8859-1", "replace").splitlines()
    match = re.match(r"^HTTP/\S+\s+(\d{3})(?:\s|$)", lines[0])
    if match is None:
        raise SchannelTransportError("Unrecognized final HTTP status")
    status = int(match.group(1))
    hdrs = CaseInsensitiveDict()
    for line in lines[1:]:
        if ':' not in line or line.startswith((' ', '\t')):
            continue
        name, value = line.split(':', 1)
        if re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
            hdrs[name] = value.strip()
    return status, hdrs


def fetch_via_schannel(url: str, *, headers: dict[str, str], max_bytes: int=2_000_000,
                       which=shutil.which, run=subprocess.run, platform: str|None=None):
    """Perform strictly verified HTTPS GET without following redirects.

    Curl return code 0 is required. A 404 robots.txt is reported to OfficialFetcher
    and is not automatically interpreted as a robots prohibition. A 403/429 is
    returned unmodified for OfficialFetcher to reject (never bypassed).
    """
    from urllib.parse import urlsplit
    actual_platform = sys.platform if platform is None else platform
    if actual_platform != 'win32':
        raise SchannelTransportError('Schannel fallback is only supported on Windows')
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.hostname not in ALLOWED or parts.port is not None
        or parts.username or parts.password or parts.query or parts.fragment
        or not parts.path.startswith('/') or len(url) > 400
        or any(piece in parts.path.lower() for piece in FORBIDDEN_PATH_PARTS)
        or parts.path.lower().endswith(BLOCKED_SUFFIXES)):
        raise SchannelTransportError('Only exact official HTTPS document URLs are allowed')
    if max_bytes < 1024 or max_bytes > 2_000_000:
        raise SchannelTransportError('Unsupported response size limit')
    curl = which('curl.exe')
    if not curl:
        raise SchannelTransportError('curl.exe not found on PATH')
    # Require libcurl's Windows TLS backend, not an unverified/silent replacement.
    try:
        version = run([curl, '-q', '--version'], capture_output=True, text=True,
                      timeout=8, shell=False, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SchannelTransportError('Unable to inspect curl.exe TLS backend') from exc
    if version.returncode != 0 or 'Schannel' not in version.stdout:
        raise SchannelTransportError('curl.exe must report the Schannel TLS backend')
    try:
        with tempfile.TemporaryDirectory(prefix='teleretain-official-') as temp:
            header_file = str(Path(temp)/'headers.txt')
            body_file = str(Path(temp)/'body.bin')
            cmd = [curl, '-q', '--silent', '--show-error', '--proto', '=https',
                   '--tlsv1.2', '--connect-timeout', '8', '--max-time', '28',
                   '--max-filesize', str(max_bytes), '--max-redirs', '0',
                   '--dump-header', header_file, '--output', body_file,
                   '--header', 'User-Agent: ' + headers.get('User-Agent', 'TeleRetainAI-PublicAcademicDemo/1.0'),
                   '--header', 'Accept: ' + headers.get('Accept', 'text/html,application/xhtml+xml'),
                   '--', url]
            result = run(cmd, capture_output=True, text=True, timeout=35,
                         shell=False, check=False)
            if result.returncode != 0:
                # curl error 60/35 means certificate or handshake could not be verified.
                raise SchannelTransportError('curl.exe HTTPS failed (exit '
                                               + str(result.returncode) + '): '
                                               + (result.stderr or '').replace('\n', ' ')[:200])
            if not Path(header_file).exists() or not Path(body_file).exists():
                raise SchannelTransportError('Verified curl request returned no header/body files')
            status, response_headers = _parse_headers(Path(header_file).read_bytes())
            if Path(body_file).stat().st_size > max_bytes:
                raise SchannelTransportError('Official response exceeded maximum allowed bytes')
            body = Path(body_file).read_bytes()
            return CurlResponse(status, response_headers, body)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SchannelTransportError('Curl subprocess or file operation failed') from exc
