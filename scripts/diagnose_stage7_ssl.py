"""Safe and detailed HTTPS diagnostic. No keys, no private data, no SSL bypass.

Run: python scripts/diagnose_stage7_ssl.py
Install optional Windows CA support: python -m pip install truststore
"""
from __future__ import annotations
import os
import ssl
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
import certifi
from src.teletalk_tls import official_session

URLS = (
    "https://www.python.org/",
    "https://www.teletalk.com.bd/robots.txt",
    "https://teletalk.com.bd/robots.txt",
)


def check(url: str, session: requests.Session) -> bool:
    try:
        response = session.get(url, timeout=(6, 12), allow_redirects=False,
                               headers={"User-Agent": "TeleRetainAI-SafeTLS-Diagnostic/1.0"},
                               stream=True)
        try:
            print(f"  TLS: PASS | HTTP {response.status_code} (HTTP status is not an offer-verification result)")
            return True
        finally:
            response.close()
    except requests.exceptions.SSLError as exc:
        print("  TLS: FAIL |", str(exc).replace('\n', ' ')[:1400])
    except requests.RequestException as exc:
        print("  NETWORK: FAIL |", type(exc).__name__, str(exc).replace('\n', ' ')[:500])
    return False


def main() -> int:
    print('TeleRetain AI | HTTPS/certificate diagnostic (read-only; no customer data)')
    print('Python:', sys.version.split()[0], '| Requests:', requests.__version__)
    print('OS:', sys.platform, '| Certifi:', certifi.__version__)
    print('TLS context: CERT_REQUIRED and hostname checking; no verification bypass')
    print('Certificate-related environment flags (values withheld):')
    for key in ('REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE','SSL_CERT_FILE','HTTPS_PROXY','HTTP_PROXY'):
        print('  ', key, 'SET' if os.environ.get(key) else 'not set')
    try:
        regular = official_session(mode='certifi')
        print('\n-- Default Requests/certifi CA --')
        normal = {url: check(url, regular) for url in URLS}
    finally:
        if 'regular' in locals(): regular.close()
    native = None
    if sys.platform == 'win32':
        print('\n-- Windows native CA (truststore) --')
        try:
            native = official_session(mode='windows')
            native_results = {url: check(url,native) for url in URLS}
            if any(native_results.get(url,False) and not normal.get(url,False) for url in URLS[1:]):
                print('SUGGESTION: Windows mode appears to help; run: export TELERETAIN_TLS_TRUST=windows')
            elif all(not native_results.get(url,False) for url in URLS[1:]):
                print('SUGGESTION: Both CA modes fail on Teletalk. Check Windows/browser certificate error, system time, proxy or server.')
        except RuntimeError as exc:
            print('Windows CA test unavailable:', exc)
        finally:
            if native is not None: native.close()
    print('\nNOTE: This diagnostic does not prove robots permission, correct HTML, or customer eligibility.')
    print('Run the original official-site verifier only after a TLS mode succeeds.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
