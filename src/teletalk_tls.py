"""Strict TLS configuration for official Teletalk public HTML.

Default is Requests/certifi. On a trusted Windows PC, an explicitly opted-in
Windows CA mode uses truststore's native verification *only* for official hosts.
No certificate/hostname verification is disabled; no unofficial hosts are allowed.
"""
from __future__ import annotations

import os
import ssl
import sys

import requests
from requests.adapters import HTTPAdapter

OFFICIAL_HOSTS = (
    "https://www.teletalk.com.bd/",
    "https://teletalk.com.bd/",
)
TLS_ENV = "TELERETAIN_TLS_TRUST"


class WindowsTrustAdapter(HTTPAdapter):
    """Requests connection pool with native Windows chain verification."""

    def __init__(self, context: ssl.SSLContext, **kwargs):
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise ValueError("Strict TLS certificate and hostname checks are required")
        self._context = context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **kwargs):
        kwargs["ssl_context"] = self._context
        return super().proxy_manager_for(proxy, **kwargs)


def official_session(*, mode: str | None = None) -> requests.Session:
    selected = (mode if mode is not None else os.environ.get(TLS_ENV, "certifi")).strip().lower()
    if selected not in ("certifi", "windows"):
        raise ValueError(f"Invalid {TLS_ENV}; allowed: certifi, windows")
    session = requests.Session()
    if selected == "windows":
        if sys.platform != "win32":
            raise RuntimeError("Windows trust mode requires Windows; use certifi on other platforms")
        try:
            import truststore
        except ImportError as exc:
            raise RuntimeError("Windows CA mode requires: python -m pip install truststore") from exc
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        adapter = WindowsTrustAdapter(ctx)
        for prefix in OFFICIAL_HOSTS:
            session.mount(prefix, adapter)
    return session
