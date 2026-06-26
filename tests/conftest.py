"""Shared pytest configuration.

Blocks outbound network access for the whole test suite. No test needs the
network — ICESat-2/SlideRule requests are pre-fetched into parquet fixtures and
contextily basemaps are disabled in plotting tests. A stray basemap fetch
(``contextily.add_basemap``) otherwise blocks on an uninterruptible socket read
and hangs CI for the full job timeout. With this guard such a call fails fast
and loudly instead, so the offending test is obvious rather than a silent hang.

Loopback connections are still allowed so local IPC (e.g. joblib worker
backends, which contextily uses) keeps working.
"""

import socket

import pytest

_real_connect = socket.socket.connect
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, (tuple, list)) else address
    if host not in _LOOPBACK_HOSTS:
        raise RuntimeError(
            f"Network access is disabled during tests (attempted connection to "
            f"{host!r}). Tests must use local fixtures; disable basemaps with "
            f"add_basemap=False / omit contextily kwargs."
        )
    return _real_connect(self, address)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Fail fast on any non-loopback socket connection (autouse, every test)."""
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
