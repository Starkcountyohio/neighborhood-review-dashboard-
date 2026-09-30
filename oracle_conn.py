"""
oracle_conn.py — Shared Oracle connection helper
Opens a python-oracledb thin-mode connection to the Tyler cloud iasWorld DB
using DB_CONFIG from config.py. No Oracle Instant Client or wallet required.

All code should connect through connect() so every connection gets the same
TLS settings.

Password lookup order: DB_CONFIG["password"] in config.py, the IAS_PASSWORD
environment variable, Windows Credential Manager (saved once with
`py set_password.py`), then an interactive prompt (console only).
"""

import getpass
import os
import ssl
import sys

import keyring
import oracledb

from config import DB_CONFIG

KEYRING_SERVICE = "neighborhood-review-dashboard"


def _ssl_context() -> ssl.SSLContext:
    """
    Default TLS context (certificate + hostname verification stay ON) with
    VERIFY_X509_STRICT cleared. Python 3.13+ enables strict X.509 checks by
    default, and Tyler's certificate chain intermittently fails them.
    """
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)
    return ctx


def _password() -> str:
    pw = (
        DB_CONFIG.get("password")
        or os.environ.get("IAS_PASSWORD", "")
        or keyring.get_password(KEYRING_SERVICE, DB_CONFIG["user"])
    )
    if not pw and sys.stdin and sys.stdin.isatty():
        pw = getpass.getpass(f"Oracle password for {DB_CONFIG['user']}: ")
    if not pw:
        raise RuntimeError(
            "No Oracle password saved. Run `py set_password.py` once in the "
            "project folder to store it in Windows Credential Manager."
        )
    return pw


def connect() -> oracledb.Connection:
    """Open a new Oracle connection using DB_CONFIG."""
    params = {**DB_CONFIG, "password": _password()}
    return oracledb.connect(**params, ssl_context=_ssl_context())
