"""
oracle_conn.py — Shared Oracle connection helper
Opens a python-oracledb thin-mode connection to the Tyler cloud iasWorld DB
using DB_CONFIG from config.py. No Oracle Instant Client or wallet required.

All code should connect through connect() so every connection gets the same
TLS settings.

Password lookup order: DB_CONFIG["password"] in config.py, then the
IAS_PASSWORD environment variable, then an interactive prompt (console only).
"""

import getpass
import os
import ssl
import sys

import oracledb

from config import DB_CONFIG


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
    pw = DB_CONFIG.get("password") or os.environ.get("IAS_PASSWORD", "")
    if not pw and sys.stdin and sys.stdin.isatty():
        pw = getpass.getpass(f"Oracle password for {DB_CONFIG['user']}: ")
    if not pw:
        raise RuntimeError(
            "No Oracle password: set it in config.py or the IAS_PASSWORD "
            "environment variable."
        )
    return pw


def connect() -> oracledb.Connection:
    """Open a new Oracle connection using DB_CONFIG."""
    params = {**DB_CONFIG, "password": _password()}
    return oracledb.connect(**params, ssl_context=_ssl_context())
