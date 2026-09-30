"""
set_password.py — Save your Oracle password in Windows Credential Manager.

Run once (and again whenever the password changes):
    py set_password.py

The dashboard and scripts then read it from there via oracle_conn.connect().
"""

import getpass

import keyring
import oracledb

from config import DB_CONFIG
from oracle_conn import KEYRING_SERVICE, _ssl_context

user = DB_CONFIG["user"]
pw = getpass.getpass(f"Oracle password for {user} (input hidden): ")
if not pw:
    raise SystemExit("No password entered; nothing saved.")

print("Testing login …")
try:
    oracledb.connect(**{**DB_CONFIG, "password": pw}, ssl_context=_ssl_context()).close()
except oracledb.Error as exc:
    raise SystemExit(f"Login failed, password NOT saved:\n{exc}")

keyring.set_password(KEYRING_SERVICE, user, pw)
print(f"Saved. Password for {user} is stored in Windows Credential Manager.")
