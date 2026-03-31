"""
config.example.py — Template configuration for the Neighborhood Review Dashboard

SETUP INSTRUCTIONS:
  1. Copy this file and rename the copy to config.py
  2. Fill in your Oracle connection details in DB_CONFIG
  3. config.py is excluded from git (see .gitignore) — your credentials stay local
"""

# ── Oracle Database Connection ────────────────────────────────────────────────
# Use the same server/credentials as SQL Developer.
# DSN format: "hostname:port/service_name"
# No Oracle Instant Client required (oracledb thin mode).

DB_CONFIG = {
    "user":     "your_oracle_username",
    "password": "your_oracle_password",
    "dsn":      "hostname:1521/service_name",
}

# ── IasWorld Schema ───────────────────────────────────────────────────────────
# Table prefix including trailing dot. Common values: "IASWORLD."  "PTIUSER."  ""
SCHEMA = "IASWORLD."

# ── Assessment Year ───────────────────────────────────────────────────────────
TAX_YEAR = "2026"

# ── Sale Date Range ───────────────────────────────────────────────────────────
SALE_YEAR_START = 2024
SALE_YEAR_END   = 2026

# ── Property Class Filter ─────────────────────────────────────────────────────
# IasWorld PARDAT.CLASS values for residential and condo.
PROPERTY_CLASSES = ("R", "C")

# ── Arm's Length Sale Filter ──────────────────────────────────────────────────
# SALEVAL values on the SALES table that identify arm's length transactions.
# Stark County: '0' and '1' are arm's length.
ARMS_LENGTH_SALEVAL = ("0", "1")

# ── DWELDAT Area Field ────────────────────────────────────────────────────────
# DWELDAT column for finished living area (sq ft).
# Common IasWorld field names: "SFLA"  "TLA"  "BLDGAREA"
AREA_FIELD = "sfla"

# ── IAAO Acceptable Ranges ───────────────────────────────────────────────────
IAAO_MEDIAN_RATIO_MIN = 0.90
IAAO_MEDIAN_RATIO_MAX = 1.10
IAAO_COD_MIN          = 5.0
IAAO_COD_MAX          = 15.0
IAAO_PRD_MIN          = 0.98
IAAO_PRD_MAX          = 1.03
IAAO_PRB_MIN          = -0.05
IAAO_PRB_MAX          =  0.05
