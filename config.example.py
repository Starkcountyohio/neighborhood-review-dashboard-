"""
config.example.py — Template configuration for the Neighborhood Review Dashboard

SETUP INSTRUCTIONS:
  1. Copy this file and rename the copy to config.py
  2. Fill in your Oracle connection details in DB_CONFIG
  3. config.py is excluded from git (see .gitignore) — your credentials stay local
"""

# ── Oracle Database Connection ────────────────────────────────────────────────
# Use the same username/password you use in SQL Developer.
# No Oracle Instant Client required (oracledb thin mode).
#
# DSN format: "hostname:port/service_name"
# The DSN below is the Stark County IasWorld production server — leave as-is
# unless you're pointing at a different environment (test, dev, etc.).

DB_CONFIG = {
    "user":     "your_oracle_username",
    "password": "your_oracle_password",
    "dsn":      "iaswdbs.starkcountyohio.gov:1521/IASW",
}

# ── IasWorld Schema ───────────────────────────────────────────────────────────
# Table prefix including trailing dot. Common values: "IASWORLD."  "PTIUSER."  ""
SCHEMA = "IASWORLD."

# ── Assessment Year ───────────────────────────────────────────────────────────
TAX_YEAR = "2026"

# ── Sale Date Range ───────────────────────────────────────────────────────────
SALE_YEAR_START = 2022
SALE_YEAR_END   = 2025   # sales < 1/1/2026 for 2027 triennial analysis

# ── Neighborhood Prefix Filter ────────────────────────────────────────────────
# Only show neighborhoods whose code starts with one of these prefixes.
# RS = Residential Single-family, RC = Residential Condo
NBHD_PREFIXES = ("RS", "RC", "AG")

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

# ── Excess Frontage Analysis ─────────────────────────────────────────────────
# Percentile threshold for the Normal/excess boundary.
# Only lots above this percentile of frontage are considered "excess."
# Example: 85 → top 15% reviewed.  90 → top 10% reviewed (more conservative).
EXCESS_FRONT_PERCENTILE = 85

# Maximum coefficient of variation (SD ÷ median) allowed before suppressing
# adjustment recommendations.  Set to a high value (e.g. 99) to disable.
EXCESS_FRONT_MAX_CV = 1.5

# ── IAAO Acceptable Ranges ───────────────────────────────────────────────────
IAAO_MEDIAN_RATIO_MIN = 0.90
IAAO_MEDIAN_RATIO_MAX = 1.10
IAAO_COD_MIN          = 5.0
IAAO_COD_MAX          = 15.0
IAAO_PRD_MIN          = 0.98
IAAO_PRD_MAX          = 1.03
IAAO_PRB_MIN          = -0.05
IAAO_PRB_MAX          =  0.05
IAAO_VEI_MIN          = -0.05
IAAO_VEI_MAX          =  0.05
