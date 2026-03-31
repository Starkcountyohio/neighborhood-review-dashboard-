"""
db.py — Database connection and query functions
Connects to Tyler IasWorld (Oracle) using python-oracledb in thin mode.
No Oracle Instant Client required.

Tables queried:
  PARDAT   — parcel master record
  DWELDAT  — residential dwelling characteristics
  APRVAL   — appraised values (replaces VALDAT in this IasWorld version)
  SALES    — sales/transfer records

Field notes (Stark County IasWorld confirmed):
  PARDAT.TAXYR     — NUMBER, assessment year
  PARDAT.CUR       — 'Y' = current record
  PARDAT.DEACTIVAT — NULL on active records
  PARDAT.NBHD      — neighborhood code
  PARDAT.CLASS     — property class
  PARDAT.LUC       — land use code
  DWELDAT.CARD     — improvement card; card=1 is the main dwelling
  DWELDAT.TAXYR    — NUMBER
  DWELDAT.CUR      — 'Y' = current record
  DWELDAT.SFLA     — finished living area (sq ft)
  DWELDAT.GRADE    — grade/quality code
  DWELDAT.CDU      — condition/desirability/utility
  DWELDAT.YRBLT    — year built
  APRVAL.TAXYR     — NUMBER
  APRVAL.SEQ       — sequence; SEQ=1 is the primary value record
  APRVAL.CUR       — 'Y' = current record
  APRVAL.APRTOT    — total appraised value
  APRVAL.APRLAND   — appraised land value
  APRVAL.APRBLDG   — appraised building value
  SALES.SALEDT     — sale date (no TAXYR on SALES table)
  SALES.PRICE      — sale price
  SALES.SALETYPE   — sale type / arm's length flag
  SALES.MKTVALID   — market validity flag (alternate arm's length indicator)
  SALES.CUR        — 'Y' = current record
  SALES.INSTRUNO   — instrument number
"""

import oracledb
import pandas as pd
import streamlit as st

from config import (
    DB_CONFIG,
    SCHEMA,
    TAX_YEAR,
    SALE_YEAR_START,
    SALE_YEAR_END,
    PROPERTY_CLASSES,
    ARMS_LENGTH_SALEVAL,
    AREA_FIELD,
)

# TAXYR is a NUMBER in this IasWorld instance — bind as int
_TAX_YEAR_INT = int(TAX_YEAR)


# ── Connection ────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_connection() -> oracledb.Connection:
    """
    Create and cache an Oracle connection (thin mode).
    Cached for the lifetime of the Streamlit server process.
    Click 'Refresh Data' in the sidebar to re-establish if the connection drops.
    """
    try:
        return oracledb.connect(**DB_CONFIG)
    except oracledb.Error as exc:
        st.error(
            f"**Database connection failed.**\n\n"
            f"{exc}\n\n"
            f"Check the DB_CONFIG settings in `config.py` and confirm the "
            f"server is reachable from this machine."
        )
        st.stop()


def _run(sql: str, params: dict) -> pd.DataFrame:
    """Execute a parameterised SQL query and return a DataFrame."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [col[0].lower() for col in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    except oracledb.Error as exc:
        # Connection may have timed out — clear cache and retry once
        st.cache_resource.clear()
        conn2 = _get_connection()
        with conn2.cursor() as cur:
            cur.execute(sql, params)
            cols = [col[0].lower() for col in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)


# ── Queries ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_neighborhoods(tax_year: str) -> list[str]:
    """Return a sorted list of neighborhood codes for the given tax year."""
    class_list = ", ".join(f"'{c}'" for c in PROPERTY_CLASSES)
    sql = f"""
        SELECT DISTINCT nbhd
        FROM   {SCHEMA}pardat
        WHERE  taxyr     = :tax_year
          AND  cur       = 'Y'
          AND  deactivat IS NULL
          AND  class     IN ({class_list})
        ORDER BY nbhd
    """
    df = _run(sql, {"tax_year": int(tax_year)})
    return df["nbhd"].tolist()


@st.cache_data(ttl=300)
def get_parcels(nbhd: str, tax_year: str) -> pd.DataFrame:
    """
    Return all active parcels for a neighborhood with dwelling and value data.
    One row per parcel (DWELDAT card=1, APRVAL seq=1).
    Assessed value = APRTOT * 0.35 (Ohio 35% assessment ratio).
    """
    sql = f"""
        SELECT
            p.parid,
            TRIM(p.adrno)
                || ' ' || TRIM(p.adrstr)
                || CASE
                       WHEN p.adrsuf IS NOT NULL
                       THEN ' ' || TRIM(p.adrsuf)
                       ELSE ''
                   END                              AS address,
            p.class,
            p.luc,
            d.yrblt,
            d.{AREA_FIELD}                         AS sfla,
            d.grade,
            d.cdu,
            d.rmbed,
            d.fixbath,
            d.fixhalf,
            av.aprland,
            av.aprbldg,
            av.aprtot,
            ROUND(av.aprtot * 0.35, 0)             AS asrtot
        FROM       {SCHEMA}pardat  p
        LEFT JOIN  {SCHEMA}dweldat d
               ON  d.parid     = p.parid
              AND  d.taxyr     = :tax_year
              AND  d.card      = 1
              AND  d.cur       = 'Y'
              AND  d.deactivat IS NULL
        LEFT JOIN  {SCHEMA}aprval  av
               ON  av.parid    = p.parid
              AND  av.taxyr    = :tax_year
              AND  av.cur      = 'Y'
              AND  av.deactivat IS NULL
        WHERE  p.nbhd      = :nbhd
          AND  p.taxyr     = :tax_year
          AND  p.cur       = 'Y'
          AND  p.deactivat IS NULL
        ORDER BY p.parid
    """
    return _run(sql, {"nbhd": nbhd, "tax_year": int(tax_year)})


@st.cache_data(ttl=300)
def get_sales(nbhd: str) -> pd.DataFrame:
    """
    Return arm's length sales within the configured date range for a neighborhood.
    Appraised values come from the current TAX_YEAR APRVAL record —
    this is correct for ratio study: current values vs. historical sale prices.

    Multiple sales of the same parcel within the date range are all returned;
    each is an independent observation in the ratio study.

    Note: SALES has no TAXYR column — it is a sale event table keyed by
    PARID + SALEDT + SEQ. The PARDAT/APRVAL joins use TAX_YEAR for current values.
    """
    sql = f"""
        SELECT
            p.parid,
            TRIM(p.adrno)
                || ' ' || TRIM(p.adrstr)
                || CASE
                       WHEN p.adrsuf IS NOT NULL
                       THEN ' ' || TRIM(p.adrsuf)
                       ELSE ''
                   END                              AS address,
            s.saledt,
            s.price,
            s.instruno,
            av.aprtot,
            av.aprland,
            av.aprbldg,
            d.{AREA_FIELD}                         AS sfla,
            d.grade,
            d.cdu,
            d.yrblt
        FROM       {SCHEMA}sales   s
        JOIN       {SCHEMA}pardat  p
               ON  p.parid     = s.parid
              AND  p.taxyr     = :tax_year
              AND  p.cur       = 'Y'
              AND  p.deactivat IS NULL
        JOIN       {SCHEMA}aprval  av
               ON  av.parid    = s.parid
              AND  av.taxyr    = :tax_year
              AND  av.cur      = 'Y'
              AND  av.deactivat IS NULL
        LEFT JOIN  {SCHEMA}dweldat d
               ON  d.parid     = s.parid
              AND  d.taxyr     = :tax_year
              AND  d.card      = 1
              AND  d.cur       = 'Y'
              AND  d.deactivat IS NULL
        WHERE  p.nbhd      = :nbhd
          AND  s.saledt   >= TO_DATE(:start_dt, 'YYYY-MM-DD')
          AND  s.saledt    < TO_DATE(:end_dt,   'YYYY-MM-DD')
          AND  s.price     > 0
          AND  s.saleval  IN ({", ".join(f"'{v}'" for v in ARMS_LENGTH_SALEVAL)})
          AND  s.cur       = 'Y'
          AND  s.deactivat IS NULL
        ORDER BY s.saledt DESC
    """
    return _run(sql, {
        "nbhd":     nbhd,
        "tax_year": _TAX_YEAR_INT,
        "start_dt": f"{SALE_YEAR_START}-01-01",
        "end_dt":   f"{SALE_YEAR_END + 1}-01-01",
    })
