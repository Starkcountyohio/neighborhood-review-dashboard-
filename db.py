"""
db.py — Database connection and query functions
Connects to Tyler IasWorld (Oracle) using python-oracledb in thin mode.
No Oracle Instant Client required.

Tables queried:
  PARDAT   — parcel master record
  DWELDAT  — residential dwelling characteristics
  VALDAT   — appraised and assessed values
  SALES    — sales/transfer records

Field notes for IasWorld (verify against your schema if queries fail):
  PARDAT.TAXYR     — 4-char string assessment year, e.g. '2026'
  PARDAT.DEACTIVAT — NULL on active records; non-NULL means deleted/inactive
  PARDAT.NBHD      — neighborhood code
  PARDAT.CLASS     — property class (R=residential, C=condo, etc.)
  PARDAT.LUC       — land use code
  PARDAT.ADRNO     — house number
  PARDAT.ADRSTR    — street name
  PARDAT.ADRSUF    — street suffix (may be NULL)
  DWELDAT.CARD     — improvement card number; card=1 is the main dwelling
  DWELDAT.SFLA     — finished living area in sq ft (may be TLA or BLDGAREA)
  DWELDAT.GRADE    — grade/quality code
  DWELDAT.CDU      — condition/desirability/utility code
  DWELDAT.YRBLT    — year built
  DWELDAT.RMBED    — bedroom count
  DWELDAT.FIXBATH  — full bath count
  DWELDAT.FIXHALF  — half bath count
  VALDAT.LLINE     — value line; lline=1 is the primary improvement value
  VALDAT.APRTOT    — total appraised value
  VALDAT.APRLAND   — appraised land value
  VALDAT.APRBLDG   — appraised building value
  VALDAT.ASRTOT    — total assessed value
  SALES.SALEDT     — sale date
  SALES.PRICE      — sale price
  SALES.SALETYPE   — sale type / arm's length flag
  SALES.INSTRUNO   — instrument number (unique sale identifier)
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
    ARMS_LENGTH_SALETYPE,
    AREA_FIELD,
)


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
          AND  deactivat IS NULL
          AND  class     IN ({class_list})
        ORDER BY nbhd
    """
    df = _run(sql, {"tax_year": tax_year})
    return df["nbhd"].tolist()


@st.cache_data(ttl=300)
def get_parcels(nbhd: str, tax_year: str) -> pd.DataFrame:
    """
    Return all active parcels for a neighborhood with dwelling and value data.
    One row per parcel (DWELDAT card=1, VALDAT lline=1).
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
                   END                       AS address,
            p.class,
            p.luc,
            d.yrblt,
            d.{AREA_FIELD}                  AS sfla,
            d.grade,
            d.cdu,
            d.rmbed,
            d.fixbath,
            d.fixhalf,
            v.aprland,
            v.aprbldg,
            v.aprtot,
            v.asrtot
        FROM       {SCHEMA}pardat  p
        LEFT JOIN  {SCHEMA}dweldat d
               ON  d.parid     = p.parid
              AND  d.taxyr     = :tax_year
              AND  d.card      = 1
              AND  d.deactivat IS NULL
        LEFT JOIN  {SCHEMA}valdat  v
               ON  v.parid     = p.parid
              AND  v.taxyr     = :tax_year
              AND  v.lline     = 1
              AND  v.deactivat IS NULL
        WHERE  p.nbhd      = :nbhd
          AND  p.taxyr     = :tax_year
          AND  p.deactivat IS NULL
        ORDER BY p.parid
    """
    return _run(sql, {"nbhd": nbhd, "tax_year": tax_year})


@st.cache_data(ttl=300)
def get_sales(nbhd: str) -> pd.DataFrame:
    """
    Return arm's length sales within the configured date range for a neighborhood.
    Values (aprtot, etc.) are pulled from the current TAX_YEAR assessment —
    this is correct for ratio study: compare current values against sale prices.

    Multiple sales of the same parcel within the date range are all returned;
    each is an independent observation in the ratio study.
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
                   END                       AS address,
            s.saledt,
            s.price,
            s.instruno,
            v.aprtot,
            v.aprland,
            v.aprbldg,
            d.{AREA_FIELD}                  AS sfla,
            d.grade,
            d.cdu,
            d.yrblt
        FROM       {SCHEMA}sales   s
        JOIN       {SCHEMA}pardat  p
               ON  p.parid     = s.parid
              AND  p.taxyr     = :tax_year
              AND  p.deactivat IS NULL
        JOIN       {SCHEMA}valdat  v
               ON  v.parid     = s.parid
              AND  v.taxyr     = :tax_year
              AND  v.lline     = 1
              AND  v.deactivat IS NULL
        LEFT JOIN  {SCHEMA}dweldat d
               ON  d.parid     = s.parid
              AND  d.taxyr     = :tax_year
              AND  d.card      = 1
              AND  d.deactivat IS NULL
        WHERE  p.nbhd      = :nbhd
          AND  s.saledt   >= TO_DATE(:start_dt, 'YYYY-MM-DD')
          AND  s.saledt    < TO_DATE(:end_dt,   'YYYY-MM-DD')
          AND  s.price     > 0
          AND  s.saletype  = :saletype
          AND  s.deactivat IS NULL
        ORDER BY s.saledt DESC
    """
    return _run(sql, {
        "nbhd":     nbhd,
        "tax_year": TAX_YEAR,
        "start_dt": f"{SALE_YEAR_START}-01-01",
        "end_dt":   f"{SALE_YEAR_END + 1}-01-01",
        "saletype": ARMS_LENGTH_SALETYPE,
    })
