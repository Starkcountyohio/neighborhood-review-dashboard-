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
    NBHD_PREFIXES,
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
    """Return a sorted list of RS/RC neighborhood codes for the given tax year."""
    prefix_filter = " OR ".join(f"nbhd LIKE '{p}%'" for p in NBHD_PREFIXES)
    sql = f"""
        SELECT DISTINCT nbhd
        FROM   {SCHEMA}pardat
        WHERE  taxyr     = :tax_year
          AND  cur       = 'Y'
          AND  deactivat IS NULL
          AND  ({prefix_filter})
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
              AND  d.cur       = 'Y'
              AND  d.deactivat IS NULL
              AND  d.card      = (
                       SELECT MIN(d2.card)
                       FROM   {SCHEMA}dweldat d2
                       WHERE  d2.parid     = p.parid
                         AND  d2.taxyr     = :tax_year
                         AND  d2.cur       = 'Y'
                         AND  d2.deactivat IS NULL
                   )
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
    Return arm's length sales from IASWORLD.SALES_SUMMARY for a neighborhood.

    SALES_SUMMARY is a staff-reviewed, locked view used for market model analysis.
    It is the authoritative source for arm's length sale records.

    Appraised values (APRTOT, APRLAND, APRBLDG) come from the current TAX_YEAR
    APRVAL record so all sales are compared against consistent current-year values.

    SALES_SUMMARY.RES_SFLA is used for square footage (already calculated in view).
    Multiple sales of the same parcel are each independent ratio study observations.
    """
    saleval_list = ", ".join(f"'{v}'" for v in ARMS_LENGTH_SALEVAL)
    sql = f"""
        SELECT
            ss.sum_parid                           AS parid,
            TRIM(p.adrno)
                || ' ' || TRIM(p.adrstr)
                || CASE
                       WHEN p.adrsuf IS NOT NULL
                       THEN ' ' || TRIM(p.adrsuf)
                       ELSE ''
                   END                             AS address,
            ss.saledt,
            ss.price,
            ss.transno                             AS instruno,
            av.aprtot,
            av.aprland,
            av.aprbldg,
            ss.res_sfla                            AS sfla,
            d.grade,
            d.cdu,
            d.yrblt
        FROM       {SCHEMA}sales_summary ss
        JOIN       {SCHEMA}pardat        p
               ON  p.parid      = ss.sum_parid
              AND  p.taxyr      = :tax_year
              AND  p.cur        = 'Y'
              AND  p.deactivat  IS NULL
        JOIN       {SCHEMA}aprval        av
               ON  av.parid     = ss.sum_parid
              AND  av.taxyr     = :tax_year
              AND  av.cur       = 'Y'
              AND  av.deactivat IS NULL
        LEFT JOIN  {SCHEMA}dweldat       d
               ON  d.parid      = ss.sum_parid
              AND  d.taxyr      = :tax_year
              AND  d.cur        = 'Y'
              AND  d.deactivat  IS NULL
              AND  d.card       = (
                       SELECT MIN(d2.card)
                       FROM   {SCHEMA}dweldat d2
                       WHERE  d2.parid     = ss.sum_parid
                         AND  d2.taxyr     = :tax_year
                         AND  d2.cur       = 'Y'
                         AND  d2.deactivat IS NULL
                   )
        WHERE  p.nbhd       = :nbhd
          AND  ss.saledt   >= TO_DATE(:start_dt, 'YYYY-MM-DD')
          AND  ss.saledt    < TO_DATE(:end_dt,   'YYYY-MM-DD')
          AND  ss.price     > 0
          AND  ss.saleval   IN ({saleval_list})
        ORDER BY ss.saledt DESC
    """
    return _run(sql, {
        "nbhd":     nbhd,
        "tax_year": _TAX_YEAR_INT,
        "start_dt": f"{SALE_YEAR_START}-01-01",
        "end_dt":   f"{SALE_YEAR_END + 1}-01-01",
    })
