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

from oracle_conn import connect
from config import (
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
        return connect()
    except (oracledb.Error, RuntimeError) as exc:
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

@st.cache_data(ttl=3600)
def get_infl_codes() -> dict:
    """
    Return a dict mapping INFL1 code → description from the IasWorld code master.

    Queries CODMST where CODETBL = 'INFL'.
    Result is cached for 1 hour (codes rarely change mid-session).
    Returns an empty dict on failure so callers degrade gracefully.
    """
    sql = f"""
        SELECT code, TRIM(decode) AS decode
        FROM   codmst
        WHERE  codetbl = 'INFL'
        ORDER BY code
    """
    try:
        df = _run(sql, {})
        return dict(zip(df["code"].astype(str).str.strip(), df["decode"]))
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_neighborhoods(tax_year: str) -> list[str]:
    """Return a sorted list of RS/RC neighborhood codes for the given tax year."""
    prefix_filter = " OR ".join(f"nbhd LIKE '{p}%'" for p in NBHD_PREFIXES)
    sql = f"""
        SELECT DISTINCT nbhd
        FROM   pardat
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
            d.effyr,
            d.{AREA_FIELD}                         AS sfla,
            d.grade,
            d.cdu,
            d.rmbed,
            d.fixbath,
            d.fixhalf,
            d.heat,
            d.pctcomplete,
            d.fundep,
            d.funrsn,
            d.ecndep,
            d.ecnrsn,
            d.user11,
            d.adjrcnld,
            d.rcnld,
            d.adjfact,
            av.aprland,
            av.aprbldg,
            av.aprtot
        FROM       pardat  p
        LEFT JOIN  dweldat d
               ON  d.parid     = p.parid
              AND  d.taxyr     = :tax_year
              AND  d.cur       = 'Y'
              AND  d.deactivat IS NULL
              AND  d.card      = (
                       SELECT MIN(d2.card)
                       FROM   dweldat d2
                       WHERE  d2.parid     = p.parid
                         AND  d2.taxyr     = :tax_year
                         AND  d2.cur       = 'Y'
                         AND  d2.deactivat IS NULL
                   )
        LEFT JOIN  aprval  av
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
    Return arm's length sales from SALES_SUMMARY for a neighborhood.

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
            p.luc,
            ss.saledt,
            ss.price,
            ss.transno                             AS instruno,
            ss.aprtot,
            ss.aprland,
            ss.aprbldg,
            ss.res_sfla                            AS sfla,
            d.grade,
            d.cdu,
            d.yrblt
        FROM       sales_summary ss
        JOIN       pardat        p
               ON  p.parid      = ss.sum_parid
              AND  p.taxyr      = :tax_year
              AND  p.cur        = 'Y'
              AND  p.deactivat  IS NULL
        LEFT JOIN  dweldat       d
               ON  d.parid      = ss.sum_parid
              AND  d.taxyr      = :tax_year
              AND  d.cur        = 'Y'
              AND  d.deactivat  IS NULL
              AND  d.card       = (
                       SELECT MIN(d2.card)
                       FROM   dweldat d2
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


@st.cache_data(ttl=300)
def get_land_data(nbhd: str, tax_year: str) -> pd.DataFrame:
    """
    Return all LAND records for a neighborhood joined to PARDAT and APRVAL.

    Each row is one land line (LLINE) per parcel. Parcels may have multiple
    lines (e.g. lline=1 primary, lline=2 excess land area).

    Key fields:
      FRONACT  — actual frontage (ft); used for excess frontage analysis
      FRONEFF  — effective frontage
      SF       — land square footage
      INFL1    — influence code 1 (e.g. excess frontage code)
      INFLU    — influence adjustment 1 (e.g. -30 = -30% reduction)
      INFL2    — influence code 2
      INFLU2   — influence adjustment 2
      PRICE    — land unit price for this line
      APRLAND  — total appraised land value from APRVAL
    """
    sql = f"""
        SELECT
            l.parid,
            TRIM(p.adrno)
                || ' ' || TRIM(p.adrstr)
                || CASE
                       WHEN p.adrsuf IS NOT NULL
                       THEN ' ' || TRIM(p.adrsuf)
                       ELSE ''
                   END                             AS address,
            p.luc,
            p.calcacres,
            l.lline,
            l.ltype,
            l.fronact,
            l.sf,
            l.infl1,
            l.influ,
            l.infl2,
            l.influ2,
            l.price                               AS land_price,
            av.aprland
        FROM       land    l
        JOIN       pardat  p
               ON  p.parid     = l.parid
              AND  p.taxyr     = :tax_year
              AND  p.cur       = 'Y'
              AND  p.deactivat IS NULL
        LEFT JOIN  aprval  av
               ON  av.parid    = l.parid
              AND  av.taxyr    = :tax_year
              AND  av.cur      = 'Y'
              AND  av.deactivat IS NULL
        WHERE  p.nbhd      = :nbhd
          AND  l.taxyr     = :tax_year
          AND  l.cur       = 'Y'
          AND  l.deactivat IS NULL
        ORDER BY l.parid, l.lline
    """
    return _run(sql, {"nbhd": nbhd, "tax_year": int(tax_year)})


@st.cache_data(ttl=3600)
def get_nbhd_summary(tax_year: str) -> pd.DataFrame:
    """
    Return one row per neighborhood with key assessment health statistics.
    Used for the All Neighborhoods overview tab.

    Columns returned:
      nbhd          — neighborhood code
      n_parcels     — total active parcel count
      median_av     — median total appraised value
      n_sales       — arm's length sale count (configured date range)
      median_ratio  — median assessment ratio (APRTOT / PRICE)
      cod           — coefficient of dispersion of ratios
      prd           — price-related differential
    """
    prefix_filter = " OR ".join(f"p.nbhd LIKE '{p}%'" for p in NBHD_PREFIXES)
    saleval_list  = ", ".join(f"'{v}'" for v in ARMS_LENGTH_SALEVAL)

    sql = f"""
        WITH parcel_stats AS (
            SELECT
                p.nbhd,
                COUNT(*)                                                           AS n_parcels,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY av.aprtot), 0)  AS median_av
            FROM   pardat  p
            LEFT JOIN aprval av
                   ON  av.parid     = p.parid
                  AND  av.taxyr     = :tax_year
                  AND  av.cur       = 'Y'
                  AND  av.deactivat IS NULL
            WHERE  p.taxyr     = :tax_year
              AND  p.cur       = 'Y'
              AND  p.deactivat IS NULL
              AND  ({prefix_filter})
            GROUP BY p.nbhd
        ),
        sale_ratios AS (
            SELECT
                p.nbhd,
                ss.aprtot / ss.price  AS ratio,
                ss.aprtot             AS aprtot,
                ss.price              AS price
            FROM   sales_summary ss
            JOIN   pardat        p
                   ON  p.parid      = ss.sum_parid
                  AND  p.taxyr      = :tax_year
                  AND  p.cur        = 'Y'
                  AND  p.deactivat  IS NULL
            WHERE  ss.saledt  >= TO_DATE(:start_dt, 'YYYY-MM-DD')
              AND  ss.saledt   < TO_DATE(:end_dt,   'YYYY-MM-DD')
              AND  ss.price    > 0
              AND  ss.saleval  IN ({saleval_list})
              AND  ({prefix_filter})
        ),
        sale_medians AS (
            SELECT
                nbhd,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ratio) AS med_ratio
            FROM  sale_ratios
            GROUP BY nbhd
        ),
        sale_stats AS (
            SELECT
                r.nbhd,
                COUNT(*)                                                                    AS n_sales,
                ROUND(m.med_ratio, 4)                                                      AS median_ratio,
                ROUND(
                    AVG(ABS(r.ratio - m.med_ratio) / NULLIF(m.med_ratio, 0)) * 100,
                    2
                )                                                                           AS cod,
                ROUND(
                    AVG(r.ratio) / NULLIF(SUM(r.aprtot) / NULLIF(SUM(r.price), 0), 0),
                    4
                )                                                                           AS prd
            FROM  sale_ratios   r
            JOIN  sale_medians  m ON m.nbhd = r.nbhd
            GROUP BY r.nbhd, m.med_ratio
        )
        SELECT
            ps.nbhd,
            ps.n_parcels,
            ps.median_av,
            NVL(ss.n_sales, 0)  AS n_sales,
            ss.median_ratio,
            ss.cod,
            ss.prd
        FROM  parcel_stats ps
        LEFT JOIN sale_stats ss ON ss.nbhd = ps.nbhd
        ORDER BY ps.nbhd
    """
    return _run(sql, {
        "tax_year": int(tax_year),
        "start_dt": f"{SALE_YEAR_START}-01-01",
        "end_dt":   f"{SALE_YEAR_END + 1}-01-01",
    })


@st.cache_data(ttl=300)
def get_market_values(nbhd: str, tax_year: str) -> pd.DataFrame:
    """
    Return market model results from SUBJCOMP for all parcels in a neighborhood.

    One row per parcel (subject property). Includes the calculated market value,
    FCC (ratio of SD to average adjusted sale price), comparability distances for
    each of the 5 comparables, and excess-adjustment flags.

    Credibility fields:
      CALCMKTVAL — market model estimated value
      FCC        — credible when between 1 and 5
      DIST1–5    — comparability index per comp; credible when ≥ 3 are under 50
      EXADJ1–5   — excess-adjustment flag per comp (>50% adj); credible when ≤ 3 flagged
    """
    sql = f"""
        SELECT
            sc.parid,
            sc.calcmktval,
            sc.fcc,
            sc.dist1,  sc.dist2,  sc.dist3,  sc.dist4,  sc.dist5,
            sc.exadj1, sc.exadj2, sc.exadj3, sc.exadj4, sc.exadj5
        FROM       subjcomp sc
        JOIN       pardat   p
               ON  p.parid     = sc.parid
              AND  p.taxyr     = :tax_year
              AND  p.cur       = 'Y'
              AND  p.deactivat IS NULL
        WHERE  p.nbhd     = :nbhd
          AND  sc.taxyr   = :tax_year
          AND  sc.cur     = 'Y'
    """
    return _run(sql, {"nbhd": nbhd, "tax_year": int(tax_year)})


@st.cache_data(ttl=3600)
def get_grade_model_data() -> pd.DataFrame:
    """
    Fetch all residential parcels with dwelling and sketch features for the
    grade creep OLS model.  Cached for 1 hour — runs once per session.

    Returns one row per parcel with:
      parid, nbhd, address, grade, yrblt, sfla, rmbed, fixbath, fixhalf, heat,
      addition_count (sketch lines > 0), garage_sqft, amenity_count.
    """
    prefix_list = ", ".join(f"'{p}'" for p in NBHD_PREFIXES)

    garage_codes  = "'GF','GB','CF','CB'"
    amenity_codes = "'DB','DF','PB','PF','LB','LF','IB','IF','OB','OF','YB','YF','BB','BF'"

    sql_parcels = f"""
        SELECT
            p.parid,
            p.nbhd,
            TRIM(p.adrno)
                || ' ' || TRIM(p.adrstr)
                || CASE
                       WHEN p.adrsuf IS NOT NULL
                       THEN ' ' || TRIM(p.adrsuf)
                       ELSE ''
                   END          AS address,
            d.grade,
            d.yrblt,
            d.{AREA_FIELD}     AS sfla,
            d.rmbed,
            d.fixbath,
            d.fixhalf,
            d.heat
        FROM       pardat  p
        JOIN       dweldat d
               ON  d.parid     = p.parid
              AND  d.taxyr     = :tax_year
              AND  d.cur       = 'Y'
              AND  d.deactivat IS NULL
              AND  d.card      = (
                       SELECT MIN(d2.card)
                       FROM   dweldat d2
                       WHERE  d2.parid     = p.parid
                         AND  d2.taxyr     = :tax_year
                         AND  d2.cur       = 'Y'
                         AND  d2.deactivat IS NULL
                   )
        WHERE  p.taxyr     = :tax_year
          AND  p.cur       = 'Y'
          AND  p.deactivat IS NULL
          AND  SUBSTR(p.nbhd, 1, 2) IN ({prefix_list})
          AND  d.grade        IS NOT NULL
          AND  d.yrblt        IS NOT NULL
          AND  d.{AREA_FIELD} IS NOT NULL
          AND  d.{AREA_FIELD} > 0
    """

    sql_addn = f"""
        SELECT
            a.parid,
            COUNT(*)  AS addition_count,
            SUM(CASE WHEN a."LOWER"  IN ({garage_codes})
                       OR a."FIRST"  IN ({garage_codes})
                       OR a."SECOND" IN ({garage_codes})
                       OR a."THIRD"  IN ({garage_codes})
                     THEN a.area ELSE 0 END) AS garage_sqft,
            SUM(CASE WHEN a."LOWER"  IN ({amenity_codes})
                       OR a."FIRST"  IN ({amenity_codes})
                       OR a."SECOND" IN ({amenity_codes})
                       OR a."THIRD"  IN ({amenity_codes})
                     THEN 1 ELSE 0 END) AS amenity_count
        FROM addn a
        WHERE a.taxyr     = :tax_year
          AND a.card      = 1
          AND a.lline     > 0
          AND a.cur       = 'Y'
          AND a.deactivat IS NULL
        GROUP BY a.parid
    """

    parcels = _run(sql_parcels, {"tax_year": _TAX_YEAR_INT})
    addn    = _run(sql_addn,    {"tax_year": _TAX_YEAR_INT})

    merged = parcels.merge(
        addn[["parid", "addition_count", "garage_sqft", "amenity_count"]],
        on="parid", how="left",
    )
    merged["addition_count"] = merged["addition_count"].fillna(0)
    merged["garage_sqft"]    = merged["garage_sqft"].fillna(0)
    merged["amenity_count"]  = merged["amenity_count"].fillna(0)
    return merged


@st.cache_data(ttl=300)
def get_frontage_summary(tax_year: str) -> pd.DataFrame:
    """
    Return one row per neighborhood with front-foot parcel count, median
    frontage, and standard deviation — used for the All Neighborhoods overview.

    Only neighborhoods matching NBHD_PREFIXES are included.
    Only front-foot land lines (ltype != 'A') are considered.
    Parcels with zero total frontage are excluded.
    """
    prefix_filter = " OR ".join(f"p.nbhd LIKE '{p}%'" for p in NBHD_PREFIXES)
    sql = f"""
        WITH parcel_fronts AS (
            SELECT
                p.nbhd,
                l.parid,
                SUM(l.fronact) AS total_fronact
            FROM   land    l
            JOIN   pardat  p
                   ON  p.parid     = l.parid
                  AND  p.taxyr     = :tax_year
                  AND  p.cur       = 'Y'
                  AND  p.deactivat IS NULL
            WHERE  l.taxyr     = :tax_year
              AND  l.cur       = 'Y'
              AND  l.deactivat IS NULL
              AND  l.ltype     != 'A'
              AND  ({prefix_filter})
            GROUP BY p.nbhd, l.parid
            HAVING SUM(l.fronact) > 0
        )
        SELECT
            nbhd,
            COUNT(*)    AS n_parcels,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_fronact), 1)
                        AS median_front,
            ROUND(STDDEV(total_fronact), 1)
                        AS sd_front
        FROM  parcel_fronts
        GROUP BY nbhd
        ORDER BY nbhd
    """
    return _run(sql, {"tax_year": int(tax_year)})
