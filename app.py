"""
app.py — Neighborhood Review Dashboard
Stark County Auditor, Mass Appraisal Analytics

Run with:
    streamlit run app.py

First-time setup:
    pip install -r requirements.txt
    # Edit config.py with your Oracle connection details, then run above.
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    TAX_YEAR,
    SALE_YEAR_START,
    SALE_YEAR_END,
    IAAO_COD_MIN,
    IAAO_COD_MAX,
    IAAO_PRD_MIN,
    IAAO_PRD_MAX,
    IAAO_PRB_MIN,
    IAAO_PRB_MAX,
    IAAO_MEDIAN_RATIO_MIN,
    IAAO_MEDIAN_RATIO_MAX,
)
from db import get_neighborhoods, get_parcels, get_sales
from stats import compute_ratio_stats, compute_uniformity_stats, flag_outliers


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neighborhood Review Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Neighborhood Review Dashboard")
st.caption("Stark County Auditor — Mass Appraisal Analytics")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")

    tax_year = st.selectbox(
        "Tax Year",
        ["2026", "2025", "2024"],
        index=["2026", "2025", "2024"].index(TAX_YEAR) if TAX_YEAR in ["2026", "2025", "2024"] else 0,
    )

    with st.spinner("Loading neighborhoods..."):
        try:
            nbhd_list = get_neighborhoods(tax_year)
        except Exception as exc:
            st.error(f"Could not load neighborhoods: {exc}")
            st.stop()

    if not nbhd_list:
        st.warning(f"No neighborhoods found for tax year {tax_year}.")
        st.stop()

    selected_nbhd = st.selectbox("Neighborhood", nbhd_list)

    sd_threshold = st.slider(
        "Outlier SD Threshold",
        min_value=1.5,
        max_value=3.0,
        value=2.0,
        step=0.5,
        help=(
            "Parcels/sales are flagged as outliers when their value falls more than "
            "this many standard deviations from the neighborhood mean.\n\n"
            "2.0 SD ≈ flags the most extreme ~5% of a normal distribution."
        ),
    )

    st.divider()

    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"Values: tax year {tax_year}  \n"
        f"Sales: {SALE_YEAR_START}–{SALE_YEAR_END}"
    )


# ── Load Data ─────────────────────────────────────────────────────────────────

with st.spinner(f"Loading NBHD {selected_nbhd}..."):
    try:
        parcels_df = get_parcels(selected_nbhd, tax_year)
        sales_df   = get_sales(selected_nbhd)
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        st.stop()

if parcels_df.empty:
    st.warning(f"No parcels found for neighborhood **{selected_nbhd}**.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# UPPER PANEL — All Parcels
# ════════════════════════════════════════════════════════════════════════════

st.header(f"All Parcels — NBHD {selected_nbhd}")
st.caption(
    "Uniformity statistics are computed on assessed value per sq ft across all "
    "parcels with area data. Outliers are parcels whose AV/SqFt falls more than "
    f"{sd_threshold:.1f} SD from the neighborhood mean."
)

# ── Uniformity Stats Row ──────────────────────────────────────────────────────

u_stats = compute_uniformity_stats(parcels_df)

if "error" not in u_stats:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Parcels",    f"{u_stats['n']:,}")
    c2.metric("Median AV",        f"${u_stats['median_av']:,.0f}")
    c3.metric("Mean AV",          f"${u_stats['mean_av']:,.0f}")
    c4.metric(
        "Median AV/SqFt",
        f"${u_stats['median_av_per_sqft']:,.2f}",
    )
    c5.metric(
        "COD (AV/SqFt)",
        f"{u_stats['cod_av_per_sqft']:.2f}",
        help=(
            "Coefficient of Dispersion of assessed value per sq ft.\n"
            "Measures intra-neighborhood uniformity of the assessment roll "
            "across all parcels, independent of sales.\n"
            f"Based on {u_stats['n_with_area']:,} of {u_stats['n']:,} parcels with area data."
        ),
    )
    c6.metric(
        "COV (AV/SqFt)",
        f"{u_stats['cov_av_per_sqft']:.2f}%",
        help=(
            "Coefficient of Variation of assessed value per sq ft.\n"
            f"Based on {u_stats['n_with_area']:,} of {u_stats['n']:,} parcels with area data."
        ),
    )
else:
    st.warning(u_stats["error"])

# ── Build Upper Panel DataFrame ───────────────────────────────────────────────

disp = parcels_df.copy()

# Compute AV per sqft
area_ok = (disp["sfla"].fillna(0) > 0) & (disp["aprtot"].fillna(0) > 0)
disp["av_per_sqft"] = np.where(area_ok, disp["aprtot"] / disp["sfla"], np.nan)

# Outlier flag
disp["flag"] = ""
if area_ok.sum() >= 5:
    outlier_mask = flag_outliers(disp.loc[area_ok, "av_per_sqft"], sd_threshold)
    outlier_idx  = outlier_mask[outlier_mask].index
    disp.loc[outlier_idx, "flag"] = "OUTLIER"

# Column order and display names
UPPER_COLS = {
    "parid":       "Parcel ID",
    "address":     "Address",
    "class":       "Class",
    "luc":         "LUC",
    "yrblt":       "Yr Built",
    "sfla":        "Sqft",
    "grade":       "Grade",
    "cdu":         "CDU",
    "rmbed":       "Beds",
    "fixbath":     "Full Bath",
    "fixhalf":     "Half Bath",
    "aprland":     "Land AV",
    "aprbldg":     "Bldg AV",
    "aprtot":      "Total AV",
    "asrtot":      "Assessed",
    "av_per_sqft": "AV/SqFt",
    "flag":        "Flag",
}
show_cols = [c for c in UPPER_COLS if c in disp.columns]
disp_show = disp[show_cols].rename(columns=UPPER_COLS)

st.dataframe(
    disp_show,
    use_container_width=True,
    height=430,
    column_config={
        "Land AV":   st.column_config.NumberColumn(format="$%d"),
        "Bldg AV":   st.column_config.NumberColumn(format="$%d"),
        "Total AV":  st.column_config.NumberColumn(format="$%d"),
        "Assessed":  st.column_config.NumberColumn(format="$%d"),
        "AV/SqFt":   st.column_config.NumberColumn(format="$%.2f"),
        "Sqft":      st.column_config.NumberColumn(format="%d"),
        "Flag":      st.column_config.TextColumn(width="small"),
    },
)


# ════════════════════════════════════════════════════════════════════════════
# LOWER PANEL — Arm's Length Sales / Ratio Study
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.header(f"Arm's Length Sales ({SALE_YEAR_START}–{SALE_YEAR_END}) — NBHD {selected_nbhd}")
st.caption(
    f"Ratio = Total AV (tax year {tax_year}) ÷ Sale Price. "
    f"Outliers flagged where ratio is more than {sd_threshold:.1f} SD from the mean."
)

if sales_df.empty:
    st.info(
        f"No arm's length sales found for NBHD {selected_nbhd} "
        f"in {SALE_YEAR_START}–{SALE_YEAR_END}."
    )
    st.stop()

s_stats = compute_ratio_stats(sales_df)

# ── Ratio Study Stats Row ─────────────────────────────────────────────────────

if "error" in s_stats:
    st.warning(s_stats["error"])
else:
    st.subheader("Ratio Study Statistics")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric("Sales (n)", f"{s_stats['n']:,}")

    r2.metric(
        "Median Ratio",
        f"{s_stats['median_ratio']:.4f}",
        help=(
            f"95% CI: [{s_stats['median_ratio_ci'][0]:.4f},  {s_stats['median_ratio_ci'][1]:.4f}]\n\n"
            f"IAAO level of assessment standard: "
            f"{IAAO_MEDIAN_RATIO_MIN:.2f} – {IAAO_MEDIAN_RATIO_MAX:.2f}"
        ),
    )

    r3.metric("Mean Ratio",    f"{s_stats['mean_ratio']:.4f}")
    r4.metric("Weighted Mean", f"{s_stats['weighted_mean']:.4f}",
              help="Σ(Appraised Value) ÷ Σ(Sale Price)")

    r5, r6, r7, r8 = st.columns(4)

    r5.metric(
        "COD",
        f"{s_stats['cod']:.2f}",
        help=(
            f"95% CI: [{s_stats['cod_ci'][0]:.2f},  {s_stats['cod_ci'][1]:.2f}]\n\n"
            f"IAAO residential standard: {IAAO_COD_MIN} – {IAAO_COD_MAX}\n"
            "Lower is better. Measures uniformity of assessment ratios."
        ),
    )

    r6.metric("COV", f"{s_stats['cov']:.2f}%",
              help="Coefficient of Variation of assessment ratios.")

    r7.metric(
        "PRD",
        f"{s_stats['prd']:.4f}",
        help=(
            f"Price-Related Differential\n"
            f"IAAO standard: {IAAO_PRD_MIN} – {IAAO_PRD_MAX}\n"
            "> 1.03 → regressive (high-value properties under-assessed)\n"
            "< 0.98 → progressive (high-value properties over-assessed)"
        ),
    )

    r8.metric(
        "PRB",
        f"{s_stats['prb']:.4f}",
        help=(
            f"Price-Related Bias  (p = {s_stats['prb_pvalue']:.4f})\n"
            f"IAAO standard: {IAAO_PRB_MIN} to {IAAO_PRB_MAX}\n"
            "Negative → regressive (high-value under-assessed)\n"
            "Positive → progressive (high-value over-assessed)\n"
            "Each unit of x = one doubling of estimated market value."
        ),
    )

    # ── IAAO Flag Indicators ──────────────────────────────────────────────────

    flags = []

    med = s_stats["median_ratio"]
    if not (IAAO_MEDIAN_RATIO_MIN <= med <= IAAO_MEDIAN_RATIO_MAX):
        direction = "above" if med > IAAO_MEDIAN_RATIO_MAX else "below"
        flags.append(
            f"Median Ratio {med:.4f} is {direction} the IAAO acceptable range "
            f"({IAAO_MEDIAN_RATIO_MIN} – {IAAO_MEDIAN_RATIO_MAX}) — "
            f"assessment level needs adjustment."
        )

    cod = s_stats["cod"]
    if not (IAAO_COD_MIN <= cod <= IAAO_COD_MAX):
        flags.append(
            f"COD {cod:.2f} is outside the IAAO acceptable range "
            f"({IAAO_COD_MIN} – {IAAO_COD_MAX}) — "
            f"{'excessive dispersion' if cod > IAAO_COD_MAX else 'suspiciously low dispersion'}."
        )

    prd = s_stats["prd"]
    if not (IAAO_PRD_MIN <= prd <= IAAO_PRD_MAX):
        direction = "regressive" if prd > IAAO_PRD_MAX else "progressive"
        flags.append(
            f"PRD {prd:.4f} indicates {direction} vertical inequity "
            f"(IAAO standard: {IAAO_PRD_MIN} – {IAAO_PRD_MAX})."
        )

    prb = s_stats["prb"]
    if not (IAAO_PRB_MIN <= prb <= IAAO_PRB_MAX):
        direction = "regressive" if prb < IAAO_PRB_MIN else "progressive"
        flags.append(
            f"PRB {prb:.4f} (p = {s_stats['prb_pvalue']:.4f}) indicates "
            f"{direction} vertical inequity "
            f"(IAAO standard: {IAAO_PRB_MIN} to {IAAO_PRB_MAX})."
        )

    if flags:
        st.warning("**IAAO Performance Flags:**\n\n" + "\n\n".join(f"• {f}" for f in flags))
    else:
        st.success(
            f"All ratio study statistics are within IAAO acceptable ranges "
            f"(n = {s_stats['n']} sales)."
        )

# ── Sales Detail Grid ─────────────────────────────────────────────────────────

st.subheader("Sales Detail")

sales_disp = sales_df.copy()
sales_disp["ratio"] = sales_disp["aprtot"] / sales_disp["price"]

# Outlier flag on ratio
if len(sales_disp) >= 5:
    sales_disp["flag"] = np.where(
        flag_outliers(sales_disp["ratio"], sd_threshold),
        "OUTLIER",
        "",
    )
else:
    sales_disp["flag"] = ""

# Format sale date
if "saledt" in sales_disp.columns:
    sales_disp["saledt"] = pd.to_datetime(sales_disp["saledt"]).dt.date

SALES_COLS = {
    "parid":    "Parcel ID",
    "address":  "Address",
    "saledt":   "Sale Date",
    "price":    "Sale Price",
    "aprtot":   "Total AV",
    "ratio":    "Ratio",
    "aprland":  "Land AV",
    "aprbldg":  "Bldg AV",
    "sfla":     "Sqft",
    "grade":    "Grade",
    "cdu":      "CDU",
    "yrblt":    "Yr Built",
    "instruno": "Instrument No.",
    "flag":     "Flag",
}
show_sales = [c for c in SALES_COLS if c in sales_disp.columns]
sales_show = sales_disp[show_sales].rename(columns=SALES_COLS)

st.dataframe(
    sales_show,
    use_container_width=True,
    height=420,
    column_config={
        "Sale Price": st.column_config.NumberColumn(format="$%d"),
        "Total AV":   st.column_config.NumberColumn(format="$%d"),
        "Land AV":    st.column_config.NumberColumn(format="$%d"),
        "Bldg AV":    st.column_config.NumberColumn(format="$%d"),
        "Ratio":      st.column_config.NumberColumn(format="%.4f"),
        "Sqft":       st.column_config.NumberColumn(format="%d"),
        "Flag":       st.column_config.TextColumn(width="small"),
    },
)
