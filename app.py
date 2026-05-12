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
    IAAO_VEI_MIN,
    IAAO_VEI_MAX,
    IAAO_MEDIAN_RATIO_MIN,
    IAAO_MEDIAN_RATIO_MAX,
    EXCESS_FRONT_PERCENTILE,
    EXCESS_FRONT_MAX_CV,
)
from db import (
    get_neighborhoods,
    get_parcels,
    get_sales,
    get_land_data,
    get_infl_codes,
    get_nbhd_summary,
    get_grade_model_data,
    get_market_values,
)
from stats import (
    GRADE_RANK,
    compute_ratio_stats,
    compute_uniformity_stats,
    flag_outliers,
    compute_land_value_uniformity,
    compute_frontage_tiers,
    compute_dwelling_outliers,
    compute_grade_distribution,
    compute_conformity_flags,
    fit_grade_model,
    compute_grade_creep_flags,
)


@st.cache_data(ttl=3600, show_spinner="Fitting grade model — one-time load…")
def _cached_grade_coefs_v2() -> dict:
    """Fit the within-group grade OLS model once per session."""
    return fit_grade_model(get_grade_model_data())


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neighborhood Review Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Neighborhood Review Dashboard")
st.caption("Stark County Auditor — Mass Appraisal Analytics")


# ── Sidebar ───────────────────────────────────────────────────────────────────

tax_year          = TAX_YEAR
iqr_multiplier    = 1.5
land_iqr_multiplier = 1.5

with st.sidebar:
    st.header("Controls")

    with st.spinner("Loading neighborhoods..."):
        try:
            nbhd_list = get_neighborhoods(tax_year)
        except Exception as exc:
            st.error(f"Could not load neighborhoods: {exc}")
            st.stop()

    if not nbhd_list:
        st.warning(f"No neighborhoods found for tax year {tax_year}.")
        st.stop()

    nbhd_search = st.text_input(
        "Search Neighborhood",
        placeholder="Type to filter or enter code…",
    ).strip().upper()

    filtered_nbhds = (
        [n for n in nbhd_list if nbhd_search in n.upper()]
        if nbhd_search else nbhd_list
    )

    if not filtered_nbhds:
        st.warning(f"No neighborhood matches '{nbhd_search}'.")
        selected_nbhd = nbhd_list[0]
    elif nbhd_search in nbhd_list:
        selected_nbhd = nbhd_search
        st.caption(f"Exact match: **{selected_nbhd}**")
    else:
        selected_nbhd = st.selectbox("Neighborhood", filtered_nbhds)

    st.divider()

    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"Tax year {tax_year}  \n"
        f"Sales: {SALE_YEAR_START}–{SALE_YEAR_END}"
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_land, tab_dwelling = st.tabs(
    ["All Neighborhoods", "Land Analysis", "Dwelling Analysis"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — All Neighborhoods Overview
# ════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("All Neighborhoods Overview")
    st.caption(
        f"One row per neighborhood. Ratio study statistics cover arm's length sales "
        f"{SALE_YEAR_START}–{SALE_YEAR_END} compared against tax year {tax_year} values. "
        f"IAAO level of assessment standard: {IAAO_MEDIAN_RATIO_MIN:.2f}–{IAAO_MEDIAN_RATIO_MAX:.2f}. "
        f"COD standard: {IAAO_COD_MIN}–{IAAO_COD_MAX}. PRD standard: {IAAO_PRD_MIN}–{IAAO_PRD_MAX}."
    )

    with st.spinner("Loading neighborhood summary..."):
        try:
            overview_df = get_nbhd_summary(tax_year)
        except Exception as exc:
            st.error(f"Could not load overview: {exc}")
            overview_df = pd.DataFrame()

    if not overview_df.empty:
        # ── Compute IAAO flag column ──────────────────────────────────────────
        def _iaao_flags(row):
            flags = []
            if pd.notna(row["median_ratio"]):
                if not (IAAO_MEDIAN_RATIO_MIN <= row["median_ratio"] <= IAAO_MEDIAN_RATIO_MAX):
                    flags.append("Level")
            if pd.notna(row["cod"]):
                if not (IAAO_COD_MIN <= row["cod"] <= IAAO_COD_MAX):
                    flags.append("COD")
            if pd.notna(row["prd"]):
                if not (IAAO_PRD_MIN <= row["prd"] <= IAAO_PRD_MAX):
                    flags.append("PRD")
            return ", ".join(flags) if flags else ""

        overview_df["iaao_flags"] = overview_df.apply(_iaao_flags, axis=1)

        has_sales   = overview_df["n_sales"] > 0
        has_flags   = overview_df["iaao_flags"] != ""

        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Total Neighborhoods",    f"{len(overview_df):,}")
        ov2.metric("With Sales Data",        f"{has_sales.sum():,}")
        ov3.metric("IAAO Flags (any)",       f"{has_flags.sum():,}",
                   help="Neighborhoods where at least one of Level, COD, or PRD is outside the IAAO acceptable range.")
        ov4.metric("No Sales Yet",           f"{(~has_sales).sum():,}",
                   help="Neighborhoods with zero arm's length sales in the configured date range.")

        st.dataframe(
            overview_df.rename(columns={
                "nbhd":          "NBHD",
                "n_parcels":     "Parcels",
                "median_av":     "Median AV",
                "n_sales":       "Sales (n)",
                "median_ratio":  "Median Ratio",
                "cod":           "COD",
                "prd":           "PRD",
                "iaao_flags":    "IAAO Flags",
            }),
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "Median AV":    st.column_config.NumberColumn(format="$%d"),
                "Median Ratio": st.column_config.NumberColumn(format="%.4f"),
                "COD":          st.column_config.NumberColumn(format="%.2f"),
                "PRD":          st.column_config.NumberColumn(format="%.4f"),
                "IAAO Flags":   st.column_config.TextColumn(
                    width="medium",
                    help="Blank = all metrics within IAAO range. Flags: Level (median ratio), COD, PRD.",
                ),
            },
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Land Analysis
# ════════════════════════════════════════════════════════════════════════════

with tab_land:
    st.header(f"Land Analysis — NBHD {selected_nbhd}")

    # ── Load Data ─────────────────────────────────────────────────────────────

    with st.spinner(f"Loading land data for NBHD {selected_nbhd}..."):
        try:
            land_df   = get_land_data(selected_nbhd, tax_year)
            sales_df  = get_sales(selected_nbhd)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

    if land_df.empty:
        st.info("No land records found for this neighborhood.")
        st.stop()


    # ── Section 1: Land Value Uniformity (all parcels) ───────────────────────

    st.subheader("Land Value Uniformity — All Parcels")
    st.caption(
        "$/Acre = APRLAND ÷ CALCACRES. Universally comparable across LTYPE "
        f"(Acres and Front Foot). Outliers are parcels outside the IQR fences "
        f"(Q1 − {land_iqr_multiplier:.1f}×IQR / Q3 + {land_iqr_multiplier:.1f}×IQR). "
        "Parcels with missing or zero CALCACRES are flagged DATA QUALITY."
    )

    lv_stats, lv_df = compute_land_value_uniformity(land_df, land_iqr_multiplier)

    if "error" in lv_stats:
        st.warning(lv_stats["error"])
    else:
        lc1, lc2, lc3, lc4, lc5, lc6 = st.columns(6)
        lc1.metric("Total Parcels",    f"{lv_stats['n']:,}")
        lc2.metric("With Acreage",     f"{lv_stats['n_with_acres']:,}")
        lc3.metric(
            "Data Quality",
            f"{lv_stats['n_dq']:,}",
            help="Parcels missing CALCACRES — cannot compute $/Acre.",
        )
        lc4.metric("Median $/Acre",    f"${lv_stats['median_dol_per_acre']:,.0f}")
        lc5.metric(
            "COD ($/Acre)",
            f"{lv_stats['cod']:.2f}",
            help=(
                "Coefficient of Dispersion of land value per acre.\n"
                "Lower = more uniform land value across parcels."
            ),
        )
        lc6.metric("COV ($/Acre)",     f"{lv_stats['cov']:.2f}%")

        if lv_stats["n_dq"] > 0:
            st.warning(
                f"{lv_stats['n_dq']} parcel(s) have missing or zero CALCACRES "
                "and are flagged DATA QUALITY. $/Acre cannot be computed for these."
            )

    LV_COLS = {
        "parid":         "Parcel ID",
        "address":       "Address",
        "luc":           "LUC",
        "ltype_summary": "LTYPE",
        "calcacres":     "Calc Acres",
        "aprland":       "Land AV",
        "dol_per_acre":  "$/Acre",
        "flag":          "Flag",
    }
    show_lv = [c for c in LV_COLS if c in lv_df.columns]
    lv_show = lv_df[show_lv].rename(columns=LV_COLS)

    st.dataframe(
        lv_show,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Land AV":    st.column_config.NumberColumn(format="$%d"),
            "$/Acre":     st.column_config.NumberColumn(format="$%,.0f"),
            "Calc Acres": st.column_config.NumberColumn(format="%.4f"),
            "Flag":       st.column_config.TextColumn(width="small"),
        },
    )


    # ── Section 2: Excess Frontage Tiers (Front Foot parcels only) ───────────

    st.divider()
    st.subheader("Excess Frontage Analysis — Front Foot Parcels")

    with st.expander("How this analysis works", expanded=False):
        p0  = int(EXCESS_FRONT_PERCENTILE)
        p1  = round(p0 + (100 - p0) / 4, 2)
        p2  = round(p0 + (100 - p0) / 2, 2)
        p3  = round(p0 + (100 - p0) * 3 / 4, 2)
        top = int(100 - p0)
        st.markdown(f"""
**Purpose — Internal Consistency and Proportionality**
This analysis answers two questions:
1. **Consistency** — Do parcels with similar frontage carry similar '01' adjustments?
2. **Proportionality** — Do parcels with more excess frontage carry larger adjustments?

No vacant land sales are required. Suggestions are derived from what peer parcels
in the same neighborhood already carry. Only the widest **{top}%** of lots by
frontage are reviewed (configurable via `EXCESS_FRONT_PERCENTILE` in config.py).

---

**Step 1 — Normal Frontage Threshold**
The boundary between "Normal" and "Excess" frontage is the **{p0}th percentile**
of total frontage across all front-foot parcels in the neighborhood.
Lots at or below this value are Normal — no excess frontage adjustment is warranted.
Normal-tier parcels that already carry a '01' adjustment are flagged for removal.

---

**Step 2 — Tier Bands (Peer Groups)**
The excess population (top {top}%) is divided into four equal percentile bands.
Each band is a peer group — parcels in the same tier should carry the same adjustment.
Tiers are ordered so that wider lots always receive a larger adjustment (proportionality):

| Frontage Percentile | Tier | Proportional Default |
|---|---|---|
| {p0}th – {p1}th | Tier 1 | 5% |
| {p1}th – {p2}th | Tier 2 | 10% |
| {p2}th – {p3}th | Tier 3 | 20% |
| Above {p3}th | Tier 4 | 30% |

---

**Step 3 — Suggested Adjustment (Peer Median)**
For each tier, the **Total INFLU %** is the median of existing non-zero '01'
adjustments among peer parcels in that tier, rounded to the nearest 5%.

If fewer than 3 peers already carry a non-zero adjustment, the proportional
default from Step 2 is used. The **Basis** column in the Consistency by Tier
summary shows whether each tier used peer data ("peer") or the default ("default").

---

**Step 4 — Adj Gap and Total INFLU %**
> `Base Value    = Land AV ÷ (1 − Current Adj% ÷ 100)`  *(parcel's unadjusted land value)*
>
> `Suggested AV  = Base Value × (1 − Total INFLU% ÷ 100)`
>
> `Adj Gap       = Total INFLU% − Current Adj%`

| Adj Gap | Meaning | Action |
|---|---|---|
| Positive | Under-adjusted | Increase INFLU to match peers |
| Negative | Over-adjusted | Reduce INFLU to match peers |
| Zero | Already consistent | No change needed |

**Total INFLU %** is the exact value to enter in the INFLU field in CAMA for INFL1 = '01'.

---

**Step 5 — Consistency by Tier Summary**
The summary table below the action metrics shows each tier as a peer group:

| Column | Description |
|---|---|
| Suggested % | The peer-consistent target for this tier |
| Basis | "peer" = from peer median; "default" = proportional fallback |
| Median Adj % | What the neighborhood currently applies on average in this tier |
| SD Adj % | Spread of current adjustments — low = consistent, high = scattered |
| Need Change | Parcels where Adj Gap ≠ 0 (inconsistent with peer target) |

A high SD Adj % within a tier is the signal that standardization is needed.
A low SD with Need Change > 0 means adjustments are consistent but at the wrong level.

---

**Column Reference — Parcel Grid**

| Column | Description |
|---|---|
| Total Frontage | Sum of FRONACT across all non-Acre land lines |
| Tier | Frontage percentile band (peer group) |
| Land Lines | Number of front-foot land lines; Current Adj % is summed across all |
| Influence Codes | All INFL1 codes present on the parcel (not just '01') |
| Current Adj % | Sum of all '01' INFLU adjustments across every land line |
| Adj Gap | Change needed to match peer target (negative = reduce) |
| Total INFLU % | Enter this value in CAMA — the peer-consistent adjustment |
| Land AV | Current appraised land value (APRLAND) |
| $/Ft | Current Land AV ÷ Total Frontage |
| Suggested AV | Land value after applying the peer-consistent Total INFLU % |
        """)

    st.caption(
        "Total frontage per parcel = sum of FRONACT across all non-Acre land lines. "
        "Current Adj % = existing INFL1 '01' adjustment. "
        "Adj Gap = additional change needed (negative = remove existing adjustment). "
        "Total INFLU % = exact value to enter in the INFLU field in CAMA."
    )

    thresholds, ft_df = compute_frontage_tiers(land_df, base_percentile=EXCESS_FRONT_PERCENTILE)
    high_cv = True  # default — overwritten inside the else block below

    if not thresholds:
        st.info("No front-foot parcels found in this neighborhood (fewer than 5).")
    else:
        # ── Compute labels and derived columns first ──────────────────────────
        infl_lookup = get_infl_codes()

        def _label_codes(raw: str) -> str:
            if not raw or raw == "—":
                return "—"
            parts = []
            for code in raw.split(","):
                code = code.strip()
                desc = infl_lookup.get(code, "")
                parts.append(f"{code}: {desc}" if desc else code)
            return "  |  ".join(parts)

        ft_df["infl1_labels"] = ft_df["infl1_summary"].apply(_label_codes)

        # Total INFLU % rounded to nearest 5 — enter this value in CAMA INFLU field
        ft_df["total_reduction"] = (
            ft_df["target_influ"]
            .apply(lambda v: float(round(v / 5) * 5))
        )

        cv = thresholds["sd"] / thresholds["median"] if thresholds["median"] > 0 else 0
        high_cv = cv > EXCESS_FRONT_MAX_CV

        net_change = float((ft_df["sugg_aprland"] - ft_df["aprland"]).sum()) if not high_cv else 0.0

        # ── Row 1: Frontage distribution ──────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Parcels",    f"{len(ft_df):,}")
        c2.metric("Median Frontage",  f"{thresholds['median']:.0f} ft")
        c3.metric("Std Dev",          f"{thresholds['sd']:.0f} ft")
        c4.metric("Normal $/Ft Rate", f"${thresholds['normal_rate']:,.0f}",
                  help="Median $/ft of Normal-tier parcels — informational reference only. Suggested AV is derived from peer adjustments, not this rate.")
        c5.metric("Net Land AV Change", f"${net_change:+,.0f}" if not high_cv else "—",
                  help="Total dollar impact if all recommendations are applied.")

        if high_cv:
            st.warning(
                f"**Excess frontage analysis suppressed.** "
                f"The frontage distribution in this neighborhood is too variable "
                f"(CV = {cv:.2f}, threshold = {EXCESS_FRONT_MAX_CV}) for the SD-based "
                f"tier logic to produce reliable recommendations. "
                f"Review land values manually."
            )
        else:
            n_add    = int((ft_df["adj_gap"] > 0).sum())
            n_remove = int(((ft_df["tier"] == "Normal") & (ft_df["current_adj"] > 0)).sum())
            n_over   = int(((ft_df["tier"] != "Normal") & (ft_df["adj_gap"] < 0)).sum())
            med_influ = ft_df.loc[ft_df["adj_gap"] > 0, "total_reduction"]

            # ── Row 2: Action summary ─────────────────────────────────────────
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Needs Adjustment",     f"{n_add:,}",
                      help="Excess-frontage parcels where the '01' INFLU should be increased.")
            s2.metric("Over-Adjusted",        f"{n_over:,}",
                      help="Excess-frontage parcels where the current '01' adjustment is too large — reduce INFLU.")
            s3.metric("Remove Adjustment",    f"{n_remove:,}",
                      help="Normal-frontage parcels with an existing '01' adjustment — set INFLU to 0.")
            s4.metric("Median Total INFLU %", f"{med_influ.median():.0f}%" if not med_influ.empty else "—",
                      help="Typical INFLU value to enter for parcels needing an adjustment increase.")

            if n_add > 0 or n_remove > 0 or n_over > 0:
                parts = []
                if n_add    > 0: parts.append(f"**{n_add}** parcel(s) need a higher INFLU")
                if n_over   > 0: parts.append(f"**{n_over}** parcel(s) are over-adjusted — reduce INFLU")
                if n_remove > 0: parts.append(f"**{n_remove}** parcel(s) need their '01' adjustment removed")
                st.warning("  |  ".join(parts))
            else:
                st.success("All parcels are within the normal frontage range — no adjustments needed.")

            # ── Row 3: Per-tier consistency summary ───────────────────────────
            st.markdown("**Adjustment Consistency by Tier**")
            st.caption(
                "Each tier is a peer group — parcels in the same tier should carry similar adjustments. "
                "SD Adj % measures how spread the current adjustments are within the tier "
                "(low = consistent, high = inconsistent). "
                "Basis: 'peer' = suggested % derived from peer median; 'default' = fewer than 3 peers with adjustments."
            )
            _tier_order = [
                "Normal",
                "Tier 1 (5%)",
                "Tier 2 (10–15%)",
                "Tier 3 (20–25%)",
                "Tier 4 (30%+)",
            ]
            _sugg_map  = thresholds.get("tier_sugg_map", {})
            _src_map   = thresholds.get("tier_sources",  {})
            _tier_rows = []
            for _t in _tier_order:
                _tp = ft_df[ft_df["tier"] == _t]
                if _tp.empty:
                    continue
                _n       = len(_tp)
                _sugg    = _sugg_map.get(_t, 0.0)
                _basis   = _src_map.get(_t, "n/a")
                _med_adj = float(_tp["current_adj"].median())
                _sd_adj  = float(_tp["current_adj"].std(ddof=1)) if _n > 1 else 0.0
                _need    = int((_tp["adj_gap"] != 0).sum())
                _tier_rows.append({
                    "Tier":         _t,
                    "Parcels":      _n,
                    "Suggested %":  _sugg,
                    "Basis":        _basis,
                    "Median Adj %": round(_med_adj, 1),
                    "SD Adj %":     round(_sd_adj, 1),
                    "Need Change":  _need,
                })
            if _tier_rows:
                st.dataframe(
                    pd.DataFrame(_tier_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Suggested %":  st.column_config.NumberColumn(format="%.0f"),
                        "Median Adj %": st.column_config.NumberColumn(format="%.1f"),
                        "SD Adj %":     st.column_config.NumberColumn(format="%.1f"),
                    },
                )

        # Grid always shown — appraisers can still inspect raw data even when
        # recommendations are suppressed.
        HIDE_RECO_COLS = {"adj_gap", "total_reduction", "sugg_aprland"}
        FT_COLS = {
            "parid":           "Parcel ID",
            "address":         "Address",
            "luc":             "LUC",
            "total_fronact":   "Total Frontage",
            "tier":            "Tier",
            "lline_count":     "Land Lines",
            "calcacres":       "Calc Acres",
            "infl1_labels":    "Influence Codes",
            "current_adj":     "Current Adj %",
            "adj_gap":         "Adj Gap",
            "total_reduction": "Total INFLU %",
            "aprland":         "Land AV",
            "dol_per_ft":      "$/Ft",
            "sugg_aprland":    "Suggested AV",
            "ft_flag":         "Flag",
        }
        if high_cv:
            FT_COLS = {k: v for k, v in FT_COLS.items() if k not in HIDE_RECO_COLS}
        show_ft = [c for c in FT_COLS if c in ft_df.columns]
        ft_show = ft_df[show_ft].rename(columns=FT_COLS)

        st.dataframe(
            ft_show,
            use_container_width=True,
            hide_index=True,
            height=420,
            column_config={
                "Total Frontage":  st.column_config.NumberColumn(format="%.1f"),
                "Land Lines":      st.column_config.NumberColumn(format="%d",
                                   help="Number of front-foot land lines contributing to total frontage. "
                                        "Current Adj % is the sum of all '01' INFLU adjustments across these lines."),
                "Calc Acres":      st.column_config.NumberColumn(format="%.4f"),
                "Current Adj %":   st.column_config.NumberColumn(format="%.0f"),
                "Adj Gap":         st.column_config.NumberColumn(format="%.0f"),
                "Total INFLU %":   st.column_config.NumberColumn(format="%.0f"),
                "Land AV":         st.column_config.NumberColumn(format="$%d"),
                "$/Ft":            st.column_config.NumberColumn(format="$%d"),
                "Suggested AV":    st.column_config.NumberColumn(format="$%d"),
                "Flag":            st.column_config.TextColumn(width="small"),
            },
        )


    # ── Section 3: Vacant Land Sales Comparison ──────────────────────────────

    st.divider()
    st.subheader("Vacant Land Sales Comparison")
    st.caption(
        f"Arm's length vacant land sales ({SALE_YEAR_START}–{SALE_YEAR_END}) compared against "
        "Suggested AV from the excess frontage analysis. "
        "Sugg/Sale Ratio = Suggested AV ÷ Sale Price — where sales exist, values near 1.00 "
        "suggest the peer-consistent adjustments also align with market evidence."
    )

    if "luc" not in sales_df.columns:
        st.info("LUC column not available in sales data — re-run query to include it.")
    else:
        vacant_sales = sales_df[sales_df["luc"].astype(str) == "500"].copy()

        if vacant_sales.empty:
            st.info(
                f"No vacant land (LUC 500) arm's length sales found in "
                f"{SALE_YEAR_START}–{SALE_YEAR_END} for NBHD {selected_nbhd}. "
                "Consider expanding the sale date range in config.py to capture more history."
            )
        else:
            # Merge with frontage tier data for total frontage and suggested AV
            if not ft_df.empty:
                vacant_sales = vacant_sales.merge(
                    ft_df[["parid", "total_fronact", "tier", "sugg_aprland"]],
                    on="parid",
                    how="left",
                )
            else:
                vacant_sales["total_fronact"] = np.nan
                vacant_sales["tier"]          = "—"
                vacant_sales["sugg_aprland"]  = np.nan

            # Sale $/ft and suggested $/ft
            has_front = vacant_sales["total_fronact"].notna() & (vacant_sales["total_fronact"] > 0)
            vacant_sales["sale_dol_per_ft"] = np.where(
                has_front,
                (vacant_sales["price"] / vacant_sales["total_fronact"]).round(0),
                np.nan,
            )
            vacant_sales["sugg_dol_per_ft"] = np.where(
                has_front & vacant_sales["sugg_aprland"].notna(),
                (vacant_sales["sugg_aprland"] / vacant_sales["total_fronact"]).round(0),
                np.nan,
            )

            # Sugg/Sale ratio
            vacant_sales["sugg_sale_ratio"] = np.where(
                vacant_sales["price"] > 0,
                (vacant_sales["sugg_aprland"] / vacant_sales["price"]).round(4),
                np.nan,
            )

            # Summary stats
            valid_ratios = vacant_sales["sugg_sale_ratio"].dropna()
            if len(valid_ratios) >= 2:
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Vacant Land Sales (n)", f"{len(vacant_sales):,}")
                sc2.metric("Median Sugg/Sale Ratio", f"{valid_ratios.median():.4f}",
                           help="Near 1.00 = well-calibrated. < 1.00 = model under market.")
                sc3.metric("Mean Sugg/Sale Ratio",   f"{valid_ratios.mean():.4f}")
                sc4.metric("Median Sale $/Ft",
                           f"${vacant_sales['sale_dol_per_ft'].median():,.0f}"
                           if vacant_sales["sale_dol_per_ft"].notna().any() else "—")

            if "saledt" in vacant_sales.columns:
                vacant_sales["saledt"] = pd.to_datetime(vacant_sales["saledt"]).dt.date

            VS_COLS = {
                "parid":           "Parcel ID",
                "address":         "Address",
                "saledt":          "Sale Date",
                "price":           "Sale Price",
                "total_fronact":   "Total Frontage",
                "sale_dol_per_ft": "Sale $/Ft",
                "tier":            "Tier",
                "aprland":         "Current Land AV",
                "sugg_aprland":    "Suggested AV",
                "sugg_dol_per_ft": "Suggested $/Ft",
                "sugg_sale_ratio": "Sugg/Sale Ratio",
            }
            show_vs = [c for c in VS_COLS if c in vacant_sales.columns]
            vs_show = vacant_sales[show_vs].rename(columns=VS_COLS)

            st.dataframe(
                vs_show,
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    "Sale Price":      st.column_config.NumberColumn(format="$%d"),
                    "Current Land AV": st.column_config.NumberColumn(format="$%d"),
                    "Suggested AV":    st.column_config.NumberColumn(format="$%d"),
                    "Total Frontage":  st.column_config.NumberColumn(format="%.1f"),
                    "Sale $/Ft":       st.column_config.NumberColumn(format="$%d"),
                    "Suggested $/Ft":  st.column_config.NumberColumn(format="$%d"),
                    "Sugg/Sale Ratio": st.column_config.NumberColumn(format="%.4f"),
                },
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dwelling Analysis
# ════════════════════════════════════════════════════════════════════════════

with tab_dwelling:

    # ── Load Data ─────────────────────────────────────────────────────────────

    with st.spinner(f"Loading NBHD {selected_nbhd}..."):
        try:
            parcels_df = get_parcels(selected_nbhd, tax_year)
            sales_df   = get_sales(selected_nbhd)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

        try:
            mkt_df = get_market_values(selected_nbhd, tax_year)
        except Exception as exc:
            st.warning(f"Market model data unavailable: {exc}")
            mkt_df = pd.DataFrame()

    if parcels_df.empty:
        st.warning(f"No parcels found for neighborhood **{selected_nbhd}**.")
        st.stop()

    # ── UPPER PANEL — All Parcels ─────────────────────────────────────────────

    st.header(f"Dwelling Uniformity — NBHD {selected_nbhd}")
    st.caption(
        "Bldg/SqFt = Dwelling AV (ADJRCNLD) ÷ Finished Sq Ft. "
        "COD measures intra-neighborhood uniformity of dwelling values across all parcels."
    )

    # ── LUC Filter ────────────────────────────────────────────────────────────

    available_lucs = sorted(parcels_df["luc"].dropna().unique().tolist())
    selected_lucs = st.multiselect(
        "Filter by LUC",
        options=available_lucs,
        default=available_lucs,
        help="Land Use Codes present in this neighborhood. Deselect to exclude parcels.",
    )

    if selected_lucs:
        parcels_filtered = parcels_df[parcels_df["luc"].isin(selected_lucs)].copy()
    else:
        parcels_filtered = parcels_df.copy()

    # ── Uniformity Stats Row ──────────────────────────────────────────────────

    u_stats = compute_uniformity_stats(parcels_filtered)

    if "error" not in u_stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Parcels", f"{u_stats['n']:,}")
        c2.metric(
            "Median Sqft",
            f"{u_stats['median_sfla']:,.0f}" if u_stats.get("median_sfla") else "—",
            help="Median finished living area across all parcels in this neighborhood.",
        )
        c3.metric(
            "Median Yr Built",
            f"{u_stats['median_yrblt']:.0f}" if u_stats.get("median_yrblt") else "—",
            help="Median year built. Used as the neighborhood baseline for NEW flags.",
        )
        c4.metric(
            "Median Bldg/SqFt",
            f"${u_stats['median_av_per_sqft']:,.2f}",
            help="Median dwelling appraised value (ADJRCNLD) per finished sq ft.",
        )
        c5.metric(
            "COD (Bldg/SqFt)",
            f"{u_stats['cod_av_per_sqft']:.2f}",
            help=(
                "Coefficient of Dispersion of dwelling AV per sq ft.\n"
                "Measures intra-neighborhood uniformity independent of sales.\n"
                f"Based on {u_stats['n_with_area']:,} of {u_stats['n']:,} parcels with area data."
            ),
        )
    else:
        st.warning(u_stats["error"])

    # ── Build Parcel DataFrame ────────────────────────────────────────────────

    disp, _, _ = compute_dwelling_outliers(
        parcels_filtered, iqr_multiplier, int(tax_year)
    )
    disp = compute_conformity_flags(disp)

    UPPER_COLS = {
        "parid":        "Parcel ID",
        "address":      "Address",
        "luc":          "LUC",
        "yrblt":        "Yr Built",
        "effyr":        "Eff Yr",
        "sfla":         "Sqft",
        "grade":        "Grade",
        "cdu":          "CDU",
        "rmbed":        "Beds",
        "fixbath":      "Full Bath",
        "fixhalf":      "Half Bath",
        "heat":         "Central Air",
        "pctcomplete":  "Pct Complete",
        "user11":       "BSMT Finish",
        "fundep":       "Func Dep",
        "funrsn":       "Func Reason",
        "ecndep":       "Econ Dep",
        "ecnrsn":       "Econ Reason",
        "adjrcnld":     "Dwelling AV",
        "aprtot":       "Total AV",
        "bldg_sqft":    "Bldg/SqFt",
        "conform_flag": "Flag",
    }
    show_cols = [c for c in UPPER_COLS if c in disp.columns]
    disp_show = disp[show_cols].rename(columns=UPPER_COLS)

    st.dataframe(
        disp_show,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config={
            "Dwelling AV": st.column_config.NumberColumn(format="$%d",
                           help="ADJRCNLD — dwelling-only appraised value, excludes OBY items."),
            "Total AV":    st.column_config.NumberColumn(format="$%d"),
            "Bldg/SqFt":  st.column_config.NumberColumn(format="$%.2f",
                           help="Dwelling AV (ADJRCNLD) ÷ Sqft. Excludes OBY and land."),
            "Sqft":        st.column_config.NumberColumn(format="%d"),
            "Pct Complete":st.column_config.NumberColumn(format="%d"),
            "Func Dep":    st.column_config.NumberColumn(format="%d"),
            "Econ Dep":    st.column_config.NumberColumn(format="%d"),
            "Flag":        st.column_config.TextColumn(
                           width="small",
                           help="LARGE = Sqft well above neighborhood norm. "
                                "NEW = Yr Built well above neighborhood norm. "
                                "LARGE+NEW = both."),
        },
    )


    # ── Grade Distribution ────────────────────────────────────────────────────

    st.divider()
    st.subheader(f"Grade Distribution — NBHD {selected_nbhd}")
    st.caption(
        "Grade reflects construction quality, not market value or condition. "
        "Parcels graded above the neighborhood mode warrant field review to confirm "
        "the grade is supported by physical construction characteristics."
    )

    grade_dist_df, mode_grade = compute_grade_distribution(parcels_filtered["grade"])

    if grade_dist_df.empty:
        st.info("No grade data available.")
    else:
        mode_rank = GRADE_RANK.get(str(mode_grade).strip().upper(), 0) if mode_grade else 0

        above_mode = disp[
            disp["grade"].apply(
                lambda g: GRADE_RANK.get(str(g).strip().upper(), 0) if pd.notna(g) else 0
            ) > mode_rank
        ]

        gd1, gd2, gd3 = st.columns(3)
        gd1.metric(
            "Mode Grade", mode_grade or "—",
            help="Most common grade in this neighborhood — the baseline for comparison.",
        )
        gd2.metric("Parcels Above Mode", f"{len(above_mode):,}")
        gd3.metric(
            "% Above Mode",
            f"{len(above_mode) / len(parcels_filtered) * 100:.1f}%"
            if len(parcels_filtered) > 0 else "—",
        )

        st.dataframe(
            grade_dist_df.rename(columns={
                "grade": "Grade",
                "count": "Count",
                "pct":   "% of NBHD",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "% of NBHD": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        if not above_mode.empty:
            st.markdown(f"**Parcels graded above neighborhood mode ({mode_grade}):**")
            ABOVE_COLS = {
                "parid":     "Parcel ID",
                "address":   "Address",
                "grade":     "Grade",
                "cdu":       "CDU",
                "yrblt":     "Yr Built",
                "effyr":     "Eff Yr",
                "sfla":      "Sqft",
                "adjrcnld":  "Dwelling AV",
                "bldg_sqft": "Bldg/SqFt",
            }
            show_ac = [c for c in ABOVE_COLS if c in above_mode.columns]
            st.dataframe(
                above_mode[show_ac].rename(columns=ABOVE_COLS),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Dwelling AV": st.column_config.NumberColumn(format="$%d"),
                    "Bldg/SqFt":  st.column_config.NumberColumn(format="$%.2f"),
                    "Sqft":       st.column_config.NumberColumn(format="%d"),
                },
            )

    # ── Grade Creep Review ────────────────────────────────────────────────────

    st.divider()
    st.subheader(f"Grade Creep — Review Candidates — NBHD {selected_nbhd}")
    st.caption(
        "Parcels where the actual grade letter is higher than the grade the model "
        "predicts from physical characteristics (SFLA, Yr Built, Beds, Baths, "
        "Central Air, addition count, garage size). Uses a county-wide OLS model "
        "with neighborhood fixed effects — each parcel is compared against "
        "neighbors with similar physical features, not against the county average."
    )

    try:
        model_data  = get_grade_model_data()
        coef_dict   = _cached_grade_coefs_v2()
        nbhd_model  = model_data[model_data["nbhd"] == selected_nbhd].copy()
        flags_df    = compute_grade_creep_flags(nbhd_model, coef_dict)

        if flags_df is None:
            st.info("Fewer than 30 parcels — grade model suppressed for this neighborhood.")
        elif flags_df.empty:
            st.success("No parcels flagged for grade creep in this neighborhood.")
        else:
            st.metric(
                "Flagged for Review",
                f"{len(flags_df):,}",
                help=(
                    "Parcels where the actual grade letter exceeds the model-predicted "
                    "grade letter by at least one full step (e.g., predicted C, actual B)."
                ),
            )
            FLAG_COLS = {
                "parid":       "Parcel ID",
                "address":     "Address",
                "yrblt":       "Yr Built",
                "grade":       "Grade",
                "pred_letter": "Pred. Grade",
                "residual":    "Over By",
            }
            show_fc = [c for c in FLAG_COLS if c in flags_df.columns]
            st.dataframe(
                flags_df[show_fc].rename(columns=FLAG_COLS),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Over By": st.column_config.NumberColumn(
                        format="%.1f",
                        help=(
                            "Grade rank points above the model prediction. "
                            "Higher = stronger signal of over-grading. "
                            "3 points ≈ one full grade step (e.g., C to B)."
                        ),
                    ),
                },
            )
    except Exception as exc:
        st.warning(f"Grade creep model unavailable: {exc}")

    # ── LOWER PANEL — Arm's Length Sales / Ratio Study ────────────────────────

    st.divider()
    st.header(f"Arm's Length Sales ({SALE_YEAR_START}–{SALE_YEAR_END}) — NBHD {selected_nbhd}")

    if sales_df.empty:
        st.info(
            f"No arm's length sales found for NBHD {selected_nbhd} "
            f"in {SALE_YEAR_START}–{SALE_YEAR_END}."
        )
        st.stop()

    s_stats = compute_ratio_stats(sales_df)

    # ── Ratio Study Stats Row ─────────────────────────────────────────────────

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

        r6.metric(
            "VEI",
            f"{s_stats['vei']:.4f}",
            help=(
                f"Vertical Equity Index  (p = {s_stats['vei_pvalue']:.4f})\n"
                f"IAAO standard: {IAAO_VEI_MIN} to {IAAO_VEI_MAX}\n"
                "Negative → regressive (high-value properties under-assessed)\n"
                "Positive → progressive (high-value properties over-assessed)\n"
                "OLS slope of ln(ratio) ~ ln(sale price)."
            ),
        )

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

        # ── IAAO Flag Indicators ──────────────────────────────────────────────

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

        vei = s_stats["vei"]
        if not (IAAO_VEI_MIN <= vei <= IAAO_VEI_MAX):
            direction = "regressive" if vei < IAAO_VEI_MIN else "progressive"
            flags.append(
                f"VEI {vei:.4f} (p = {s_stats['vei_pvalue']:.4f}) indicates "
                f"{direction} vertical inequity "
                f"(IAAO standard: {IAAO_VEI_MIN} to {IAAO_VEI_MAX})."
            )

        if flags:
            st.warning("**IAAO Performance Flags:**\n\n" + "\n\n".join(f"• {f}" for f in flags))
        else:
            st.success(
                f"All ratio study statistics are within IAAO acceptable ranges "
                f"(n = {s_stats['n']} sales)."
            )

    # ── Sales Detail Grid ─────────────────────────────────────────────────────

    st.subheader("Sales Detail")

    sales_disp = sales_df.copy()
    sales_disp["ratio"] = sales_disp["aprtot"] / sales_disp["price"]

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
    }
    show_sales = [c for c in SALES_COLS if c in sales_disp.columns]
    sales_show = sales_disp[show_sales].rename(columns=SALES_COLS)

    st.dataframe(
        sales_show,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Sale Price": st.column_config.NumberColumn(format="$%d"),
            "Total AV":   st.column_config.NumberColumn(format="$%d"),
            "Land AV":    st.column_config.NumberColumn(format="$%d"),
            "Bldg AV":    st.column_config.NumberColumn(format="$%d"),
            "Ratio":      st.column_config.NumberColumn(format="%.4f"),
            "Sqft":       st.column_config.NumberColumn(format="%d"),
        },
    )
