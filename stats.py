"""
stats.py — IAAO ratio study and uniformity statistics

References:
  IAAO Standard on Ratio Studies (2013; 2025 Exposure Draft)
  IAAO Standard on Mass Appraisal of Real Property

─────────────────────────────────────────────────────────────────────────────
RATIO STUDY STATISTICS  (lower panel — arm's length sales)
─────────────────────────────────────────────────────────────────────────────

  ratio_i         = aprtot_i / price_i  (assessment-sales ratio per parcel)

  Median Ratio    = median(ratio_i)
  Mean Ratio      = mean(ratio_i)
  Weighted Mean   = Σ(aprtot_i) / Σ(price_i)

  COD             = [ mean(|ratio_i − median|) / median ] × 100
                    Coefficient of Dispersion; measures uniformity.
                    IAAO residential standard: 5 – 15.

  VEI             = slope b in OLS regression:
                      ln(ratio_i) = α + b × ln(price_i) + ε_i
                    Vertical Equity Index. Measures price-related bias.
                    IAAO standard: −0.05 to +0.05.
                    Negative VEI → regressive (high-value properties under-assessed)
                    Positive VEI → progressive (high-value properties over-assessed)

  PRD             = mean_ratio / weighted_mean_ratio
                    Price-Related Differential; measures vertical equity.
                    IAAO standard: 0.98 – 1.03.
                    PRD > 1.03 → regressive (high-value under-assessed)
                    PRD < 0.98 → progressive (high-value over-assessed)

  PRB             = slope b in OLS regression:
                      y_i = (ratio_i − median) / median
                      x_i = log₂[ (aprtot_i + price_i) / 2 ]
                    Each unit of x = one doubling of estimated market value.
                    IAAO standard: −0.05 to +0.05.
                    Negative PRB → regressive bias.

  Confidence Intervals for Median Ratio and COD:
                    Bootstrap method, 1 000 iterations, 95% CI.
                    Bootstrap is recommended by IAAO for small samples.

─────────────────────────────────────────────────────────────────────────────
UNIFORMITY STATISTICS  (upper panel — all parcels)
─────────────────────────────────────────────────────────────────────────────

  av_per_sqft_i   = aprtot_i / sfla_i  (assessed value per sq ft)

  COD (AV/SqFt)   = [ mean(|av_per_sqft_i − median|) / median ] × 100
  COV (AV/SqFt)   = [ std(av_per_sqft_i, ddof=1) / mean ] × 100

  These measure intra-neighborhood uniformity of the assessment roll
  independent of sales, covering all parcels.

─────────────────────────────────────────────────────────────────────────────
LAND VALUE UNIFORMITY
─────────────────────────────────────────────────────────────────────────────

  dol_per_acre_i  = aprland_i / calcacres_i

  COD ($/Acre)    = [ mean(|dol_per_acre_i − median|) / median ] × 100
  COV ($/Acre)    = [ std(dol_per_acre_i, ddof=1) / mean ] × 100

  $/Acre is a universal metric that equitably compares parcels valued by
  different LTYPE (Acres vs Front Foot). DATA QUALITY flag when CALCACRES
  is null or zero.

─────────────────────────────────────────────────────────────────────────────
OUTLIER DETECTION
─────────────────────────────────────────────────────────────────────────────

  A value is flagged when:  |value − mean| > threshold × SD
  Threshold is set interactively in the dashboard sidebar.

─────────────────────────────────────────────────────────────────────────────
EXCESS FRONTAGE TIERS  (Internal Consistency & Proportionality Framework)
─────────────────────────────────────────────────────────────────────────────

  Total frontage per parcel = Σ FRONACT across all non-Acre (ltype ≠ 'A')
  land lines. Thresholds derive from the percentile distribution of parcel totals.

  Normal  : total ≤ Pth percentile              → no adjustment warranted
  Tier 1  : Pth – (P + band)th percentile       → peer group, default 5%
  Tier 2  : next equal band                      → peer group, default 10%
  Tier 3  : next equal band                      → peer group, default 20%
  Tier 4  : above (P + 3*band)th percentile     → peer group, default 30%

  (P = EXCESS_FRONT_PERCENTILE; band = (100−P)/4)

  Suggested adjustment per tier = median of existing non-zero '01' INFLU
  adjustments among peer parcels in that tier, rounded to the nearest 5%.
  Falls back to the proportional default if fewer than 3 peers have adjustments.

  Adj Gap = Suggested % − Current Adj % (positive = under-adj, negative = over-adj).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ── Ratio Study ───────────────────────────────────────────────────────────────

def compute_ratio_stats(
    sales_df: pd.DataFrame,
    n_bootstrap: int = 1_000,
    ci_level: float = 0.95,
) -> dict:
    """
    Compute IAAO ratio study statistics for a set of arm's length sales.

    Parameters
    ----------
    sales_df    DataFrame with at minimum columns: aprtot (float), price (float).
    n_bootstrap Number of bootstrap resamples for confidence intervals.
    ci_level    Confidence level; default 0.95 produces 95% CIs.

    Returns
    -------
    dict with keys: n, median_ratio, mean_ratio, weighted_mean, cod,
                    vei, vei_pvalue, prd, prb, prb_pvalue,
                    median_ratio_ci, cod_ci.
    Returns {'error': str} if data are insufficient.
    """
    df = sales_df.dropna(subset=["aprtot", "price"])
    df = df[(df["price"] > 0) & (df["aprtot"] > 0)].copy()

    if len(df) < 5:
        return {
            "error": (
                f"Insufficient sales for statistics (n={len(df)}). "
                "At least 5 arm's length sales are required."
            )
        }

    ratios = (df["aprtot"] / df["price"]).reset_index(drop=True)
    n = len(ratios)

    median_ratio  = float(ratios.median())
    mean_ratio    = float(ratios.mean())
    weighted_mean = float(df["aprtot"].sum() / df["price"].sum())

    # COD
    cod = float((np.abs(ratios - median_ratio).mean() / median_ratio) * 100)

    # VEI — OLS slope of ln(ratio) on ln(sale price)
    ln_ratio = np.log(ratios.values)
    ln_price = np.log(df["price"].values)
    vei_slope, _vei_int, _vei_r, vei_pvalue, _vei_se = sp_stats.linregress(ln_price, ln_ratio)
    vei       = float(vei_slope)
    vei_pvalue = float(vei_pvalue)

    # PRD
    prd = float(mean_ratio / weighted_mean)

    # PRB — OLS regression of normalised ratio deviation on log₂ of estimated value
    est_value = (df["aprtot"].values + df["price"].values) / 2.0
    y = (ratios.values - median_ratio) / median_ratio
    x = np.log2(est_value)
    slope, _intercept, _r, p_value, _se = sp_stats.linregress(x, y)
    prb        = float(slope)
    prb_pvalue = float(p_value)

    # Bootstrap confidence intervals
    rng          = np.random.default_rng(seed=42)
    ratios_arr   = ratios.values
    boot_medians = np.empty(n_bootstrap)
    boot_cods    = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        samp             = rng.choice(ratios_arr, size=n, replace=True)
        boot_med         = float(np.median(samp))
        boot_medians[i]  = boot_med
        boot_cods[i]     = (np.abs(samp - boot_med).mean() / boot_med) * 100

    alpha = 1.0 - ci_level
    lo_pct, hi_pct = alpha / 2 * 100, (1 - alpha / 2) * 100

    return {
        "n":               n,
        "median_ratio":    median_ratio,
        "mean_ratio":      mean_ratio,
        "weighted_mean":   weighted_mean,
        "cod":             cod,
        "vei":             vei,
        "vei_pvalue":      vei_pvalue,
        "prd":             prd,
        "prb":             prb,
        "prb_pvalue":      prb_pvalue,
        "median_ratio_ci": (
            float(np.percentile(boot_medians, lo_pct)),
            float(np.percentile(boot_medians, hi_pct)),
        ),
        "cod_ci": (
            float(np.percentile(boot_cods, lo_pct)),
            float(np.percentile(boot_cods, hi_pct)),
        ),
    }


# ── Uniformity Stats (all parcels) ────────────────────────────────────────────

def compute_uniformity_stats(parcels_df: pd.DataFrame) -> dict:
    """
    Compute uniformity statistics for all parcels using assessed value per sq ft.

    Parcels with zero or missing area (sfla) are excluded from COD/COV
    but counted in the total parcel count.

    Parameters
    ----------
    parcels_df  DataFrame with columns: adjrcnld (float), sfla (float).

    Returns
    -------
    dict with keys: n, n_with_area, median_av, mean_av,
                    median_av_per_sqft, mean_av_per_sqft,
                    cod_av_per_sqft, cov_av_per_sqft.
    Returns {'error': str} if data are insufficient.
    """
    total_n = len(parcels_df)

    df = parcels_df.dropna(subset=["adjrcnld", "sfla"])
    df = df[(df["sfla"] > 0) & (df["adjrcnld"] > 0)].copy()

    if len(df) < 5:
        return {
            "error": (
                f"Insufficient parcels with area data (n={len(df)} of {total_n}). "
                "Bldg/SqFt uniformity statistics cannot be computed."
            )
        }

    df["av_per_sqft"] = df["adjrcnld"] / df["sfla"]
    vals = df["av_per_sqft"]

    median_val = float(vals.median())
    mean_val   = float(vals.mean())
    cod = float((np.abs(vals - median_val).mean() / median_val) * 100)
    cov = float((vals.std(ddof=1) / mean_val) * 100)

    sfla_vals  = parcels_df["sfla"].dropna()
    yrblt_vals = parcels_df["yrblt"].dropna() if "yrblt" in parcels_df.columns else pd.Series(dtype=float)

    return {
        "n":                  total_n,
        "n_with_area":        len(df),
        "median_av":          float(parcels_df["aprtot"].median()),
        "mean_av":            float(parcels_df["aprtot"].mean()),
        "median_sfla":        float(sfla_vals.median())  if len(sfla_vals)  > 0 else None,
        "median_yrblt":       float(yrblt_vals.median()) if len(yrblt_vals) > 0 else None,
        "median_av_per_sqft": median_val,
        "mean_av_per_sqft":   mean_val,
        "cod_av_per_sqft":    cod,
        "cov_av_per_sqft":    cov,
    }


# ── Outlier Detection ─────────────────────────────────────────────────────────

def flag_outliers(values: pd.Series, iqr_multiplier: float) -> pd.Series:
    """
    Return a boolean Series: True where value is outside Tukey IQR fences.
    Low fence:  Q1 - iqr_multiplier × IQR
    High fence: Q3 + iqr_multiplier × IQR
    Standard multiplier = 1.5; use 3.0 for extreme outliers only.
    NaN values are treated as non-outliers (False).
    """
    q1  = values.quantile(0.25)
    q3  = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=values.index)
    return (values < q1 - iqr_multiplier * iqr) | (values > q3 + iqr_multiplier * iqr)


# ── Dwelling Outlier Detection ────────────────────────────────────────────────

GRADE_RANK = {
    'X+': 16, 'X':  15, 'X-': 14,
    'A+': 13, 'A':  12, 'A-': 11,
    'B+': 10, 'B':   9, 'B-':  8,
    'C+':  7, 'C':   6, 'C-':  5,
    'D+':  4, 'D':   3, 'D-':  2,
    'E':   1,
}
_CDU_RANK = {
    'EX': 8, 'VG': 7, 'GD': 6, 'AV': 5,
    'FR': 4, 'PR': 3, 'VP': 2, 'UN': 1,
}


def _ordinal_encode(series: pd.Series, rank_map: dict) -> np.ndarray:
    """Map known codes to ordinal integers; assign median rank to unknowns."""
    median_rank = float(np.median(list(rank_map.values())))
    return np.array([
        rank_map.get(str(v).strip().upper(), median_rank)
        for v in series
    ], dtype=float)


def compute_dwelling_outliers(
    parcels_df: pd.DataFrame,
    iqr_multiplier: float,
    current_year: int,
    min_n: int = 20,
    min_r2: float = 0.25,
) -> tuple:
    """
    Flag dwelling Bldg/SqFt outliers using regression, falling back to
    Grade+CDU peer grouping if regression fails or R² is insufficient.

    Regression model: Bldg/SqFt ~ Grade + CDU + log(Sqft) + Age
    where Age = current_year - effyr. Grade and CDU are ordinally encoded.

    Flags are applied to the residuals (actual - predicted) using IQR fences,
    so a HIGH/LOW flag means the parcel is unexpectedly high or low *given*
    its grade, condition, size, and age.

    Parameters
    ----------
    parcels_df    DataFrame with columns: adjrcnld, sfla, grade, cdu, effyr.
    iqr_multiplier  IQR fence multiplier (1.5 = standard Tukey).
    current_year  Used to compute age from effyr.
    min_n         Minimum complete observations for regression (default 20).
    min_r2        Minimum R² to accept regression results (default 0.25).

    Returns
    -------
    (result_df, method, r_squared)
      result_df   Copy of parcels_df with added columns:
                    bldg_sqft, predicted_bldg_sqft, residual_pct, dwell_flag
      method      'regression' | 'peer_group' | 'insufficient_data'
      r_squared   Float if regression was attempted, else None
    """
    df = parcels_df.copy()

    df["bldg_sqft"] = np.where(
        (df["sfla"].fillna(0) > 0) & (df["adjrcnld"].fillna(0) > 0),
        df["adjrcnld"] / df["sfla"],
        np.nan,
    )
    df["predicted_bldg_sqft"] = np.nan
    df["residual_pct"]        = np.nan
    df["dwell_flag"]          = ""

    # ── Regression attempt ────────────────────────────────────────────────────
    reg_mask = (
        df["bldg_sqft"].notna()
        & df["grade"].notna()
        & df["cdu"].notna()
        & df["effyr"].notna()
        & (df["sfla"] > 0)
    )
    reg_df    = df[reg_mask].copy()
    r_squared = None
    method    = "insufficient_data"

    if len(reg_df) >= min_n:
        grade_enc = _ordinal_encode(reg_df["grade"], GRADE_RANK)
        cdu_enc   = _ordinal_encode(reg_df["cdu"],   _CDU_RANK)
        log_sfla  = np.log(reg_df["sfla"].values.astype(float))
        age       = np.clip(current_year - reg_df["effyr"].values.astype(float), 0, None)
        y         = reg_df["bldg_sqft"].values.astype(float)

        X = np.column_stack([np.ones(len(reg_df)), grade_enc, cdu_enc, log_sfla, age])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            predicted = X @ coeffs
            ss_res    = float(np.sum((y - predicted) ** 2))
            ss_tot    = float(np.sum((y - y.mean()) ** 2))
            r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            if r_squared >= min_r2:
                method = "regression"
                df.loc[reg_mask, "predicted_bldg_sqft"] = predicted
                df.loc[reg_mask, "residual_pct"] = np.where(
                    predicted > 0,
                    (y - predicted) / predicted * 100,
                    np.nan,
                )
        except np.linalg.LinAlgError:
            pass

    # ── Peer grouping fallback ────────────────────────────────────────────────
    if method != "regression":
        peer_mask = df["bldg_sqft"].notna() & df["grade"].notna() & df["cdu"].notna()
        if peer_mask.sum() >= 5:
            method = "peer_group"
            peer_med = (
                df[peer_mask]
                .groupby(["grade", "cdu"])["bldg_sqft"]
                .median()
                .rename("_peer_med")
            )
            df = df.join(peer_med, on=["grade", "cdu"])
            df.loc[peer_mask, "predicted_bldg_sqft"] = df.loc[peer_mask, "_peer_med"]
            df.loc[peer_mask, "residual_pct"] = np.where(
                df.loc[peer_mask, "_peer_med"] > 0,
                (df.loc[peer_mask, "bldg_sqft"] - df.loc[peer_mask, "_peer_med"])
                / df.loc[peer_mask, "_peer_med"] * 100,
                np.nan,
            )
            df = df.drop(columns=["_peer_med"])

    # ── IQR flagging on residuals ─────────────────────────────────────────────
    valid_resid = df["residual_pct"].dropna()
    if len(valid_resid) >= 5:
        q1  = valid_resid.quantile(0.25)
        q3  = valid_resid.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            df.loc[df["residual_pct"] > q3 + iqr_multiplier * iqr, "dwell_flag"] = "HIGH"
            df.loc[df["residual_pct"] < q1 - iqr_multiplier * iqr, "dwell_flag"] = "LOW"

    return df, method, r_squared


# ── Land Value Uniformity ─────────────────────────────────────────────────────

def compute_land_value_uniformity(
    land_df: pd.DataFrame,
    sd_threshold: float = 2.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute land value uniformity using APRLAND / CALCACRES ($/acre).

    One row per parcel is produced. APRLAND and CALCACRES are parcel-level
    fields (same value on all land lines for the same parcel).

    A LTYPE summary column is added showing all land types present for the
    parcel (e.g. 'A', 'F', 'A,F').

    DATA QUALITY flag : CALCACRES is NULL or <= 0.
    OUTLIER flag      : $/acre deviates more than sd_threshold SD from the mean.

    Parameters
    ----------
    land_df      : DataFrame from get_land_data()
    sd_threshold : SD multiplier for outlier flagging

    Returns
    -------
    stats  : dict with keys n, n_with_acres, n_dq, median_dol_per_acre,
             mean_dol_per_acre, cod, cov — or {'error': str} if insufficient.
    result : one-row-per-parcel DataFrame with added columns ltype_summary,
             dol_per_acre, flag.
    """
    # One row per parcel (parcel-level fields repeat across llines)
    parcels = land_df.drop_duplicates(subset=["parid"]).copy()
    total_n = len(parcels)

    # LTYPE summary: sorted, comma-separated unique types per parcel
    ltype_summary = (
        land_df.groupby("parid")["ltype"]
        .apply(lambda x: ",".join(sorted(x.dropna().unique())))
        .reset_index()
    )
    ltype_summary.columns = ["parid", "ltype_summary"]
    parcels = parcels.merge(ltype_summary, on="parid", how="left")

    # DATA QUALITY: missing or zero CALCACRES
    dq_mask = parcels["calcacres"].isna() | (parcels["calcacres"] <= 0)
    n_dq = int(dq_mask.sum())

    # $/acre for valid parcels
    valid_mask = ~dq_mask & parcels["aprland"].notna() & (parcels["aprland"] > 0)
    parcels["dol_per_acre"] = np.where(
        valid_mask,
        parcels["aprland"] / parcels["calcacres"],
        np.nan,
    )

    valid = parcels[valid_mask].copy()

    if len(valid) < 5:
        parcels["flag"] = np.where(dq_mask, "DATA QUALITY", "")
        return {
            "error": (
                f"Insufficient parcels with acreage data (n={len(valid)} of {total_n}). "
                "Land value uniformity statistics cannot be computed."
            )
        }, parcels

    vals       = valid["dol_per_acre"]
    median_val = float(vals.median())
    mean_val   = float(vals.mean())
    cod        = float((np.abs(vals - median_val).mean() / median_val) * 100)
    cov        = float((vals.std(ddof=1) / mean_val) * 100)

    stats = {
        "n":                   total_n,
        "n_with_acres":        len(valid),
        "n_dq":                n_dq,
        "median_dol_per_acre": median_val,
        "mean_dol_per_acre":   mean_val,
        "cod":                 cod,
        "cov":                 cov,
    }

    # Outlier flagging on $/acre
    outlier_bool    = flag_outliers(valid["dol_per_acre"], sd_threshold)
    outlier_parids  = set(valid.loc[outlier_bool, "parid"])

    flags = pd.Series("", index=parcels.index)
    flags[dq_mask] = "DATA QUALITY"
    flags[parcels["parid"].isin(outlier_parids)] = "OUTLIER"
    parcels["flag"] = flags

    return stats, parcels


# ── Grade Independence Analysis ──────────────────────────────────────────────

def compute_grade_ratio_stats(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ratio study statistics broken out by grade.

    Used to detect over-grading: if higher-grade properties carry higher
    assessment ratios than lower-grade properties, the grade premium is
    not supported by market evidence — grade was likely set to match price
    rather than reflect physical characteristics.

    Parameters
    ----------
    sales_df   DataFrame with columns: grade, price, aprtot.

    Returns
    -------
    DataFrame with one row per grade, sorted lowest to highest grade.
    Columns: grade, n_sales, median_sale, median_aprtot, median_ratio,
             cod, vs_nbhd_pct (difference from overall median ratio,
             in percentage points).
    """
    df = sales_df.copy()
    df = df[df["price"].fillna(0) > 0]
    df = df[df["aprtot"].fillna(0) > 0]
    df = df[df["grade"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    df["ratio"] = df["aprtot"] / df["price"]
    overall_median = float(df["ratio"].median())

    rows = []
    for grade, grp in df.groupby("grade"):
        ratios = grp["ratio"]
        n      = len(ratios)
        med_r  = float(ratios.median())
        cod    = (
            round(float((ratios - med_r).abs().mean() / med_r * 100), 2)
            if n >= 3 and med_r > 0 else None
        )
        rows.append({
            "grade":        grade,
            "n_sales":      n,
            "median_sale":  float(grp["price"].median()),
            "median_aprtot": float(grp["aprtot"].median()),
            "median_ratio": round(med_r, 4),
            "cod":          cod,
            "vs_nbhd_pct":  round((med_r - overall_median) * 100, 2),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["_rank"] = result["grade"].map(
        lambda g: GRADE_RANK.get(str(g).strip().upper(), 6.5)
    )
    return (
        result
        .sort_values("_rank")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


# ── Conformity Flags ─────────────────────────────────────────────────────────

def compute_conformity_flags(parcels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag parcels that are over-improvements relative to their neighborhood.

    Uses IQR fences (Q3 + 1.5×IQR) on SFLA and YRBLT independently.
    A parcel is flagged if it exceeds the upper fence on either dimension.

    Flag values:
      LARGE      — SFLA significantly above neighborhood norm
      NEW        — YRBLT significantly newer than neighborhood norm
      LARGE+NEW  — both dimensions flagged

    Adds column: conform_flag
    """
    df = parcels_df.copy()
    df["conform_flag"] = ""

    def _upper_fence(series: pd.Series) -> float | None:
        clean = series.dropna()
        if len(clean) < 5:
            return None
        q1  = float(clean.quantile(0.25))
        q3  = float(clean.quantile(0.75))
        iqr = q3 - q1
        return (q3 + 1.5 * iqr) if iqr > 0 else None

    sfla_fence  = _upper_fence(df["sfla"])
    yrblt_fence = _upper_fence(df["yrblt"]) if "yrblt" in df.columns else None

    large = pd.Series(False, index=df.index)
    new   = pd.Series(False, index=df.index)

    if sfla_fence is not None:
        large = df["sfla"].fillna(0) > sfla_fence
    if yrblt_fence is not None:
        new   = df["yrblt"].fillna(0) > yrblt_fence

    df.loc[ large & ~new, "conform_flag"] = "LARGE"
    df.loc[~large &  new, "conform_flag"] = "NEW"
    df.loc[ large &  new, "conform_flag"] = "LARGE+NEW"

    return df


# ── Grade Creep Model ─────────────────────────────────────────────────────────

_LETTER_ORDER   = {"E": 1, "D": 2, "C": 3, "B": 4, "A": 5, "X": 6}
_MODEL_PHYSICAL = [
    "sfla", "yrblt", "rmbed", "fixbath", "fixhalf",
    "heat_bin", "addition_count", "garage_sqft",
]


def _grade_letter(rank: float) -> str:
    """Map a continuous grade rank to its letter tier (ignoring +/-)."""
    if rank >= 14: return "X"
    if rank >= 11: return "A"
    if rank >= 8:  return "B"
    if rank >= 5:  return "C"
    if rank >= 2:  return "D"
    return "E"


def _encode_model_df(df: pd.DataFrame) -> pd.DataFrame:
    """Encode raw parcel DataFrame into model-ready columns. Returns a copy."""
    df = df.copy()

    def _heat(v):
        if pd.isna(v): return 0
        return 0 if str(v).strip().upper() in ("", "N", "0", "NO") else 1

    df["heat_bin"] = df["heat"].apply(_heat)
    for col in ["sfla", "yrblt", "rmbed", "fixbath", "fixhalf",
                "addition_count", "garage_sqft"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["grade_clean"] = df["grade"].astype(str).str.strip().str.upper()
    df["grade_rank"]  = df["grade_clean"].map(GRADE_RANK)
    return df


def fit_grade_model(df: pd.DataFrame) -> dict:
    """
    Fit an OLS grade model with neighborhood fixed effects using the
    within-group (demeaning) transformation.

    Mathematically equivalent to including one dummy per neighborhood but
    runs on a 134K × 9 matrix instead of 134K × 495 — orders of magnitude
    faster, no statsmodels dependency.

    Returns a dict with:
      - one float per physical predictor (the β coefficients)
      - "nbhd_means": nested dict {col: {nbhd: mean}} for prediction
    """
    df    = _encode_model_df(df)
    valid = df[_MODEL_PHYSICAL + ["grade_rank", "nbhd"]].notna().all(axis=1)
    df    = df[valid].copy()

    group_cols  = ["grade_rank"] + _MODEL_PHYSICAL
    nbhd_means  = df.groupby("nbhd")[group_cols].mean()

    # Demean each variable by its neighborhood mean
    df_dm = df[["nbhd", "grade_rank"] + _MODEL_PHYSICAL].copy()
    for col in group_cols:
        df_dm[col] = df[col] - df["nbhd"].map(nbhd_means[col])

    X_fit = np.column_stack([np.ones(len(df_dm)),
                             df_dm[_MODEL_PHYSICAL].values])
    y_fit = df_dm["grade_rank"].values

    coefs, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

    result = dict(zip(_MODEL_PHYSICAL, coefs[1:]))   # skip intercept (≈0 on demeaned)
    result["nbhd_means"] = nbhd_means.to_dict()
    return result


def compute_grade_creep_flags(
    nbhd_df: pd.DataFrame,
    coef_dict: dict,
    min_n: int = 30,
) -> pd.DataFrame | None:
    """
    Apply the grade model to one neighborhood's parcels.

    Returns a DataFrame of parcels where the actual grade LETTER is higher
    than the predicted grade letter (e.g., actual B, predicted C), sorted by
    residual descending.  Returns None if the neighborhood has fewer than
    min_n parcels (model unreliable at small sample sizes).

    Adds columns: pred_rank, pred_letter, residual.
    """
    df = _encode_model_df(nbhd_df)

    for col in ["addition_count", "garage_sqft"]:
        if col not in df.columns:
            df[col] = 0.0

    valid = df[_MODEL_PHYSICAL + ["grade_rank"]].notna().all(axis=1)
    df    = df[valid].copy()

    if len(df) < min_n:
        return None

    # Prediction: nbhd_mean_grade + β·(x - nbhd_mean_x)
    # Equivalent to dummy-variable fixed effects but uses stored group means.
    nbhd        = df["nbhd"].iloc[0]
    nbhd_means  = coef_dict["nbhd_means"]
    mean_grade  = nbhd_means["grade_rank"].get(nbhd, df["grade_rank"].mean())

    pred = pd.Series(mean_grade, index=df.index, dtype=float)
    for col in _MODEL_PHYSICAL:
        mean_col = nbhd_means[col].get(nbhd, 0.0)
        coef     = float(coef_dict.get(col, 0.0))
        pred     = pred + coef * (df[col].fillna(mean_col) - mean_col)

    df["pred_rank"]   = pred.values
    df["residual"]    = df["grade_rank"] - df["pred_rank"]
    df["pred_letter"] = df["pred_rank"].apply(_grade_letter)
    df["actual_letter"] = df["grade_rank"].apply(_grade_letter)

    flagged = df[
        df["actual_letter"].map(_LETTER_ORDER) > df["pred_letter"].map(_LETTER_ORDER)
    ].copy()

    return flagged.sort_values("residual", ascending=False)


# ── Market Model Flag ────────────────────────────────────────────────────────

def compute_market_flag(
    parcels_df: pd.DataFrame,
    mkt_df: pd.DataFrame,
    iqr_multiplier: float,
) -> pd.DataFrame:
    """
    Merge SUBJCOMP market model values onto parcels and flag where the cost
    model AV diverges from the market model value.

    Credibility — all three must pass:
      FCC        : between 1 and 5
      EXADJ1–5   : ≤ 3 of the 5 comps have an excess-adjustment flag
      DIST1–5    : ≥ 3 of the 5 comps have a comparability index < 50

    Flag is IQR-based on (APRTOT / CALCMKTVAL) across credible parcels only:
      HIGH = cost model significantly above market (possible over-assessment)
      LOW  = market model significantly above cost (possible under-assessment)

    Adds columns: calcmktval, fcc, mkt_credible, cost_mkt_ratio, mkt_flag
    """
    df = parcels_df.copy()

    if mkt_df.empty or "calcmktval" not in mkt_df.columns:
        df["calcmktval"]   = np.nan
        df["fcc"]          = np.nan
        df["mkt_credible"] = False
        df["cost_mkt_ratio"] = np.nan
        df["mkt_flag"]     = ""
        return df

    df = df.merge(mkt_df, on="parid", how="left")

    # ── Credibility ───────────────────────────────────────────────────────────
    fcc_ok = df["fcc"].between(1, 5, inclusive="both")

    exadj_cols = [c for c in ("exadj1", "exadj2", "exadj3", "exadj4", "exadj5")
                  if c in df.columns]
    if exadj_cols:
        exadj_flagged = df[exadj_cols].apply(
            lambda row: sum(
                1 for v in row
                if pd.notna(v) and str(v).strip() not in ("", "0")
            ),
            axis=1,
        )
        exadj_ok = exadj_flagged <= 3
    else:
        exadj_ok = pd.Series(True, index=df.index)

    dist_cols = [c for c in ("dist1", "dist2", "dist3", "dist4", "dist5")
                 if c in df.columns]
    if dist_cols:
        dist_under_50 = df[dist_cols].apply(
            lambda row: sum(
                1 for v in row
                if pd.notna(v) and pd.to_numeric(v, errors="coerce") is not None
                and float(pd.to_numeric(v, errors="coerce")) < 50
            ),
            axis=1,
        )
        dist_ok = dist_under_50 >= 3
    else:
        dist_ok = pd.Series(True, index=df.index)

    df["mkt_credible"] = (
        fcc_ok & exadj_ok & dist_ok
        & df["calcmktval"].notna()
        & (df["calcmktval"] > 0)
    )

    # ── Cost vs. market ratio ─────────────────────────────────────────────────
    df["cost_mkt_ratio"] = np.where(
        df["mkt_credible"] & (df["calcmktval"] > 0) & df["aprtot"].notna(),
        df["aprtot"] / df["calcmktval"],
        np.nan,
    )

    # ── IQR flagging ─────────────────────────────────────────────────────────
    df["mkt_flag"] = ""
    valid = df["cost_mkt_ratio"].dropna()
    if len(valid) >= 5:
        q1  = valid.quantile(0.25)
        q3  = valid.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            df.loc[df["cost_mkt_ratio"] > q3 + iqr_multiplier * iqr, "mkt_flag"] = "HIGH"
            df.loc[df["cost_mkt_ratio"] < q1 - iqr_multiplier * iqr, "mkt_flag"] = "LOW"

    return df


# ── Time Adjustment ──────────────────────────────────────────────────────────

def time_adjust_sales(
    sales_df: pd.DataFrame,
    lien_date: str = "2026-01-01",
) -> tuple[pd.DataFrame, float]:
    """
    Time-adjust sale prices to the lien date using log-linear regression.

    Estimates a monthly appreciation rate from the neighborhood's own sales
    and adjusts each sale price forward to the lien date, so all ratios
    reflect value as of a common point in time.

    Formula: adj_price = price × exp(monthly_rate × months_to_lien)
             adj_ratio = aprtot / adj_price

    Parameters
    ----------
    sales_df  : DataFrame from get_sales() with price, aprtot, saledt columns.
    lien_date : Target date as 'YYYY-MM-DD'. Default 1/1/2026 for 2027 triennial.

    Returns
    -------
    df            : Copy of sales_df with added columns:
                      months_to_lien, adj_price, adj_ratio
    monthly_rate  : Estimated monthly appreciation rate as a decimal
                    (e.g. 0.005 = 0.5%/month). Zero if < 5 sales.
    """
    df = sales_df.copy()
    lien = pd.Timestamp(lien_date)

    saledt = pd.to_datetime(df["saledt"])
    df["months_to_lien"] = (
        (lien.year  - saledt.dt.year)  * 12
        + (lien.month - saledt.dt.month)
    ).astype(float)

    # Log-linear regression: log(price) ~ months_since_lien
    # x = months since lien (negative for past sales)
    # coefficient b > 0 → prices rising over time
    monthly_rate = 0.0
    valid = df[(df["price"] > 0) & df["months_to_lien"].notna()]

    if len(valid) >= 5:
        x = -valid["months_to_lien"].values          # negative for past sales
        y = np.log(valid["price"].values.astype(float))
        X = np.column_stack([np.ones(len(valid)), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            monthly_rate = float(coeffs[1])
        except np.linalg.LinAlgError:
            monthly_rate = 0.0

    df["adj_price"] = df["price"] * np.exp(monthly_rate * df["months_to_lien"])
    df["adj_ratio"] = np.where(
        df["adj_price"] > 0,
        df["aprtot"] / df["adj_price"],
        np.nan,
    )

    return df, monthly_rate


# ── Grade Distribution ───────────────────────────────────────────────────────

def compute_grade_distribution(grades: pd.Series) -> tuple[pd.DataFrame, str | None]:
    """
    Return grade distribution sorted from highest to lowest grade quality.

    Parameters
    ----------
    grades : Series of grade codes (may contain nulls/blanks)

    Returns
    -------
    df         : DataFrame with columns grade, count, pct — sorted high to low
    mode_grade : The most common grade code, or None if no data
    """
    clean = grades.dropna().astype(str).str.strip().str.upper()
    clean = clean[clean != ""]
    if clean.empty:
        return pd.DataFrame(columns=["grade", "count", "pct"]), None

    counts = clean.value_counts()
    mode_grade = counts.idxmax()

    df = counts.reset_index()
    df.columns = ["grade", "count"]
    df["pct"] = df["count"] / df["count"].sum() * 100
    df["_rank"] = df["grade"].map(lambda g: GRADE_RANK.get(g, 0))
    df = (
        df.sort_values("_rank", ascending=False)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return df, mode_grade


# ── Frontage Tier Analysis ────────────────────────────────────────────────────

def compute_frontage_tiers(
    land_df: pd.DataFrame,
    base_percentile: float = 85.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Apply graduated excess-frontage tier methodology to front-foot land records.

    Sums FRONACT across all non-Acre (ltype != 'A') llines per parcel to get
    total parcel frontage. Neighborhood thresholds are derived from the
    distribution of parcel-level totals. Returns one row per participating parcel.

    Tier thresholds (based on median and SD of parcel-total FRONACT):
      Normal : total <= median + 0.5 SD               → no adjustment needed
      Tier 1 : median + 0.5 SD < total <= median + 1.0 SD → 5% (flat)
      Tier 2 : median + 1.0 SD < total <= median + 1.5 SD → 10–15%
      Tier 3 : median + 1.5 SD < total <= median + 2.0 SD → 20–25%
      Tier 4 : total > median + 2.0 SD                    → 30%+

    Tier 1 is flat 5%. Tiers 2–4 scale linearly within their ranges.
    Current adjustment = max abs(INFLU) across all non-A llines for the parcel.
    Adj Gap = Suggested % − Current Adj % (positive = under-adjusted).

    Parameters
    ----------
    land_df : DataFrame from get_land_data() (full or pre-filtered by LUC).
              Non-Acre llines are selected internally.

    Returns
    -------
    thresholds : dict with keys median, sd, t1_lo, t1_hi, t2_hi, t3_hi
    result_df  : one row per parcel with columns parid, address, luc,
                 total_fronact, tier, sugg_adj_pct, current_adj, adj_gap.
                 Empty DataFrame if fewer than 5 front-foot parcels.
    """
    # Select non-Acre llines with valid frontage
    ff = land_df[
        (land_df["ltype"] != "A") &
        land_df["fronact"].notna() &
        (land_df["fronact"] > 0)
    ].copy()

    if ff["parid"].nunique() < 5:
        return {}, pd.DataFrame()

    # Parcel-level total frontage (sum across llines)
    parcel_front = (
        ff.groupby("parid")["fronact"]
        .sum()
        .reset_index()
        .rename(columns={"fronact": "total_fronact"})
    )

    # Current excess frontage adjustment: sum abs(INFLU) across all llines where
    # INFL1 = '01', PLUS abs(INFLU2) across all llines where INFL2 = '01'.
    # Each land line carries its own independent adjustment — they must be summed,
    # not max'd.  The excess frontage code can appear on either influence slot.
    ff["_ef_contrib"] = 0.0
    infl1_mask = ff["infl1"].astype(str).str.strip() == "01"
    infl2_mask = ff["infl2"].astype(str).str.strip() == "01"
    if infl1_mask.any():
        ff.loc[infl1_mask, "_ef_contrib"] += ff.loc[infl1_mask, "influ"].fillna(0).abs()
    if infl2_mask.any():
        ff.loc[infl2_mask, "_ef_contrib"] += ff.loc[infl2_mask, "influ2"].fillna(0).abs()

    ef_adj = (
        ff.groupby("parid")["_ef_contrib"]
        .sum()
        .reset_index()
        .rename(columns={"_ef_contrib": "current_adj"})
    )

    # Merge so every parcel gets a value — 0 if no '01' code applied
    parcel_adj = parcel_front[["parid"]].merge(ef_adj, on="parid", how="left")
    parcel_adj["current_adj"] = parcel_adj["current_adj"].fillna(0.0)

    # Land line count per parcel
    lline_count = (
        ff.groupby("parid")["fronact"]
        .count()
        .reset_index()
        .rename(columns={"fronact": "lline_count"})
    )

    # Parcel-level address, LUC, acreage, and land value (first occurrence — parcel-level fields)
    parcel_meta = ff.groupby("parid")[["address", "luc", "calcacres", "aprland"]].first().reset_index()
    parcel_meta = parcel_meta.merge(lline_count, on="parid", how="left")

    # INFL1 summary: sorted, comma-separated unique non-null codes across llines
    infl1_summary = (
        ff.groupby("parid")["infl1"]
        .apply(lambda x: ",".join(sorted(x.dropna().astype(str).unique())) or "—")
        .reset_index()
        .rename(columns={"infl1": "infl1_summary"})
    )
    parcel_meta = parcel_meta.merge(infl1_summary, on="parid", how="left")
    parcel_meta["infl1_summary"] = parcel_meta["infl1_summary"].fillna("—")

    # Neighborhood distribution of parcel totals
    totals       = parcel_front["total_fronact"]
    median_front = float(totals.median())
    sd_front     = float(totals.std(ddof=1))

    # Percentile-based tier boundaries.
    # The excess population (above base_percentile) is divided into 4 equal
    # percentile bands, each carrying a progressively larger discount.
    tier_band = (100.0 - base_percentile) / 4.0
    t1_lo = float(np.percentile(totals, base_percentile))
    t1_hi = float(np.percentile(totals, base_percentile + tier_band))
    t2_hi = float(np.percentile(totals, base_percentile + tier_band * 2))
    t3_hi = float(np.percentile(totals, base_percentile + tier_band * 3))

    thresholds = {
        "median":          median_front,
        "sd":              sd_front,
        "base_percentile": base_percentile,
        "tier_band":       tier_band,
        "t1_lo":           t1_lo,
        "t1_hi":           t1_hi,
        "t2_hi":           t2_hi,
        "t3_hi":           t3_hi,
    }

    def _round5(pct: float) -> float:
        """Round to the nearest 5 (e.g. 5, 10, 15, 20, 25...)."""
        return float(round(pct / 5) * 5)

    def _tier_and_sugg(total: float) -> tuple[str, float]:
        if pd.isna(total) or total <= 0:
            return "—", np.nan
        if total <= t1_lo:
            return "Normal", 0.0
        if total <= t1_hi:
            return "Tier 1 (5%)", 5.0
        if total <= t2_hi:
            pct = 10.0 + (total - t1_hi) / (t2_hi - t1_hi) * 5.0
            return "Tier 2 (10–15%)", _round5(pct)
        if total <= t3_hi:
            pct = 20.0 + (total - t2_hi) / (t3_hi - t2_hi) * 5.0
            return "Tier 3 (20–25%)", _round5(pct)
        tier_width = max(t3_hi - t2_hi, 1.0)
        pct = min(30.0 + (total - t3_hi) / tier_width * 5.0, 35.0)
        return "Tier 4 (30%+)", _round5(pct)

    tier_vals = parcel_front["total_fronact"].apply(_tier_and_sugg)
    parcel_front["tier"]         = [t for t, _ in tier_vals]
    parcel_front["sugg_adj_pct"] = [s for _, s in tier_vals]

    # Assemble parcel-level result
    result = parcel_meta.merge(parcel_front, on="parid", how="left")
    result = result.merge(parcel_adj, on="parid", how="left")

    # $/ft for each parcel (used for rate derivation and data quality flagging)
    result["dol_per_ft"] = (result["aprland"] / result["total_fronact"]).round(0)

    # Reference rate: median $/ft of Normal-tier parcels — informational only.
    # Under the consistency framework, this does not drive suggested values.
    normal_rows = result[result["tier"] == "Normal"]
    if len(normal_rows) >= 3:
        normal_rate = float(
            (normal_rows["aprland"] / normal_rows["total_fronact"]).median()
        )
    else:
        normal_rate = float(result["dol_per_ft"].median())
    thresholds["normal_rate"] = normal_rate

    # ── Peer-median suggested adjustment per tier ──────────────────────────────
    # Framework: internal consistency and proportionality.
    # For each excess tier, the suggested adjustment is the median of existing
    # non-zero '01' adjustments among peer parcels in that tier (rounded to the
    # nearest 5%).  Falls back to proportional formula defaults when fewer than
    # 3 peers carry a non-zero adjustment.
    #
    # Defaults: Tier 1 = 5%, Tier 2 = 10%, Tier 3 = 20%, Tier 4 = 30%.
    # These represent a proportional scale, not a market-derived rate.
    _TIER_DEFAULTS: dict[str, float] = {
        "Tier 1 (5%)":     5.0,
        "Tier 2 (10–15%)": 10.0,
        "Tier 3 (20–25%)": 20.0,
        "Tier 4 (30%+)":   30.0,
    }
    tier_sugg_map: dict[str, float] = {"Normal": 0.0}
    tier_sources:  dict[str, str]   = {"Normal": "n/a"}

    for tier_label, default_pct in _TIER_DEFAULTS.items():
        peers = result.loc[
            (result["tier"] == tier_label) & (result["current_adj"] > 0),
            "current_adj",
        ]
        if len(peers) >= 3:
            tier_sugg_map[tier_label] = float(round(peers.median() / 5) * 5)
            tier_sources[tier_label]  = "peer"
        else:
            tier_sugg_map[tier_label] = default_pct
            tier_sources[tier_label]  = "default"

    result["tier_sugg_pct"] = result["tier"].map(tier_sugg_map).fillna(0.0)
    thresholds["tier_sugg_map"] = tier_sugg_map
    thresholds["tier_sources"]  = tier_sources

    # Base value: reverse out the current excess-frontage adjustment to get what
    # the parcel would be worth with no '01' INFLU on it.  For parcels with no
    # current adjustment, base_value == aprland.
    result["base_value"] = np.where(
        result["current_adj"] > 0,
        result["aprland"] / (1.0 - result["current_adj"] / 100.0),
        result["aprland"],
    )

    # Suggested AV under the consistency framework:
    #   Normal parcels: restore to base_value (remove any erroneous '01' adj)
    #   Excess parcels: apply tier_sugg_pct to base_value
    normal_mask = result["tier"] == "Normal"
    result["sugg_aprland"] = np.where(
        normal_mask,
        result["base_value"],
        (result["base_value"] * (1.0 - result["tier_sugg_pct"] / 100.0)).round(0),
    )

    # Target INFLU % = tier_sugg_pct for excess parcels, 0 for Normal.
    result["target_influ"] = np.where(
        normal_mask, 0.0, result["tier_sugg_pct"]
    ).round(1)

    # Adj Gap: positive = under-adjusted, negative = over-adjusted, zero = correct
    result["adj_gap"] = (result["target_influ"] - result["current_adj"]).round(0)

    # DATA QUALITY flag: $/ft more than 2 SD below the neighborhood mean.
    # Indicates the current land value is anomalously low relative to frontage
    # and should be investigated before any excess-frontage adjustment is applied.
    dpf = result["dol_per_ft"].dropna()
    if len(dpf) >= 5:
        mean_dpf = float(dpf.mean())
        sd_dpf   = float(dpf.std(ddof=1))
        result["ft_flag"] = np.where(
            result["dol_per_ft"] < mean_dpf - 2.0 * sd_dpf,
            "DATA QUALITY",
            "",
        )
    else:
        result["ft_flag"] = ""

    return thresholds, result
