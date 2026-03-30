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

  COV             = [ std(ratio_i, ddof=1) / mean(ratio_i) ] × 100
                    Coefficient of Variation.

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
OUTLIER DETECTION
─────────────────────────────────────────────────────────────────────────────

  A value is flagged when:  |value − mean| > threshold × SD
  Threshold is set interactively in the dashboard sidebar.
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
    dict with keys: n, median_ratio, mean_ratio, weighted_mean, cod, cov,
                    prd, prb, prb_pvalue, median_ratio_ci, cod_ci.
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

    # COV
    cov = float((ratios.std(ddof=1) / mean_ratio) * 100)

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
        "cov":             cov,
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
    parcels_df  DataFrame with columns: aprtot (float), sfla (float).

    Returns
    -------
    dict with keys: n, n_with_area, median_av, mean_av,
                    median_av_per_sqft, mean_av_per_sqft,
                    cod_av_per_sqft, cov_av_per_sqft.
    Returns {'error': str} if data are insufficient.
    """
    total_n = len(parcels_df)

    df = parcels_df.dropna(subset=["aprtot", "sfla"])
    df = df[(df["sfla"] > 0) & (df["aprtot"] > 0)].copy()

    if len(df) < 5:
        return {
            "error": (
                f"Insufficient parcels with area data (n={len(df)} of {total_n}). "
                "AV/SqFt uniformity statistics cannot be computed."
            )
        }

    df["av_per_sqft"] = df["aprtot"] / df["sfla"]
    vals = df["av_per_sqft"]

    median_val = float(vals.median())
    mean_val   = float(vals.mean())
    cod = float((np.abs(vals - median_val).mean() / median_val) * 100)
    cov = float((vals.std(ddof=1) / mean_val) * 100)

    return {
        "n":                  total_n,
        "n_with_area":        len(df),
        "median_av":          float(parcels_df["aprtot"].median()),
        "mean_av":            float(parcels_df["aprtot"].mean()),
        "median_av_per_sqft": median_val,
        "mean_av_per_sqft":   mean_val,
        "cod_av_per_sqft":    cod,
        "cov_av_per_sqft":    cov,
    }


# ── Outlier Detection ─────────────────────────────────────────────────────────

def flag_outliers(values: pd.Series, sd_threshold: float) -> pd.Series:
    """
    Return a boolean Series: True where |value − mean| > sd_threshold × SD.
    NaN values are treated as non-outliers (False).
    """
    mean = values.mean()
    sd   = values.std(ddof=1)
    if sd == 0:
        return pd.Series(False, index=values.index)
    return (values - mean).abs() > sd_threshold * sd
