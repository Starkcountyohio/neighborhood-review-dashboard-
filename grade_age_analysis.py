"""
grade_age_analysis.py — Grade vs. Physical Characteristics: Partial Correlation
Stark County Auditor, Mass Appraisal Analytics

Hypothesis: Year Built (YRBLT) independently predicts Grade after controlling
for physical size, amenities, and sketch complexity (ADDN table).

Model 1 — Baseline: SFLA, YRBLT, Beds, Full Baths, Half Baths, Central Air
Model 2 — Enhanced: adds Addition Count, Garage SqFt, Amenity Count from ADDN

If YRBLT remains significant in Model 2, age independently predicts grade
beyond ALL physical characteristics — strong evidence of grade creep.

Run with:  py grade_age_analysis.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import oracledb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from oracle_conn import connect

from config import DB_CONFIG, TAX_YEAR, NBHD_PREFIXES, AREA_FIELD

# ── Grade → numeric rank ──────────────────────────────────────────────────────

GRADE_RANK = {
    'X+': 16, 'X': 15, 'X-': 14,
    'A+': 13, 'A': 12, 'A-': 11,
    'B+': 10, 'B':  9, 'B-':  8,
    'C+':  7, 'C':  6, 'C-':  5,
    'D+':  4, 'D':  3, 'D-':  2,
    'E':   1,
}

# ── Sketch feature code sets ──────────────────────────────────────────────────

GARAGE_CODES  = {"GF", "GB", "CF", "CB"}          # garage / carport
AMENITY_CODES = {                                   # non-living, non-garage
    "DB", "DF",                                     # deck
    "PB", "PF", "LB", "LF",                        # porch (covered/enclosed)
    "OB", "OF", "IB", "IF",                        # patio (covered/enclosed)
    "YB", "YF",                                     # bay window
    "BB", "BF",                                     # breezeway
}
LIVING_CODES  = {"1AB", "1AF", "2AB", "2AF", "2AL", "2AR"}  # floor additions

# ── Parcel query ──────────────────────────────────────────────────────────────

PREFIX_LIST = ", ".join(f"'{p}'" for p in NBHD_PREFIXES)

SQL_PARCELS = f"""
    SELECT
        p.parid,
        p.nbhd,
        d.grade,
        d.yrblt,
        d.{AREA_FIELD}  AS sfla,
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
      AND  SUBSTR(p.nbhd, 1, 2) IN ({PREFIX_LIST})
      AND  d.grade        IS NOT NULL
      AND  d.yrblt        IS NOT NULL
      AND  d.{AREA_FIELD} IS NOT NULL
      AND  d.{AREA_FIELD} > 0
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def discover_columns(conn, table: str) -> list[str]:
    """Return uppercase column names for a table, resolved via its PUBLIC synonym."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.column_name "
            "FROM   all_synonyms s "
            "JOIN   all_tab_columns c "
            "       ON c.owner = s.table_owner AND c.table_name = s.table_name "
            "WHERE  s.owner = 'PUBLIC' AND s.synonym_name = :tbl "
            "ORDER BY c.column_id",
            {"tbl": table.upper()},
        )
        return [row[0].upper() for row in cur.fetchall()]


def find_col(available: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in the table."""
    avail_set = set(available)
    for c in candidates:
        if c.upper() in avail_set:
            return c.upper()
    return None


def classify_addn_rows(addn_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw ADDN rows (one row per addition line), classify each line and
    aggregate per parcel into: addition_count, garage_sqft, amenity_count.
    """
    # Normalise floor code columns — they may have reserved-word names
    floor_col_map = {"lower": "lwr", "first": "fst", "second": "sec", "third": "thr"}
    addn = addn_raw.rename(columns=floor_col_map)

    def codes_on_line(row) -> set:
        raw = [row.get("lwr"), row.get("fst"), row.get("sec"), row.get("thr")]
        return {str(v).strip().upper() for v in raw if v and str(v).strip()} - {""}

    addn["_codes"]   = addn.apply(codes_on_line, axis=1)
    addn["_garage"]  = addn["_codes"].apply(lambda c: bool(c & GARAGE_CODES))
    addn["_amenity"] = addn["_codes"].apply(lambda c: bool(c & AMENITY_CODES))
    addn["area"]     = pd.to_numeric(addn["area"], errors="coerce").fillna(0)

    return (
        addn.groupby("parid")
        .agg(
            addition_count=("parid", "count"),
            garage_sqft   =("area",  lambda x: x[addn.loc[x.index, "_garage"]].sum()),
            amenity_count =("_amenity", "sum"),
        )
        .reset_index()
    )


def fit_ols(df: pd.DataFrame, predictors: list[str], label: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(df[predictors])
    y = df["grade_rank"]
    model = sm.OLS(y, X).fit()
    return model, X, predictors


def print_model(model, X, predictors: list[str], rename: dict, label: str):
    sep = "-" * 65
    print(f"\n{sep}")
    print(f" {label}")
    print(f" R² = {model.rsquared:.4f}   Adj. R² = {model.rsquared_adj:.4f}"
          f"   n = {int(model.nobs):,}")
    print(sep)

    coef_df = pd.DataFrame({
        "Coef":    model.params,
        "Std Err": model.bse,
        "t":       model.tvalues,
        "p-value": model.pvalues,
        "Sig":     model.pvalues.apply(
                       lambda p: "***" if p < 0.001
                            else "**"  if p < 0.01
                            else "*"   if p < 0.05
                            else "."   if p < 0.10
                            else ""),
    })
    coef_df.index = coef_df.index.map(rename)
    print(coef_df.to_string(float_format="{:.4f}".format))
    print("  Signif: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")

    vif_df = pd.DataFrame({
        "Variable": [rename.get(p, p) for p in predictors],
        "VIF":      [variance_inflation_factor(X[predictors].values, i)
                     for i in range(len(predictors))],
    }).sort_values("VIF", ascending=False)
    print(f"\n  VIF (multicollinearity):")
    print(vif_df.to_string(index=False, float_format="{:.1f}".format))
    print("  VIF > 10 = problematic")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tax_year_int = int(TAX_YEAR)

    print(f"\nConnecting to {DB_CONFIG['dsn']} …")
    try:
        conn = connect()
    except oracledb.Error as exc:
        sys.exit(f"Connection failed: {exc}")

    # ── Step 1: Parcel data ───────────────────────────────────────────────────

    print("Querying parcels …")
    with conn.cursor() as cur:
        cur.execute(SQL_PARCELS, {"tax_year": tax_year_int})
        cols = [c[0].lower() for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    print(f"  {len(df):,} rows fetched, {df['nbhd'].nunique():,} neighborhoods")

    # ── Step 2: Discover ADDN schema ─────────────────────────────────────────

    print("\nDiscovering ADDN table columns …")
    addn_cols = discover_columns(conn, "ADDN")
    if not addn_cols:
        print("  WARNING: ADDN table not found or no access. Skipping sketch features.")
        addn_df = None
    else:
        print(f"  Columns: {', '.join(addn_cols)}")

        # Identify the four floor-level code columns
        lower_col  = find_col(addn_cols, ["LOWER",  "LFLR",  "LOLEVEL",  "LOWER_LVL"])
        first_col  = find_col(addn_cols, ["FIRST",  "FFLR",  "FIRSTFLR", "FIRST_FLR"])
        second_col = find_col(addn_cols, ["SECOND", "SFLR",  "SECFLR",   "SECOND_FLR"])
        third_col  = find_col(addn_cols, ["THIRD",  "TFLR",  "THRFLR",   "THIRD_FLR"])
        has_cur       = "CUR"       in addn_cols
        has_deactivat = "DEACTIVAT" in addn_cols

        floor_cols = [c for c in [lower_col, first_col, second_col, third_col] if c]
        print(f"  Floor code columns identified: {floor_cols}")
        if not floor_cols:
            print("  WARNING: No floor code columns identified. Sketch features unavailable.")
            addn_df = None
        else:
            # ── Step 3: ADDN aggregation query ───────────────────────────────

            cur_filter       = "AND a.cur       = 'Y'"       if has_cur       else ""
            deactivat_filter = "AND a.deactivat IS NULL"     if has_deactivat else ""

            # LOWER/FIRST/SECOND/THIRD must be quoted — Oracle treats them as
            # built-ins / keywords when unquoted.
            # Line number column is LLINE in this IasWorld instance.
            lline_col = "lline" if "LLINE" in addn_cols else "line"

            SQL_ADDN = f"""
                SELECT
                    a.parid,
                    a."LOWER",
                    a."FIRST",
                    a."SECOND",
                    a."THIRD",
                    a.area
                FROM addn a
                WHERE a.taxyr      = :tax_year
                  AND a.card       = 1
                  AND a.{lline_col} > 0
                  {cur_filter}
                  {deactivat_filter}
            """

            print("Querying ADDN sketch features …")
            with conn.cursor() as cur:
                cur.execute(SQL_ADDN, {"tax_year": tax_year_int})
                acols = [c[0].lower() for c in cur.description]
                addn_raw = pd.DataFrame(cur.fetchall(), columns=acols)
            print(f"  {len(addn_raw):,} addition lines fetched")
            addn_df = classify_addn_rows(addn_raw)
            print(f"  {len(addn_df):,} parcels with sketch additions")

    conn.close()

    # ── Step 4: Encode parcel predictors ─────────────────────────────────────

    def encode_heat(val):
        if pd.isna(val): return 0
        s = str(val).strip().upper()
        return 0 if s in ("", "N", "0", "NO") else 1

    df["grade_clean"] = df["grade"].astype(str).str.strip().str.upper()
    df["grade_rank"]  = df["grade_clean"].map(GRADE_RANK)
    df["heat_bin"]    = df["heat"].apply(encode_heat)
    for col in ["sfla", "yrblt", "rmbed", "fixbath", "fixhalf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["grade_rank", "sfla", "yrblt"])
    print(f"\n  Dropped {before - len(df):,} rows (null grade/sfla/yrblt)")

    # ── Step 5: Merge ADDN features ───────────────────────────────────────────

    if addn_df is not None:
        for col in ["addition_count", "garage_sqft", "amenity_count"]:
            addn_df[col] = pd.to_numeric(addn_df[col], errors="coerce").fillna(0)
        df = df.merge(addn_df[["parid", "addition_count", "garage_sqft", "amenity_count"]],
                      on="parid", how="left")
        # Parcels with no additions get zero
        df["addition_count"] = df["addition_count"].fillna(0)
        df["garage_sqft"]    = df["garage_sqft"].fillna(0)
        df["amenity_count"]  = df["amenity_count"].fillna(0)

    # ── Step 6: Grade distribution ────────────────────────────────────────────

    rank_to_grade = {v: k for k, v in GRADE_RANK.items()}
    grade_counts  = (
        df["grade_rank"].value_counts()
        .sort_index(ascending=False)
        .rename(index=rank_to_grade)
    )

    sep = "=" * 65
    print(f"\n{sep}")
    print(" GRADE vs. PHYSICAL CHARACTERISTICS — Partial Correlation")
    print(f" Tax Year: {TAX_YEAR}")
    print(sep)

    print("\n-- Grade Distribution ------------------------------------------")
    print(grade_counts.to_string())

    # ── Step 7: Model 1 — Baseline ────────────────────────────────────────────

    BASE_PRED = ["sfla", "yrblt", "rmbed", "fixbath", "fixhalf", "heat_bin"]
    RENAME = {
        "const":          "Intercept",
        "sfla":           "SFLA (sq ft)",
        "yrblt":          "Yr Built",
        "rmbed":          "Bedrooms",
        "fixbath":        "Full Baths",
        "fixhalf":        "Half Baths",
        "heat_bin":       "Central Air",
        "addition_count": "Addition Count",
        "garage_sqft":    "Garage SqFt",
        "amenity_count":  "Amenity Count",
    }

    df_base = df[["grade_rank"] + BASE_PRED].dropna()
    print(f"\n  Model 1 sample: {len(df_base):,} parcels")
    m1, X1, p1 = fit_ols(df_base, BASE_PRED, "Model 1")
    print_model(m1, X1, p1, RENAME, "Model 1 — Baseline (no sketch features)")

    # ── Step 8: Model 2 — Enhanced with ADDN ─────────────────────────────────

    if addn_df is not None:
        ENH_PRED = BASE_PRED + ["addition_count", "garage_sqft", "amenity_count"]
        df_enh = df[["grade_rank"] + ENH_PRED].dropna()
        print(f"\n  Model 2 sample: {len(df_enh):,} parcels")
        m2, X2, p2 = fit_ols(df_enh, ENH_PRED, "Model 2")
        print_model(m2, X2, p2, RENAME, "Model 2 — Enhanced (with sketch features)")

        # ── Step 9: Interpretation ────────────────────────────────────────────

        print(f"\n-- Interpretation ----------------------------------------------")

        yrblt_b  = m1.params["yrblt"]
        yrblt_b2 = m2.params["yrblt"]
        yrblt_p2 = m2.pvalues["yrblt"]
        r2_lift  = m2.rsquared - m1.rsquared

        print(f"\n  R² lift from sketch features: +{r2_lift:.4f}")
        print(f"  (sketch complexity explains an additional {r2_lift*100:.1f}% of grade variance)")

        print(f"\n  YRBLT coefficient:  Model 1 = {yrblt_b:+.5f}"
              f"   Model 2 = {yrblt_b2:+.5f}  (p = {yrblt_p2:.4f})")

        shrink = (yrblt_b - yrblt_b2) / abs(yrblt_b) * 100 if yrblt_b != 0 else 0
        print(f"  YRBLT coefficient shrank by {shrink:.1f}% after adding sketch features.")

        if yrblt_p2 < 0.05:
            print(f"\n  FINDING: Yr Built is STILL significant after controlling for")
            print(f"  sketch complexity (additions, garage size, amenities).")
            print(f"  Age predicts grade independently of ALL measured physical")
            print(f"  characteristics — this is the clearest evidence yet of")
            print(f"  systematic grade creep.")
        else:
            print(f"\n  FINDING: Yr Built is no longer significant once sketch")
            print(f"  complexity is controlled for. The earlier age effect was")
            print(f"  likely capturing legitimate differences in construction")
            print(f"  quality that show up in the sketch (more additions, larger")
            print(f"  garages) rather than pure grade inflation.")

        addn_sig = m2.pvalues["addition_count"] < 0.05
        gar_sig  = m2.pvalues["garage_sqft"]    < 0.05
        print(f"\n  Addition Count significant: {'YES' if addn_sig else 'no'}"
              f"  (coef {m2.params['addition_count']:+.4f}, p={m2.pvalues['addition_count']:.4f})")
        print(f"  Garage SqFt significant:    {'YES' if gar_sig  else 'no'}"
              f"  (coef {m2.params['garage_sqft']:+.6f}, p={m2.pvalues['garage_sqft']:.4f})")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
