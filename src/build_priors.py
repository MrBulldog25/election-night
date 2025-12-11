import argparse
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

NON_GUESS_PREFIXES = [
    "2017_Gov", "2021_Gov",
    "2016_Pre", "2020_Pre",
    "2018_Sen", "2020_Sen",
]

BASELINE_PREFIX = "Baseline"

COUNTY_NAME_COL_CANDIDATES = ["county", "County.1", "County"]

SHARE_SIGMA_FLOOR = 0.03
SHARE_SIGMA_MULT  = 2.0

TURNOUT_SD_FRAC   = 0.05

CLIP_EPS = 1e-4

def logit(p: float) -> float:
    p = min(max(p, CLIP_EPS), 1 - CLIP_EPS)
    return math.log(p / (1 - p))

def inv_logit(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def locate_county_col(df: pd.DataFrame) -> str:
    for col in COUNTY_NAME_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("Could not find a county name column.")

def compute_two_party_share_and_total(df: pd.DataFrame, prefix: str) -> Tuple[pd.Series, pd.Series]:
    D = df.get(f"{prefix}_D")
    R = df.get(f"{prefix}_R")
    O = df.get(f"{prefix}_O")
    if D is None or R is None:
        raise ValueError(f"Missing columns for prefix {prefix} (need _D and _R).")
    twoparty = (D + R).replace(0, np.nan)
    share = (D / (D + R).replace(0, np.nan)).clip(CLIP_EPS, 1 - CLIP_EPS)
    total_all = D + R + (O if O is not None else 0)
    return share.fillna(0.5), total_all

def empirical_share_volatility(df: pd.DataFrame, prefixes: List[str]) -> pd.Series:
    shares = []
    for pfx in prefixes:
        s, _ = compute_two_party_share_and_total(df, pfx)
        shares.append(s)
    X = pd.concat(shares, axis=1)
    v = X.std(axis=1, ddof=1).fillna(X.mean(axis=1)*0 + 0.05)
    sigma_share = (SHARE_SIGMA_MULT * v).clip(lower=SHARE_SIGMA_FLOOR)
    return sigma_share

def empirical_turnout_volatility(df: pd.DataFrame, _prefixes: List[str]) -> pd.Series:
    sigma = math.sqrt(math.log(1 + TURNOUT_SD_FRAC ** 2))
    return pd.Series(sigma, index=df.index)

def build_priors(df: pd.DataFrame, total_votes: int | None = None) -> pd.DataFrame:
    county_col = locate_county_col(df)
    out = pd.DataFrame()
    out["county"] = df[county_col]

    baseline_dem = df[f"{BASELINE_PREFIX}_D"].astype(float).copy()
    baseline_rep = df[f"{BASELINE_PREFIX}_R"].astype(float).copy()
    baseline_other = df.get(f"{BASELINE_PREFIX}_O", pd.Series(0.0, index=df.index)).astype(float)
    baseline_total = baseline_dem + baseline_rep + baseline_other

    scale_baselines = total_votes is not None
    if scale_baselines:
        statewide_total = float(baseline_total.sum())
        scale = float(total_votes) / statewide_total
        baseline_dem *= scale
        baseline_rep *= scale
        baseline_other *= scale
        baseline_total = baseline_dem + baseline_rep + baseline_other

    p0, _ = compute_two_party_share_and_total(df, BASELINE_PREFIX)
    N0 = baseline_total
    if scale_baselines:
        out["baseline_dem_votes"] = baseline_dem
        out["baseline_rep_votes"] = baseline_rep
        out["baseline_other_votes"] = baseline_other
        out["baseline_total_votes"] = N0
    else:
        out["baseline_dem_votes"] = baseline_dem.astype(int)
        out["baseline_rep_votes"] = baseline_rep.astype(int)
        out["baseline_other_votes"] = baseline_other.astype(int)
        out["baseline_total_votes"] = N0.astype(int)
    out["p0"] = p0
    out["logit_p0"] = out["p0"].apply(logit)

    out["uncert_p_share"] = empirical_share_volatility(df, NON_GUESS_PREFIXES)
    out["uncert_turnout"] = empirical_turnout_volatility(df, NON_GUESS_PREFIXES)

    out["log_turnout_total"] = np.log(out["baseline_total_votes"].clip(lower=1))

    DEMO_KEEP = ["Population 2020", "VAP", "Number of Households",
                 "Percent White VAP", "Percent Black VAP", "Percent Asian VAP", "Percent Native VAP", "Percent Latino VAP",
                 "Percent High School Attainment", "Percent Bachelors Attainment", "Percent Graduate Attainment",
                 "Percent Senior VAP",
                 "Percent Income <25k Households", "Percent Income 25-50k Households", "Percent Income 50-100k Households", "Percent Income 100-200k Households", "Percent Income >200k Households",
                 "Population Density (Per Square KM)"]
    for col in DEMO_KEEP:
        if col in df.columns:
            out[col] = df[col]

    tw = out["baseline_dem_votes"] + out["baseline_rep_votes"]
    statewide_p = (out["baseline_dem_votes"].sum() / tw.sum()) if tw.sum() > 0 else np.nan
    statewide_margin = 2*statewide_p - 1
    print(f"[info] statewide two-party share (baseline): Dem {statewide_p:.3f} (margin {statewide_margin:+.3f})")

    return out

def create_priors_file(state: str, denomination: str, total_votes: int | None = None) -> None:
    infile = f"data/clean/{state}_{denomination}_cleaned.csv"
    outfile = f"data/priors/{state}_{denomination}_priors.csv"
    df = pd.read_csv(infile)
    priors = build_priors(df, total_votes=total_votes)

    col_order = [
        "county",
        "baseline_dem_votes","baseline_rep_votes","baseline_other_votes","baseline_total_votes",
        "p0","logit_p0",
        "uncert_p_share","uncert_turnout",
        "log_turnout_total",
    ] + [c for c in priors.columns if c not in {
        "county","baseline_dem_votes","baseline_rep_votes","baseline_other_votes","baseline_total_votes",
        "p0","logit_p0","uncert_p_share","uncert_turnout","log_turnout_total"}]

    priors = priors[col_order]
    priors.to_csv(outfile, index=False)
    print(f"[ok] wrote {outfile} ({len(priors)} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", help="State abbreviation, e.g. NJ")
    ap.add_argument("denomination", help="Denomination column name, e.g. county or district")
    args = ap.parse_args()

    create_priors_file(args.state, args.denomination.lower())

if __name__ == "__main__":
    create_priors_file("NJ", "county", 3310000)
