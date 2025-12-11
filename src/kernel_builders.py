from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from build_priors import locate_county_col, logit


DEMO_KEEP = [
    "Population 2020",
    "VAP",
    "Number of Households",
    "Percent White VAP",
    "Percent Black VAP",
    "Percent Asian VAP",
    "Percent Native VAP",
    "Percent Latino VAP",
    "Percent High School Attainment",
    "Percent Bachelors Attainment",
    "Percent Graduate Attainment",
    "Percent Senior VAP",
    "Percent Income <25k Households",
    "Percent Income 25-50k Households",
    "Percent Income 50-100k Households",
    "Percent Income 100-200k Households",
    "Percent Income >200k Households",
    "Population Density (Per Square KM)",
]

ELECTION_COL_PATTERN = re.compile(r"^(?P<year>\d{4})_(?P<office>[A-Za-z0-9]+)_(?P<party>[DRO])$")


def _list_clean_files(clean_dir: Path) -> List[Path]:
    return sorted(clean_dir.glob("*_county_cleaned.csv"))


def _coerce_year(year_str: str) -> Optional[int]:
    try:
        yr = int(year_str)
    except ValueError:
        return None
    if yr < 1900 or yr > 2050:
        return None
    return yr


def _standardize_feature_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    mats = []
    for col in cols:
        s = df[col].astype(float)
        mu = float(s.mean(skipna=True))
        sd = float(s.std(skipna=True, ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        z = (s - mu) / sd
        z = z.fillna(0.0)
        mats.append(z.values)
        stats[col] = (mu, sd)
    return np.vstack(mats).T, stats


def _demo_metric_columns(df: pd.DataFrame) -> List[str]:
    banned = {"Population 2020", "VAP", "Number of Households"}
    return [c for c in DEMO_KEEP if c in df.columns and c not in banned]


def _rbf_kernel_weighted(X: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    d2 = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(X ** 2, axis=1) - 2 * X @ X.T
    d2 = np.maximum(d2, 0.0)
    return np.exp(-0.5 * d2 / (length_scale ** 2))


def _weighted_demo_kernel(X: np.ndarray, alpha: np.ndarray, ell_demo: float = 1.0) -> np.ndarray:
    alpha = np.maximum(alpha, 0.0)
    X_scaled = X * np.sqrt(alpha)[None, :]
    d2 = np.sum(X_scaled ** 2, axis=1, keepdims=True) + np.sum(X_scaled ** 2, axis=1) - 2 * X_scaled @ X_scaled.T
    d2 = np.maximum(d2, 0.0) / (ell_demo ** 2)
    return np.exp(-d2)


def build_swing_profiles(
    clean_dir: str | Path = "data/clean",
    output_csv: str | Path = "data/derived/county_swing_profiles.csv",
) -> pd.DataFrame:
    clean_dir = Path(clean_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    files = _list_clean_files(clean_dir)
    if not files:
        raise FileNotFoundError(f"No cleaned county files found in {clean_dir}")

    profiles: Dict[Tuple[str, str], Dict[str, object]] = {}
    swing_cols: List[str] = []

    for path in files:
        state = path.name.split("_", 1)[0]
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        county_col = locate_county_col(df)
        counties = df[county_col].astype(str)

        for idx, county_name in counties.items():
            key = (state, county_name)
            rec = profiles.setdefault(key, {"state": state, "county": county_name})
            for demo_col in DEMO_KEEP:
                if demo_col in df.columns:
                    rec.setdefault(demo_col, df.loc[idx, demo_col])

        election_cols: Dict[Tuple[str, str], Dict[str, str]] = {}
        for col in df.columns:
            m = ELECTION_COL_PATTERN.match(col)
            if not m:
                continue
            office = m.group("office")
            if office.lower() == "gue":
                continue
            year = _coerce_year(m.group("year"))
            if year is None:
                continue
            party = m.group("party")
            key = (office, str(year))
            election_cols.setdefault(key, {})[party] = col

        residuals: Dict[Tuple[str, str], pd.Series] = {}
        for (office, year), cols in election_cols.items():
            D_col, R_col = cols.get("D"), cols.get("R")
            if not D_col or not R_col:
                continue
            D = df[D_col].astype(float)
            R = df[R_col].astype(float)
            N = D + R
            share = (D / N.replace(0, np.nan)).clip(1e-4, 1 - 1e-4)
            m = share.apply(logit)
            weights = N.replace(0, np.nan)
            m_bar = (weights * m).sum(skipna=True) / weights.sum(skipna=True)
            residuals[(office, year)] = m - m_bar

        office_years: Dict[str, List[int]] = {}
        for office, year in {(o, int(y)) for (o, y) in residuals.keys()}:
            office_years.setdefault(office, []).append(year)
        for office, years in office_years.items():
            years_sorted = sorted(years)
            for prev, curr in zip(years_sorted[:-1], years_sorted[1:]):
                prev_key = (office, str(prev))
                curr_key = (office, str(curr))
                if prev_key not in residuals or curr_key not in residuals:
                    continue
                delta = residuals[curr_key] - residuals[prev_key]
                colname = f"delta_r_{state}_{office}_{prev}_{curr}"
                swing_cols.append(colname)
                for idx, val in delta.items():
                    county_name = counties.iloc[idx]
                    key = (state, county_name)
                    rec = profiles.setdefault(key, {"state": state, "county": county_name})
                    rec[colname] = float(val) if pd.notna(val) else np.nan

    swing_cols = sorted(set(swing_cols))
    rows = []
    for (_state, _county), rec in profiles.items():
        rec["n_swing_obs"] = sum(
            1 for c in swing_cols if c in rec and pd.notna(rec.get(c))
        )
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    for col in swing_cols:
        if col not in out_df.columns:
            out_df[col] = np.nan
    ordered_cols = ["state", "county", "n_swing_obs"] + swing_cols + [
        c for c in DEMO_KEEP if c in out_df.columns
    ]
    out_df = out_df[ordered_cols]
    out_df.to_csv(output_csv, index=False)
    return out_df


def compute_shift_kernel(
    profiles_csv: str | Path = "data/derived/county_swing_profiles.csv",
    output_npz: str | Path = "data/derived/swing_kernel.npz",
    ell_shift: float = 1.0,
    reliability_c: float = 3.0,
    office_filter: Optional[Iterable[str]] = ("Pre",),
) -> np.ndarray:
    profiles_csv = Path(profiles_csv)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(profiles_csv)
    swing_cols_all = [c for c in df.columns if c.startswith("delta_r_")]
    if office_filter:
        swing_cols = []
        pat = re.compile(r"^delta_r_[^_]+_(?P<office>[^_]+)_[0-9]{4}_[0-9]{4}$")
        for c in swing_cols_all:
            m = pat.match(c)
            if m and m.group("office") in office_filter:
                swing_cols.append(c)
    else:
        swing_cols = swing_cols_all
    if not swing_cols:
        return np.array([])

    S, swing_stats = _standardize_feature_matrix(df, swing_cols)
    K_base = _rbf_kernel_weighted(S, length_scale=ell_shift)

    n_obs = df.get("n_swing_obs", pd.Series(0, index=df.index)).astype(float)
    r = n_obs / (n_obs + reliability_c)
    r = r.clip(lower=0.0, upper=1.0).values
    K_shift = (r[:, None]) * K_base * (r[None, :])

    county_keys = (df["state"].astype(str) + "|" + df["county"].astype(str)).values
    np.savez(
        output_npz,
        K_shift=K_shift,
        county_keys=county_keys,
        states=df["state"].values,
        counties=df["county"].values,
        ell_shift=ell_shift,
        reliability_c=reliability_c,
        swing_cols=np.array(swing_cols),
        swing_stats=np.array([swing_stats[c] for c in swing_cols], dtype=object),
    )
    return K_shift


def learn_demo_weights(
    profiles_csv: str | Path = "data/derived/county_swing_profiles.csv",
    shift_kernel_npz: str | Path = "data/derived/swing_kernel.npz",
    output_json: str | Path = "data/derived/demo_kernel_weights.json",
    lambda_reg: float = 1e-3,
    ell_demo: float = 1.0,
    restrict_state: Optional[str] = "NJ",
) -> Dict[str, object]:
    profiles_csv = Path(profiles_csv)
    shift_kernel_npz = Path(shift_kernel_npz)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(profiles_csv)
    data = np.load(shift_kernel_npz, allow_pickle=True)
    K_shift = data["K_shift"]
    states = data["states"]
    counties = data["counties"]

    profiles["county_key"] = profiles["state"].astype(str) + "|" + profiles["county"].astype(str)
    key_to_idx = {k: i for i, k in enumerate(profiles["county_key"])}
    order = []
    for s, c in zip(states, counties):
        k = f"{s}|{c}"
        if k not in key_to_idx:
            continue
        order.append(key_to_idx[k])
    profiles = profiles.iloc[order].reset_index(drop=True)

    if restrict_state:
        mask = profiles["state"].str.upper() == restrict_state.upper()
    else:
        mask = np.ones(len(profiles), dtype=bool)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {}

    K_target = K_shift[np.ix_(idx, idx)]
    demo_cols = _demo_metric_columns(profiles)
    if not demo_cols:
        return {}
    X_demo_df = profiles.iloc[idx][demo_cols].astype(float).copy()
    dens_col = "Population Density (Per Square KM)"
    if dens_col in X_demo_df.columns:
        X_demo_df[dens_col] = np.log(X_demo_df[dens_col].clip(lower=1e-3))
    X_demo, _ = _standardize_feature_matrix(X_demo_df, list(X_demo_df.columns))

    d = X_demo.shape[1]
    alpha0 = np.ones(d, dtype=float)
    bounds = [(0.0, None) for _ in range(d)]

    def objective(alpha_vec: np.ndarray) -> float:
        K_demo = _weighted_demo_kernel(X_demo, alpha_vec, ell_demo=ell_demo)
        diff = K_demo - K_target
        return float(np.sum(diff ** 2) + lambda_reg * np.sum(alpha_vec ** 2))

    res = minimize(
        objective,
        alpha0,
        method="L-BFGS-B",
        bounds=bounds,
    )
    alpha_opt = np.maximum(res.x, 0.0)

    payload = {
        "demo_cols": demo_cols,
        "alpha": alpha_opt.tolist(),
        "lambda_reg": lambda_reg,
        "ell_demo": ell_demo,
        "restrict_state": restrict_state,
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": res.message,
    }
    with output_json.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


def build_NJ_composite_covariance(
    priors_csv: str | Path = "data/priors/NJ_county_priors.csv",
    shift_kernel_npz: str | Path = "data/derived/swing_kernel.npz",
    demo_weights_json: str | Path = "data/derived/demo_kernel_weights.json",
    output_npz: str | Path = "data/derived/NJ_composite_kernel.npz",
    config_json: str | Path = "data/derived/kernel_config.json",
    hyperparams: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    priors_csv = Path(priors_csv)
    shift_kernel_npz = Path(shift_kernel_npz)
    demo_weights_json = Path(demo_weights_json)
    output_npz = Path(output_npz)
    config_json = Path(config_json)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    config_json.parent.mkdir(parents=True, exist_ok=True)

    priors = pd.read_csv(priors_csv)
    with demo_weights_json.open() as f:
        weights = json.load(f)
    alpha = np.array(weights["alpha"], dtype=float)
    demo_cols = weights["demo_cols"]
    ell_demo = float(weights.get("ell_demo", 1.0))

    kernel_data = np.load(shift_kernel_npz, allow_pickle=True)
    K_shift = kernel_data["K_shift"]
    states = kernel_data["states"]
    counties = kernel_data["counties"]
    key_to_idx = {f"{s}|{c}": i for i, (s, c) in enumerate(zip(states, counties))}

    nj_counties = priors["county"].astype(str).tolist()
    nj_indices = []
    for c in nj_counties:
        key = f"NJ|{c}"
        if key not in key_to_idx:
            continue
        nj_indices.append(key_to_idx[key])
    if not nj_indices:
        return {}
    K_shift_NJ = K_shift[np.ix_(nj_indices, nj_indices)]

    if not all(col in priors.columns for col in demo_cols):
        missing = [c for c in demo_cols if c not in priors.columns]
        if missing:
            return {}
    X_demo = priors[demo_cols].astype(float).copy()
    dens_col = "Population Density (Per Square KM)"
    if dens_col in X_demo.columns:
        X_demo[dens_col] = np.log(X_demo[dens_col].clip(lower=1e-3))
    X_demo = (X_demo - X_demo.mean()) / X_demo.std(ddof=0).replace(0, 1.0)
    X_demo = X_demo.fillna(0.0)
    K_demo_NJ = _weighted_demo_kernel(X_demo.values, alpha, ell_demo=ell_demo)

    hp_default = {
        "sigma_s2": 0.005,
        "tau_demo2": 0.04,
        "tau_shift2": 0.04,
        "sigma_eps2": 1e-4,
    }
    if hyperparams:
        hp_default.update(hyperparams)
    hp = hp_default

    n = len(nj_counties)
    J = np.ones((n, n), dtype=float)
    I = np.eye(n, dtype=float)
    Sigma = (
        hp["sigma_s2"] * J
        + hp["tau_demo2"] * K_demo_NJ
        + hp["tau_shift2"] * K_shift_NJ
        + hp["sigma_eps2"] * I
    )

    np.savez(
        output_npz,
        Sigma=Sigma,
        counties=np.array(nj_counties),
        K_demo_NJ=K_demo_NJ,
        K_shift_NJ=K_shift_NJ,
        hyperparams=hp,
        demo_cols=np.array(demo_cols),
        alpha=alpha,
        ell_demo=ell_demo,
    )

    cfg_payload = {
        "mode": "composite",
        "state": "NJ",
        "shift_kernel_path": str(shift_kernel_npz),
        "demo_weights_path": str(demo_weights_json),
        "sigma_path": str(output_npz),
        "hyperparams": hp,
        "ell_demo": ell_demo,
        "demo_cols": demo_cols,
    }
    with config_json.open("w") as f:
        json.dump(cfg_payload, f, indent=2)
    return cfg_payload

if __name__ == "__main__":
    build_swing_profiles()
    compute_shift_kernel()
    learn_demo_weights()
    build_NJ_composite_covariance()
