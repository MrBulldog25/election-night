import math, json, random
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kstest

from gaussian import GaussianProcessModel, GaussianProcessConfig

PRIORS = "data/priors/NJ_county_priors.csv"
CLEAN  = "data/clean/NJ_county_cleaned.csv"

TrialResult = namedtuple("TrialResult",
    "n_obs mean_pct ks cal68 cal95 width68 width95 resp_near resp_far statewide_sd"
)

def load_realized() -> Dict[str, Tuple[int,int,int]]:
    clean_df  = pd.read_csv(CLEAN)
    priors_df = pd.read_csv(PRIORS)

    total_24 = (
        clean_df["2024_Pre_D"].sum()
        + clean_df["2024_Pre_R"].sum()
        + clean_df["2024_Pre_O"].sum()
    )
    total_baseline = priors_df["baseline_total_votes"].sum()
    frac = total_baseline / max(total_24, 1)

    out = {}
    for _, r in clean_df.iterrows():
        county = str(r.get("County") or r.get("county"))
        out[county] = (
            int(math.floor(r["2024_Pre_D"] * frac)),
            int(math.floor(r["2024_Pre_R"] * frac)),
            int(math.floor(r["2024_Pre_O"] * frac)),
        )
    return out

def realized_share(realized: Dict[str, Tuple[int,int,int]], county: str) -> float:
    d, r, _ = realized[county]
    return d / max(d + r, 1)

def neighbors_by_kernel(model: GaussianProcessModel, k: int = 10) -> Dict[int, List[int]]:
    S = model.Sigma.copy()
    n = S.shape[0]
    S -= model.cfg.sigma_s2 * np.ones((n, n))
    np.fill_diagonal(S, 0.0)
    nbrs = {}
    for i in range(n):
        order = np.argsort(-S[i])
        nbrs[i] = [j for j in order if j != i][:k]
    return nbrs

def run_one_setting(sigma_s2: float, tau2: float, length_scale: float,
                    trials_per_k: int = 40, ks_bins: int = 200,
                    seed: int = 42) -> List[TrialResult]:
    random.seed(seed)
    np.random.seed(seed)

    realized = load_realized()
    cfg = GaussianProcessConfig(
        sigma_s2=sigma_s2, tau2=tau2, length_scale=length_scale,
        sigma_eps2=1e-4, use_turnout_updates=True
    )
    base_model = GaussianProcessModel(PRIORS, cfg=cfg)
    counties = base_model.counties
    name_to_idx = {c:i for i,c in enumerate(counties)}

    nbrs = neighbors_by_kernel(base_model, k=10)
    anti_nbrs = {i: list(reversed(nbrs[i])) for i in nbrs}

    results: List[TrialResult] = []
    for n_obs in [0, 1, 3, 10]:
        all_pcts: List[float] = []
        cover68 = cover95 = 0
        width68s: List[float] = []
        width95s: List[float] = []
        resp_near_vals: List[float] = []
        resp_far_vals: List[float] = []
        statewide_sds: List[float] = []

        for _ in range(trials_per_k):
            model = GaussianProcessModel(PRIORS, cfg=cfg)
            selected = random.sample(counties, n_obs) if n_obs > 0 else []
            for c in selected:
                d, r, _ = realized[c]
                model.observe_votes(c, d, r)
            model.update_posterior()

            statewide = model.sample_statewide(n_draws=1500)
            statewide_sds.append(statewide["sd"])

            ci68 = model.county_share_ci(z=1.0)
            ci95 = model.county_share_ci(z=1.96)
            mean  = model.county_share_mean()
            for c in counties:
                if c in selected:
                    continue
                obs = realized_share(realized, c)
                pct = model.county_result_surprise(c, obs)["percentile"]
                all_pcts.append(pct)
                lo68, hi68 = float(ci68.loc[c, "lo"]), float(ci68.loc[c, "hi"])
                lo95, hi95 = float(ci95.loc[c, "lo"]), float(ci95.loc[c, "hi"])
                cover68 += int(lo68 <= obs <= hi68)
                cover95 += int(lo95 <= obs <= hi95)
                width68s.append(hi68 - lo68)
                width95s.append(hi95 - lo95)

            if n_obs >= 1:
                anchor = selected[0]
                i = name_to_idx[anchor]
                prior_model = GaussianProcessModel(PRIORS, cfg=cfg)
                prior_mean = prior_model.county_share_mean().to_numpy()
                post_mean  = model.county_share_mean().to_numpy()
                near_idx = nbrs[i]
                far_idx  = anti_nbrs[i]
                resp_near_vals.append(np.mean(np.abs(post_mean[near_idx] - prior_mean[near_idx])))
                resp_far_vals.append(np.mean(np.abs(post_mean[far_idx]  - prior_mean[far_idx])))

        m = len(all_pcts)
        if m == 0:
            continue

        mean_pct = float(np.mean(all_pcts))
        ks_stat = float(kstest(all_pcts, 'uniform').statistic)

        denom = len(counties) - n_obs
        cal68 = cover68 / max(denom * trials_per_k, 1)
        cal95 = cover95 / max(denom * trials_per_k, 1)

        results.append(TrialResult(
            n_obs=n_obs,
            mean_pct=mean_pct,
            ks=ks_stat,
            cal68=cal68,
            cal95=cal95,
            width68=float(np.mean(width68s)),
            width95=float(np.mean(width95s)),
            resp_near=float(np.mean(resp_near_vals) if resp_near_vals else 0.0),
            resp_far=float(np.mean(resp_far_vals) if resp_far_vals else 0.0),
            statewide_sd=float(np.mean(statewide_sds)),
        ))

    return results

def score_results(results: List[TrialResult]) -> float:
    score = 0.0
    for r in results:
        score += 2.0 * abs(r.mean_pct - 0.5) + 1.5 * r.ks
        score += 1.0 * abs(r.cal68 - 0.68) + 0.7 * abs(r.cal95 - 0.95)
        score += 2.0 * r.width68 + 1.0 * r.width95
        score +=  - 3.0 * max(r.resp_near - r.resp_far, 0.0)
        score += 3.0 * abs(r.statewide_sd - 0.025)
    return score

def main():
    grid_sigma_s2 = [0.003, 0.005, 0.007]
    grid_tau2     = [0.02, 0.04, 0.06]
    grid_l        = [1.8, 2.1, 2.4, 2.7]

    all_runs = []
    for s2 in grid_sigma_s2:
        for t2 in grid_tau2:
            for l in grid_l:
                print(f"\n=== Testing sigma_s2={s2:.3f}, tau2={t2:.2f}, ell={l:.2f} ===")
                res = run_one_setting(s2, t2, l, trials_per_k=40)
                sc  = score_results(res)
                all_runs.append((sc, s2, t2, l, res))
                for r in res:
                    print(f"  n_obs={r.n_obs:2d}  "
                          f"mean_pct={r.mean_pct:5.3f}  KS={r.ks:5.3f}  "
                          f"cov68={r.cal68:5.3f}  cov95={r.cal95:5.3f}  "
                          f"w68={r.width68:5.3f}  w95={r.width95:5.3f}  "
                          f"resp_near={r.resp_near:5.3f}  resp_far={r.resp_far:5.3f}  "
                          f"state_sd={r.statewide_sd:5.3f}")

    all_runs.sort(key=lambda x: x[0])
    best = all_runs[0]
    print("\n=== Top settings (lower score = better) ===")
    for rank, (sc, s2, t2, l, res) in enumerate(all_runs[:5], start=1):
        print(f"{rank:>2}. score={sc:7.3f}  sigma_s2={s2:.3f}  tau2={t2:.2f}  ell={l:.2f}")
        for r in res:
            print(f"    n_obs={r.n_obs:2d} mean_pct={r.mean_pct:5.3f} KS={r.ks:5.3f} "
                  f"cov68={r.cal68:5.3f} cov95={r.cal95:5.3f} w68={r.width68:5.3f} w95={r.width95:5.3f} "
                  f"respΔ={r.resp_near - r.resp_far:5.3f} state_sd={r.statewide_sd:5.3f}")

    out = []
    for sc, s2, t2, l, res in all_runs:
        out.append({
            "score": sc, "sigma_s2": s2, "tau2": t2, "ell": l,
            "results": [r._asdict() for r in res]
        })
    with open("backtest_grid_results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
