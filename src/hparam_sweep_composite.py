from __future__ import annotations

import json
import math
import random
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kstest

from gaussian import GaussianProcessModel, GaussianProcessConfig

PRIORS = Path("data/priors/NJ_county_priors.csv")
CLEAN = Path("data/clean/NJ_county_cleaned.csv")
KERNEL_CONFIG = Path("data/derived/kernel_config.json")

TrialResult = namedtuple(
    "TrialResult",
    "n_obs mean_pct ks cal68 cal95 width68 width95 resp_near resp_far statewide_sd",
)


def load_realized() -> Dict[str, Tuple[int, int, int]]:
    return {
        "Atlantic":  (51201,  47603, 0),
        "Bergen":    (190461, 152682, 0),
        "Burlington":(116774, 75346,  0),
        "Camden":    (130752, 60129,  0),
        "Cape May":  (18270,  25588,  0),
        "Cumberland":(21348,  19272,  0),
        "Essex":     (188681, 55503,  0),
        "Gloucester":(67066,  61265,  0),
        "Hudson":    (127181, 41021,  0),
        "Hunterdon": (31663,  34683,  0),
        "Mercer":    (91713,  36156,  0),
        "Middlesex": (174038, 101830, 0),
        "Monmouth":  (131484, 154166, 0),
        "Morris":    (116488, 111422, 0),
        "Ocean":     (90323,  185957, 0),
        "Passaic":   (86053,  61966,  0),
        "Salem":     (9782,   13281,  0),
        "Somerset":  (83355,  56095,  0),
        "Sussex":    (26305,  38117,  0),
        "Union":     (124470, 59646,  0),
        "Warren":    (19199,  25980,  0),
    }


def realized_share(realized: Dict[str, Tuple[int, int, int]], county: str) -> float:
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


def run_one_setting(
    hyperparams: Dict[str, float],
    trials_per_k: int = 40,
    ks_bins: int = 200,
    seed: int = 42,
) -> List[TrialResult]:
    random.seed(seed)
    np.random.seed(seed)

    realized = load_realized()
    cfg = GaussianProcessConfig(
        length_scale=2.4,
        tau2=0.04,
        sigma_s2=0.005,
        sigma_eps2=1e-4,
        use_turnout_updates=True,
    )
    base_model = GaussianProcessModel(str(PRIORS), cfg=cfg)
    counties = base_model.counties
    name_to_idx = {c: i for i, c in enumerate(counties)}

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
            model = GaussianProcessModel(str(PRIORS), cfg=cfg)
            selected = random.sample(counties, n_obs) if n_obs > 0 else []
            for c in selected:
                d, r, _ = realized[c]
                model.observe_votes(c, d, r)
            model.update_posterior()

            statewide = model.sample_statewide(n_draws=1500)
            statewide_sds.append(statewide["sd"])

            ci68 = model.county_share_ci(z=1.0)
            ci95 = model.county_share_ci(z=1.96)
            mean = model.county_share_mean()
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
                prior_model = GaussianProcessModel(str(PRIORS), cfg=cfg)
                prior_mean = prior_model.county_share_mean().to_numpy()
                post_mean = mean.to_numpy()
                near_idx = nbrs[i]
                far_idx = anti_nbrs[i]
                resp_near_vals.append(
                    np.mean(np.abs(post_mean[near_idx] - prior_mean[near_idx]))
                )
                resp_far_vals.append(
                    np.mean(np.abs(post_mean[far_idx] - prior_mean[far_idx]))
                )

        m = len(all_pcts)
        if m == 0:
            continue

        mean_pct = float(np.mean(all_pcts))
        ks_stat = float(kstest(all_pcts, "uniform").statistic)
        denom = len(counties) - n_obs
        cal68 = cover68 / max(denom * trials_per_k, 1)
        cal95 = cover95 / max(denom * trials_per_k, 1)

        results.append(
            TrialResult(
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
            )
        )

    return results


def score_results(results: List[TrialResult]) -> float:
    score = 0.0
    for r in results:
        score += 2.0 * abs(r.mean_pct - 0.5) + 1.5 * r.ks
        score += 1.0 * abs(r.cal68 - 0.68) + 0.7 * abs(r.cal95 - 0.95)
        score += 2.0 * r.width68 + 1.0 * r.width95
        score += -3.0 * max(r.resp_near - r.resp_far, 0.0)
        score += 3.0 * abs(r.statewide_sd - 0.025)
    return score


def load_base_config() -> Dict:
    if not KERNEL_CONFIG.exists():
        raise FileNotFoundError("kernel_config.json not found; run build_NJ_composite_covariance first.")
    with KERNEL_CONFIG.open() as f:
        return json.load(f)


def write_config(base_cfg: Dict, hyperparams: Dict[str, float]) -> None:
    cfg = dict(base_cfg)
    hp = dict(cfg.get("hyperparams", {}))
    hp.update(hyperparams)
    cfg["hyperparams"] = hp
    with KERNEL_CONFIG.open("w") as f:
        json.dump(cfg, f, indent=2)


def sweep(
    grid_sigma_s2: Iterable[float],
    grid_tau_demo2: Iterable[float],
    grid_tau_shift2: Iterable[float],
    grid_sigma_eps2: Iterable[float],
    trials_per_k: int = 40,
) -> List[Tuple[float, Dict[str, float], List[TrialResult]]]:
    base_cfg = load_base_config()
    original_cfg = base_cfg.copy()
    results_bundle: List[Tuple[float, Dict[str, float], List[TrialResult]]] = []

    try:
        for s2 in grid_sigma_s2:
            for td in grid_tau_demo2:
                for ts in grid_tau_shift2:
                    for eps2 in grid_sigma_eps2:
                        hparams = {
                            "sigma_s2": s2,
                            "tau_demo2": td,
                            "tau_shift2": ts,
                            "sigma_eps2": eps2,
                        }
                        print(f"\n=== Testing {hparams} ===")
                        write_config(base_cfg, hparams)
                        res = run_one_setting(hparams, trials_per_k=trials_per_k)
                        sc = score_results(res)
                        results_bundle.append((sc, hparams, res))
                        for r in res:
                            print(
                                f"  n_obs={r.n_obs:2d} mean_pct={r.mean_pct:5.3f} "
                                f"KS={r.ks:5.3f} cov68={r.cal68:5.3f} cov95={r.cal95:5.3f} "
                                f"w68={r.width68:5.3f} w95={r.width95:5.3f} "
                                f"respΔ={r.resp_near - r.resp_far:5.3f} state_sd={r.statewide_sd:5.3f}"
                            )
    finally:
        with KERNEL_CONFIG.open("w") as f:
            json.dump(original_cfg, f, indent=2)

    results_bundle.sort(key=lambda x: x[0])
    return results_bundle


def main():
    grid_sigma_s2 = [0.001, 0.002, 0.004, 0.008, 0.016]
    grid_tau_demo2 = [0.02, 0.04, 0.08, 0.16]
    grid_tau_shift2 = [0.02, 0.04, 0.08, 0.16]
    grid_sigma_eps2 = [1e-4]

    runs = sweep(
        grid_sigma_s2=grid_sigma_s2,
        grid_tau_demo2=grid_tau_demo2,
        grid_tau_shift2=grid_tau_shift2,
        grid_sigma_eps2=grid_sigma_eps2,
        trials_per_k=30,
    )

    print("\n=== Top settings (lower score = better) ===")
    for rank, (sc, hparams, res) in enumerate(runs[:5], start=1):
        print(f"{rank:>2}. score={sc:7.3f} {hparams}")
        for r in res:
            print(
                f"    n_obs={r.n_obs:2d} mean_pct={r.mean_pct:5.3f} KS={r.ks:5.3f} "
                f"cov68={r.cal68:5.3f} cov95={r.cal95:5.3f} w68={r.width68:5.3f} "
                f"w95={r.width95:5.3f} respΔ={r.resp_near - r.resp_far:5.3f} state_sd={r.statewide_sd:5.3f}"
            )

    payload = []
    for sc, hparams, res in runs:
        payload.append(
            {
                "score": sc,
                "hyperparams": hparams,
                "results": [r._asdict() for r in res],
            }
        )
    out_path = Path("data/derived/hparam_sweep_composite.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ok] wrote {out_path} with {len(runs)} grid points")


if __name__ == "__main__":
    main()
