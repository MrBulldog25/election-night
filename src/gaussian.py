import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from scipy.stats import norm

import numpy as np
import pandas as pd

from kernel_builders import _weighted_demo_kernel

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))

def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def zscore_cols(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, Dict[str, Tuple[float,float]]]:
    X = df[cols].astype(float).values
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    stats = {c: (float(mu[i]), float(sd[i])) for i, c in enumerate(cols)}
    return Z, stats

def rbf_kernel(Z: np.ndarray, length_scale: float) -> np.ndarray:
    d2 = np.sum(Z**2, axis=1, keepdims=True) + np.sum(Z**2, axis=1) - 2 * Z @ Z.T
    d2 = np.maximum(d2, 0.0)
    K = np.exp(-0.5 * d2 / (length_scale ** 2))
    return K

@dataclass
class GaussianProcessConfig:
    length_scale: float = 2.4
    tau2: float = 0.04
    sigma_eps2: float = 1e-4
    sigma_s2: float = 0.005
    use_turnout_updates: bool = True

class GaussianProcessModel:
    def __init__(self, priors_csv: str, demographic_cols: Optional[List[str]] = None, cfg: Optional[GaussianProcessConfig] = None):
        self.cfg = cfg or GaussianProcessConfig()
        self.df = pd.read_csv(priors_csv)
        req = ["county","p0","uncert_p_share","baseline_total_votes","uncert_turnout"]
        missing = [c for c in req if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in priors: {missing}")

        self.counties: List[str] = list(self.df["county"].astype(str).values)
        self.n = len(self.counties)

        self.mu0 = logit(self.df["p0"].values)

        if demographic_cols is None:
            block = {
                "baseline_dem_votes","baseline_rep_votes","baseline_other_votes",
                "baseline_total_votes","p0","logit_p0","uncert_p_share","uncert_turnout",
                "log_turnout_total","twoparty","margin"
            }
            demo_candidates = [
                c for c in self.df.columns
                if c not in block and self.df[c].dtype != object
            ]
            demo_candidates = [c for c in demo_candidates if np.isfinite(self.df[c].astype(float)).all()]
            keep = [c for c in demo_candidates if any(k in c.lower() for k in
                    ["%","density","income","households","vap","population","bachelors","graduate","senior","white","black","asian","latino","age"])]
            demographic_cols = keep or demo_candidates[:6]
        self.demo_cols = demographic_cols

        self.Sigma = self._build_covariance()

        self.obs_idx: List[int] = []
        self.y_obs: List[float] = []
        self.v_obs: List[float] = []

        self.N0 = self.df["baseline_total_votes"].astype(float).values
        self.sN = self.df["uncert_turnout"].astype(float).values
        self.eta_mean = self.mu0.copy()
        self.eta_cov = self.Sigma.copy()
        self.N_mean = self.N0.copy()
        self.N_var = (np.exp(self.sN**2) - 1.0) * np.exp(2*np.log(self.N0) + self.sN**2)

        self.winner_constraints: List[Tuple[int, str]] = []

    def _load_kernel_config(self) -> Optional[Dict]:
        cfg_path = Path("data/derived/kernel_config.json")
        if not cfg_path.exists():
            return None
        try:
            with cfg_path.open() as f:
                return json.load(f)
        except Exception:
            return None

    def _build_legacy_covariance(self) -> np.ndarray:
        Z, self.zstats = zscore_cols(self.df, self.demo_cols)
        K_demo = rbf_kernel(Z, self.cfg.length_scale)
        J = np.ones((self.n, self.n), dtype=float)
        return (self.cfg.sigma_s2 * J) + (self.cfg.tau2 * K_demo) + (self.cfg.sigma_eps2 * np.eye(self.n))

    def _build_composite_covariance(self, cfg: Dict) -> np.ndarray:
        state = str(cfg.get("state", "")).upper()
        if state != "NJ":
            return self._build_legacy_covariance()

        shift_path = Path(cfg["shift_kernel_path"])
        weights_path = Path(cfg["demo_weights_path"])
        ell_demo = float(cfg.get("ell_demo", 1.0))
        hp = cfg.get("hyperparams", {})

        kernel_data = np.load(shift_path, allow_pickle=True)
        K_shift = kernel_data["K_shift"]
        states = kernel_data["states"]
        counties = kernel_data["counties"]
        key_to_idx = {f"{s}|{c}": i for i, (s, c) in enumerate(zip(states, counties))}

        priors_counties = self.df["county"].astype(str).tolist()
        idx = []
        for c in priors_counties:
            key = f"{state}|{c}"
            if key not in key_to_idx:
                continue
            idx.append(key_to_idx[key])
        if not idx:
            return self._build_legacy_covariance()
        K_shift_NJ = K_shift[np.ix_(idx, idx)]

        with weights_path.open() as f:
            weights = json.load(f)
        demo_cols = weights["demo_cols"]
        alpha = np.array(weights["alpha"], dtype=float)
        ell_demo = float(weights.get("ell_demo", ell_demo))
        self.demo_cols = demo_cols

        X_demo = self.df[demo_cols].astype(float)
        X_demo = (X_demo - X_demo.mean()) / X_demo.std(ddof=0).replace(0, 1.0)
        X_demo = X_demo.fillna(0.0)
        self.zstats = {c: (float(X_demo[c].mean()), float(X_demo[c].std(ddof=0))) for c in demo_cols}
        K_demo_NJ = _weighted_demo_kernel(X_demo.values, alpha, ell_demo=ell_demo)

        hp_full = {
            "sigma_s2": self.cfg.sigma_s2,
            "tau_demo2": self.cfg.tau2,
            "tau_shift2": self.cfg.tau2,
            "sigma_eps2": self.cfg.sigma_eps2,
        }
        hp_full.update(hp)

        J = np.ones((self.n, self.n), dtype=float)
        I = np.eye(self.n, dtype=float)
        return (
            hp_full["sigma_s2"] * J
            + hp_full["tau_demo2"] * K_demo_NJ
            + hp_full["tau_shift2"] * K_shift_NJ
            + hp_full["sigma_eps2"] * I
        )

    def _build_covariance(self) -> np.ndarray:
        cfg = self._load_kernel_config()
        if cfg:
            try:
                return self._build_composite_covariance(cfg)
            except Exception as exc:
                print(f"[warn] Falling back to legacy kernel: {exc}")
        return self._build_legacy_covariance()

    def _normalize_winner_label(self, winner: str) -> str:
        if winner is None:
            raise ValueError("winner label must not be None")
        w = str(winner).strip().lower()
        if w in ("d", "dem", "democrat", "democrats", "blue"):
            return "D"
        if w in ("r", "rep", "republican", "republicans", "red"):
            return "R"
        raise ValueError(f"Unrecognized winner label: {winner!r}")

    def _apply_single_winner_constraint(self, idx: int, winner: str) -> None:
        winner_norm = self._normalize_winner_label(winner)

        mu_i = float(self.eta_mean[idx])
        var_i = float(self.eta_cov[idx, idx])
        if var_i <= 0:
            return

        sd_i = math.sqrt(var_i)
        a = 0.0
        alpha = (a - mu_i) / sd_i

        eps = 1e-12

        if winner_norm == "D":
            cdf_alpha = float(norm.cdf(alpha))
            Z = max(1.0 - cdf_alpha, eps)
            lambda_ = float(norm.pdf(alpha)) / Z
            new_mu_i = mu_i + sd_i * lambda_
            new_var_i = var_i * (1.0 + alpha * lambda_ - lambda_**2)
        else:
            cdf_alpha = float(norm.cdf(alpha))
            Z = max(cdf_alpha, eps)
            lambda_ = float(norm.pdf(alpha)) / Z
            new_mu_i = mu_i - sd_i * lambda_
            new_var_i = var_i * (1.0 - alpha * lambda_ - lambda_**2)

        new_var_i = max(new_var_i, 1e-9)

        S = self.eta_cov
        s_i = S[:, idx].copy()
        s_ii = var_i

        delta_mu_i = new_mu_i - mu_i
        if abs(delta_mu_i) > 0:
            self.eta_mean = self.eta_mean + (s_i / s_ii) * delta_mu_i

        factor = (new_var_i - var_i) / (s_ii ** 2)
        if abs(factor) > 0:
            self.eta_cov = S + factor * np.outer(s_i, s_i)

    def apply_winner_constraints(self) -> None:
        if not self.winner_constraints:
            return

        for idx, winner in self.winner_constraints:
            self._apply_single_winner_constraint(idx, winner)

        self.winner_constraints = []

    def observe_votes(
        self,
        county: str,
        dem_votes: int,
        rep_votes: int,
        other_votes: int = 0,
        reported_fraction: Optional[float] = None,
    ):
        i = self.counties.index(county)
        two_party_total = dem_votes + rep_votes
        if two_party_total <= 0:
            return

        p_hat = dem_votes / two_party_total
        v = 1.0 / max(two_party_total * p_hat * (1 - p_hat), 1e-6)

        if self.cfg.use_turnout_updates:
            frac = 1.0 if reported_fraction is None else float(reported_fraction)
            frac = min(max(frac, 1e-3), 1.0)
            observed_total = dem_votes + rep_votes + other_votes
            est_total = observed_total / frac
            self.N_mean[i] = float(est_total)
            if frac >= 0.999:
                self.N_var[i] = 1e-6
            else:
                self.N_var[i] = max(self.N_var[i] / frac, 1e-6)

        self.obs_idx.append(i)
        self.y_obs.append(float(np.log(p_hat / (1 - p_hat))))
        self.v_obs.append(float(v))

    def observe_share(self, county: str, dem_share: float):
        i = self.counties.index(county)
        p = float(np.clip(dem_share, 1e-4, 1 - 1e-4))
        v = 1e-6
        self.obs_idx.append(i)
        self.y_obs.append(float(np.log(p / (1 - p))))
        self.v_obs.append(float(v))
    
    def observe_winner(self, county: str, winner: str):
        if county not in self.counties:
            raise ValueError(f"County '{county}' not recognized.")
        idx = self.counties.index(county)
        winner_norm = self._normalize_winner_label(winner)
        self.winner_constraints.append((idx, winner_norm))

    def update_posterior(self):
        if not self.obs_idx:
            self.eta_mean = self.mu0.copy()
            self.eta_cov = self.Sigma.copy()
            return

        idx = np.array(self.obs_idx, dtype=int)
        y = np.array(self.y_obs, dtype=float)
        V = np.diag(np.array(self.v_obs, dtype=float))

        S_oo = self.Sigma[np.ix_(idx, idx)]
        S_ou = self.Sigma[np.ix_(idx, np.arange(self.n))]
        S_uo = S_ou.T
        mu_o = self.mu0[idx]
        mu_u = self.mu0

        A = S_oo + V
        L = np.linalg.cholesky(A + 1e-10*np.eye(A.shape[0]))
        r = y - mu_o
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))

        self.eta_mean = mu_u + S_uo @ alpha
        tmp = np.linalg.solve(L, S_ou)
        self.eta_cov = self.Sigma - tmp.T @ tmp

    def county_share_mean(self) -> pd.Series:
        return pd.Series(inv_logit(self.eta_mean), index=self.counties, name="p_mean")

    def county_share_ci(self, z: float = 1.0) -> pd.DataFrame:
        var = np.clip(np.diag(self.eta_cov), 1e-12, None)
        lo = inv_logit(self.eta_mean - z*np.sqrt(var))
        hi = inv_logit(self.eta_mean + z*np.sqrt(var))
        return pd.DataFrame({"lo": lo, "hi": hi}, index=self.counties)

    def _draw_vote_components(self, n_draws: int, rng: Optional[np.random.Generator]) -> Tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(42)
        L = np.linalg.cholesky(self.eta_cov + 1e-12*np.eye(self.n))
        Z = rng.standard_normal((self.n, n_draws))
        eta_draws = self.eta_mean[:, None] + L @ Z
        p_draws = inv_logit(eta_draws)

        muN = np.log(np.maximum(self.N_mean, 1.0))
        sN = np.sqrt(np.log(self.N_var / (self.N_mean**2) + 1.0))
        N_draws = rng.lognormal(mean=muN[:, None], sigma=sN[:, None], size=(self.n, n_draws))
        return p_draws, N_draws

    def sample_statewide(self, n_draws: int = 5000, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        rng = rng or np.random.default_rng(42)
        p_draws, N_draws = self._draw_vote_components(n_draws=n_draws, rng=rng)
        state_share = (p_draws * N_draws).sum(axis=0) / N_draws.sum(axis=0)
        return {
            "mean": float(state_share.mean()),
            "sd": float(state_share.std()),
            "p_dem_win": float((state_share > 0.5).mean()),
            "q05": float(np.quantile(state_share, 0.05)),
            "q50": float(np.quantile(state_share, 0.50)),
            "q95": float(np.quantile(state_share, 0.95)),
        }

    def vote_statistics(self, z: float) -> Dict[str, Dict[str, Dict[str, float]]]:
        p_draws, N_draws = self._draw_vote_components(
            n_draws=5000, rng=np.random.default_rng(42)
        )
        dem_draws = p_draws * N_draws
        rep_draws = (1.0 - p_draws) * N_draws
        tot_draws = N_draws

        def summarize(values: np.ndarray, lower: float = 0.0, upper: Optional[float] = None) -> Dict[str, float]:
            mean = float(values.mean())
            sd = float(values.std(ddof=0))
            delta = z * sd
            lo = mean - delta
            hi = mean + delta
            if lower is not None:
                lo = max(lo, lower)
            if upper is not None:
                hi = min(hi, upper)
            return {"mean": mean, "lo": lo, "hi": hi}

        statewide_dem = dem_draws.sum(axis=0)
        statewide_rep = rep_draws.sum(axis=0)
        statewide_tot = tot_draws.sum(axis=0)
        statewide_share = statewide_dem / np.maximum(statewide_tot, 1e-9)
        statewide_margin = (2.0 * statewide_share - 1.0) * 100.0
        p_dem_win = float((statewide_share > 0.5).mean())
        p_rep_win = 1.0 - p_dem_win

        margin_bins_config = [
            ("R win", -np.inf, 0.0),
            ("D +0–2", 0.0, 2.0),
            ("D +2–4", 2.0, 4.0),
            ("D +4–6", 4.0, 6.0),
            ("D +6–8", 6.0, 8.0),
            ("D +8–10", 8.0, 10.0),
            ("D +10–12", 10.0, 12.0),
            ("D +12–14", 12.0, 14.0),
            ("D +14+", 14.0, np.inf),
        ]

        margin_bins = []
        for label, lo, hi in margin_bins_config:
            if np.isinf(lo):
                mask = statewide_margin < hi
            elif np.isinf(hi):
                mask = statewide_margin >= lo
            else:
                mask = (statewide_margin >= lo) & (statewide_margin < hi)
            margin_bins.append(
                {
                    "label": label,
                    "lower": None if np.isneginf(lo) else float(lo),
                    "upper": None if np.isposinf(hi) else float(hi),
                    "prob": float(mask.mean()),
                }
            )

        county_stats = {}
        for idx, county in enumerate(self.counties):
            p_win = float((p_draws[idx, :] > 0.5).mean())
            county_stats[county] = {
                "dem_votes": summarize(dem_draws[idx, :]),
                "rep_votes": summarize(rep_draws[idx, :]),
                "total_votes": summarize(tot_draws[idx, :]),
                "dem_share": summarize(p_draws[idx, :], lower=0.0, upper=1.0),
                "rep_share": summarize(1.0 - p_draws[idx, :], lower=0.0, upper=1.0),
                "p_dem_win": p_win,
                "p_rep_win": 1.0 - p_win,
            }

        return {
            "statewide": {
                "dem_votes": summarize(statewide_dem),
                "rep_votes": summarize(statewide_rep),
                "total_votes": summarize(statewide_tot),
                "dem_share": summarize(statewide_share, lower=0.0, upper=1.0),
                "rep_share": summarize(1.0 - statewide_share, lower=0.0, upper=1.0),
                "p_dem_win": p_dem_win,
                "p_rep_win": p_rep_win,
                "margin_bins": margin_bins,
            },
            "counties": county_stats,
        }

    def county_result_surprise(self, county: str, dem_share: float) -> Dict[str, float]:
        if county not in self.counties:
            raise ValueError(f"County '{county}' not recognized.")

        i = self.counties.index(county)
        mu = self.eta_mean[i]
        var = float(self.eta_cov[i, i])
        sd = math.sqrt(max(var, 1e-12))

        p_obs = np.clip(float(dem_share), 1e-9, 1 - 1e-9)
        eta_obs = math.log(p_obs / (1 - p_obs))

        z = (eta_obs - mu) / sd
        p_val = 2 * (1 - norm.cdf(abs(z)))
        percentile = norm.cdf(z)

        return {"z": float(z), "p_value": float(p_val), "percentile": float(percentile)}

    def info(self) -> None:
        print(f"[GaussianProcessModel] n_counties={self.n}")
        print(f"  demo_cols: {self.demo_cols}")
        print(f"  kernel: RBF length_scale={self.cfg.length_scale}, tau2={self.cfg.tau2}")
        print(f"  swing sigma^2={self.cfg.sigma_s2}, nugget sigma^2={self.cfg.sigma_eps2}")
