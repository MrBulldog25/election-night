import math
import numpy as np
import pandas as pd

INFILE = "data/priors/NJ_county_priors.csv" 

def logit(p):
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))

def inv_logit(x):
    return 1 / (1 + math.exp(-x))

def pct(x):
    return f"{100*x:5.1f}%"

def fmt_int(x):
    return f"{int(round(x)):>8,d}"

def ci_logitnormal(p0, sigma, z=1.0):
    mu = logit(p0)
    return inv_logit(mu - z*sigma), inv_logit(mu + z*sigma)

def ci_lognormal(n0, sigma, z=1.0):
    mu = math.log(max(float(n0), 1.0))
    return math.exp(mu - z*sigma), math.exp(mu + z*sigma)

def main():
    df = pd.read_csv(INFILE)

    req = ["county","baseline_dem_votes","baseline_rep_votes","baseline_total_votes",
           "p0","uncert_p_share","uncert_turnout"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df.copy()
    df["twoparty"] = df["baseline_dem_votes"] + df["baseline_rep_votes"]
    df["margin"] = 2*df["p0"] - 1

    print("NJ County Priors — quick inspect\n")
    print("County".ljust(16),
          "Dem(2p)".rjust(9),
          "Rep(2p)".rjust(9),
          "Total".rjust(9),
          "P(D)".rjust(7),
          "Margin".rjust(8),
          "Share 95%".rjust(13),
          "Turnout 95%".rjust(16),
          sep=" ")

    print("-"*16, "-"*9, "-"*9, "-"*9, "-"*7, "-"*8, "-"*13, "-"*16)

    for _, r in df.sort_values("margin").iterrows():
        p0 = float(r["p0"])
        s_sig = float(r["uncert_p_share"])
        t_sig = float(r["uncert_turnout"])
        dem = float(r["baseline_dem_votes"])
        rep = float(r["baseline_rep_votes"])
        tot = float(r["baseline_total_votes"])

        loS, hiS = ci_logitnormal(p0, s_sig, z=2.0)
        loT, hiT = ci_lognormal(tot, t_sig, z=2.0)

        print(str(r["county"]).ljust(16),
              fmt_int(dem), fmt_int(rep), fmt_int(tot),
              pct(p0).rjust(7),
              f"{(2*p0-1):+6.1%}",
              f"[{pct(loS)}, {pct(hiS)}]".rjust(13),
              f"[{fmt_int(loT)}, {fmt_int(hiT)}]".rjust(16),
              sep=" ")

    tw = (df["baseline_dem_votes"] + df["baseline_rep_votes"]).sum()
    state_p = df["baseline_dem_votes"].sum() / tw if tw > 0 else float("nan")
    state_margin = 2*state_p - 1
    state_tot = int(df["baseline_total_votes"].sum())

    print("\nSTATEWIDE")
    print(f"  Two-party Dem share: {state_p:.2%} (margin {state_margin:+.1%})")
    print(f"  Total baseline turnout: {state_tot:,d}")

if __name__ == "__main__":
    main()
