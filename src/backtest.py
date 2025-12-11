import math
from gaussian import GaussianProcessModel, GaussianProcessConfig
from constant import COUNTIES
from collections import defaultdict

PRIORS = "data/priors/NJ_county_priors.csv"

def get_realized():
    path = "data/clean/NJ_county_cleaned.csv"
    clean_df = pd.read_csv(path)
    path = "data/priors/NJ_county_priors.csv"
    priors_df = pd.read_csv(path)
    returned_dict = {}
    total_votes_24 = clean_df["2024_Pre_D"].sum() + clean_df["2024_Pre_R"].sum() + clean_df["2024_Pre_O"].sum()
    total_votes_baseline = priors_df["baseline_total_votes"].sum()
    fraction = total_votes_baseline / total_votes_24
    for _, row in clean_df.iterrows():
        county = row["County"]
        dem_votes = math.floor(row["2024_Pre_D"] * fraction)
        rep_votes = math.floor(row["2024_Pre_R"] * fraction)
        other_votes = math.floor(row["2024_Pre_O"] * fraction)
        returned_dict[county] = (dem_votes, rep_votes, other_votes)
    return returned_dict


def main():
    cfg = GaussianProcessConfig(
        length_scale=1.0,
        tau2=0.50,
        sigma_s2=0.04,
        sigma_eps2=1e-4,
        use_turnout_updates=True
    )
    model = GaussianProcessModel(PRIORS, cfg=cfg)

    realized = get_realized()

    prior_state = model.sample_statewide(n_draws=3000)
    prior_p_mean = model.county_share_mean()
    prior_ci68 = model.county_share_ci(z=1.0)

    log_backtest = {}
    log_stats = defaultdict(list)

    for county, (dem_votes, rep_votes, other_votes) in realized.items():
        model.observe_votes(county, dem_votes=dem_votes, rep_votes=rep_votes, reported_fraction=1.0)
        model.update_posterior()
        
        log_key_1 = (county,)
        log_for_county = {}
        for county2, (dem_votes2, rep_votes2, other_votes2) in realized.items():
            if county2 == county:
                continue
            share = dem_votes2 / (dem_votes2 + rep_votes2)
            surprise = model.county_result_surprise(county2, share)
            log_for_county[county2] = surprise["percentile"]
            log_stats[1].append(surprise["percentile"])
        log_backtest[log_key_1] = log_for_county
        model = GaussianProcessModel(PRIORS, cfg=cfg)
    
    counties_list = list(realized.keys())
    for i in range(len(counties_list)):
        for j in range(i+1, len(counties_list)):
            for k in range(j+1, len(counties_list)):
                county_i = counties_list[i]
                county_j = counties_list[j]
                county_k = counties_list[k]
                
                model.observe_votes(
                    county_i, 
                    dem_votes=realized[county_i][0],
                    rep_votes=realized[county_i][1], 
                    reported_fraction=1.0,
                )
                model.observe_votes(
                    county_j, 
                    dem_votes=realized[county_j][0],
                    rep_votes=realized[county_j][1], 
                    reported_fraction=1.0,
                )
                model.observe_votes(
                    county_k, 
                    dem_votes=realized[county_k][0],
                    rep_votes=realized[county_k][1], 
                    reported_fraction=1.0,
                )
                model.update_posterior()
                
                log_key_3 = (county_i, county_j, county_k)
                log_for_county_3 = {}
                for county2, (dem_votes2, rep_votes2, other_votes2) in realized.items():
                    if county2 in log_key_3:
                        continue
                    share = dem_votes2 / (dem_votes2 + rep_votes2)
                    surprise = model.county_result_surprise(county2, share)
                    log_for_county_3[county2] = surprise["percentile"]
                    log_stats[3].append(surprise["percentile"])
                log_backtest[log_key_3] = log_for_county_3
                model = GaussianProcessModel(PRIORS, cfg=cfg)
    
    import random
    random.seed(42)
    for _ in range(200):
        selected_counties = random.sample(counties_list, 10)
        for county in selected_counties:
            model.observe_votes(
                county, 
                dem_votes=realized[county][0],
                rep_votes=realized[county][1], 
                reported_fraction=1.0
            )
        model.update_posterior()
        
        log_key_10 = tuple(selected_counties)
        log_for_county_10 = {}
        for county2, (dem_votes2, rep_votes2, other_votes2) in realized.items():
            if county2 in selected_counties:
                continue
            share = dem_votes2 / (dem_votes2 + rep_votes2)
            surprise = model.county_result_surprise(county2, share)
            log_for_county_10[county2] = surprise["percentile"]
            log_stats[10].append(surprise["percentile"])
        log_backtest[log_key_10] = log_for_county_10
        
        model = GaussianProcessModel(PRIORS, cfg=cfg)
    
    for n_obs in sorted(log_stats.keys()):
        stats = log_stats[n_obs]
        mean_percentile = sum(stats) / len(stats)
        print(f"After observing {n_obs} counties: Mean percentile of unobserved counties' actual results = {mean_percentile:.2f}%")
        for pct in [25, 50, 75]:
            pct_value = sorted(stats)[int(len(stats) * pct / 100)]
            print(f"  {pct}th percentile: {pct_value:.2f}%")
    
    import json
    with open("backtest_log.json", "w") as f:
        json.dump(log_backtest, f, indent=2)


if __name__ == "__main__":
    import pandas as pd
    main()
