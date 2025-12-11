from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import requests

from gaussian import GaussianProcessConfig, GaussianProcessModel

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

PRIORS_CSV = Path("data/priors/NJ_county_priors.csv")
COUNTIES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)
Z_SCORE = 1.96

STATE = {
    "base_model": None,
    "active_model": None,
    "geojson": None,
    "priors_df": None,
    "applied_observations": [],
    "real_results": [],
}

def load_priors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "county" not in df.columns:
        for cand in ("County.1", "County", "County_Name"):
            if cand in df.columns:
                df = df.rename(columns={cand: "county"})
                break
    if "county" not in df.columns:
        raise ValueError("Couldn't find a county name column; expected 'county'.")
    
    df["county_clean"] = df["county"].str.strip().str.title()
    
    NJ_COUNTY_FIPS = {
        "Atlantic": "34001", "Bergen": "34003", "Burlington": "34005", "Camden": "34007",
        "Cape May": "34009", "Cumberland": "34011", "Essex": "34013", "Gloucester": "34015",
        "Hudson": "34017", "Hunterdon": "34019", "Mercer": "34021", "Middlesex": "34023",
        "Monmouth": "34025", "Morris": "34027", "Ocean": "34029", "Passaic": "34031",
        "Salem": "34033", "Somerset": "34035", "Sussex": "34037", "Union": "34039",
        "Warren": "34041",
    }
    
    def geo_id(county_name: str) -> str:
        county_name = county_name.strip().title()
        if county_name not in NJ_COUNTY_FIPS:
            return county_name
        return "0500000US" + NJ_COUNTY_FIPS[county_name]

    df["geo_id"] = df["county_clean"].apply(geo_id)
    return df

def get_or_create_base_model() -> GaussianProcessModel:
    if STATE["base_model"] is None:
        cfg = GaussianProcessConfig(
            length_scale=2.4,
            tau2=0.04,
            sigma_s2=0.005,
            sigma_eps2=1e-4,
            use_turnout_updates=True,
        )
        STATE["base_model"] = GaussianProcessModel(str(PRIORS_CSV), cfg=cfg)
        STATE["priors_df"] = load_priors(PRIORS_CSV)
    return STATE["base_model"]

def apply_observations(model: GaussianProcessModel, observations: List[Dict]) -> None:
    observations.sort(key=lambda obs: obs.get("kind") == "winner")
    for obs in observations:
        kind = obs.get("kind")
        county = str(obs["county"])
        reporting_fraction = obs.get("reporting_fraction")
        other_votes = int(obs.get("other_votes", 0))
        if kind == "votes":
            model.observe_votes(
                county=county,
                dem_votes=int(obs["dem_votes"]),
                rep_votes=int(obs["rep_votes"]),
                other_votes=other_votes,
                reported_fraction=reporting_fraction,
            )
        elif kind == "share":
            model.observe_share(
                county=county,
                dem_share=float(obs["share"]),
            )
        elif kind == "winner":
            model.observe_winner(
                county=county,
                winner=str(obs["winner"]),
            )
    model.update_posterior()

def build_model_with_observations(
    real_results: List[Dict], applied_observations: List[Dict]
) -> GaussianProcessModel:
    base = get_or_create_base_model()
    model = GaussianProcessModel(str(PRIORS_CSV), cfg=base.cfg)

    apply_observations(model, real_results)
    apply_observations(model, applied_observations)

    model.apply_winner_constraints()

    return model

def rebuild_active_model() -> GaussianProcessModel:
    model = build_model_with_observations(
        STATE["real_results"],
        STATE["applied_observations"],
    )
    STATE["active_model"] = model
    return model

def get_geojson():
    if STATE["geojson"] is None:
        resp = requests.get(COUNTIES_GEOJSON_URL, timeout=20)
        resp.raise_for_status()
        STATE["geojson"] = resp.json()
    return STATE["geojson"]

@app.route("/api/initial-data", methods=["GET"])
def initial_data():
    try:
        base_model = get_or_create_base_model()
        geojson = get_geojson()
        
        if STATE["active_model"] is None:
            STATE["active_model"] = build_model_with_observations(
                STATE["real_results"],
                STATE["applied_observations"],
            )
            
        stats = STATE["active_model"].vote_statistics(Z_SCORE)
        
        priors_df = STATE["priors_df"]
        county_metadata = priors_df[["county", "county_clean", "geo_id"]].to_dict(orient="records")
        
        return jsonify({
            "geojson": geojson,
            "stats": stats,
            "county_metadata": county_metadata,
            "applied_observations": STATE["applied_observations"],
            "real_results": STATE["real_results"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/update", methods=["POST"])
def update_model():
    try:
        data = request.json
        observations = data.get("observations", [])
        
        STATE["applied_observations"] = observations
        new_model = rebuild_active_model()
        
        stats = new_model.vote_statistics(Z_SCORE)
        return jsonify({
            "stats": stats,
            "applied_observations": observations,
            "real_results": STATE["real_results"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ingest-real", methods=["POST"])
def ingest_real():
    try:
        data = request.json or {}
        required = ["county", "dem_votes", "rep_votes", "reporting_pct"]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        county = str(data["county"])
        dem_votes = int(data["dem_votes"])
        rep_votes = int(data["rep_votes"])
        other_votes = int(data.get("other_votes", 0))
        reporting_pct = round(float(data["reporting_pct"]), 1)
        if reporting_pct < 0 or reporting_pct > 100:
            return jsonify({"error": "reporting_pct must be between 0 and 100"}), 400

        entry = {
            "kind": "votes",
            "county": county,
            "dem_votes": dem_votes,
            "rep_votes": rep_votes,
            "other_votes": other_votes,
            "reporting_pct": reporting_pct,
            "reporting_fraction": reporting_pct / 100.0,
            "source": "real",
        }

        STATE["real_results"] = [r for r in STATE["real_results"] if r.get("county") != county]
        STATE["real_results"].append(entry)

        new_model = rebuild_active_model()
        stats = new_model.vote_statistics(Z_SCORE)

        return jsonify({
            "stats": stats,
            "applied_observations": STATE["applied_observations"],
            "real_results": STATE["real_results"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def reset_model():
    try:
        base_model = get_or_create_base_model()
        STATE["active_model"] = base_model
        STATE["applied_observations"] = []
        STATE["real_results"] = []
        
        stats = base_model.vote_statistics(Z_SCORE)
        return jsonify({
            "stats": stats,
            "applied_observations": [],
            "real_results": [],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")
