import argparse
import json
import sys

import requests

API_URL = "http://localhost:5001/api/ingest-real"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest a single county result into the local API.")
    ap.add_argument("--county", required=True, help="County name as used in priors (e.g., 'Bergen').")
    ap.add_argument("--dem", type=int, required=True, help="Democratic votes.")
    ap.add_argument("--rep", type=int, required=True, help="Republican votes.")
    ap.add_argument("--other", type=int, default=0, help="Other/third-party votes.")
    ap.add_argument("--reporting", type=float, required=True, help="Percent reporting (0-100, one decimal ok).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "county": args.county,
        "dem_votes": args.dem,
        "rep_votes": args.rep,
        "other_votes": args.other,
        "reporting_pct": args.reporting,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"[error] failed to reach backend: {exc}", file=sys.stderr)
        return 1

    if resp.status_code != 200:
        print(f"[error] backend responded {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1

    data = resp.json()
    real = data.get("real_results", [])
    print("[ok] ingested")
    print(json.dumps({"real_results": real}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())