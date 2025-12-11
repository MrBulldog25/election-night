#!/usr/bin/env python3
import sys
import re
import argparse
import pandas as pd
from pathlib import Path

def to_numeric_clean(s):
    if pd.isna(s):
        return pd.NA
    if isinstance(s, (int, float)):
        return s
    s = str(s).strip()
    s = s.replace(',', '')
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except:
            return pd.NA
    try:
        return float(s)
    except:
        return pd.NA

def main():
    ap = argparse.ArgumentParser(description="Convert XLSX to cleaned CSV.")
    ap.add_argument("state", help="State in question (abbreviated as in filenames)")
    ap.add_argument("denomination", help="County, Parish, City, etc.")
    args = ap.parse_args()

    demographics = {
        "Pop20": "Population 2020",
        "DVAP": "VAP",
        "DWhiteVAP": "Citizen Voting Age Population White",
        "DBlackVAP": "Citizen Voting Age Population Black",
        "DAsianVAP": "Citizen Voting Age Population Asian",
        "DNativeVAP": "Citizen Voting Age Population Native",
        "DLatinoVAP": "Citizen Voting Age Population Latino",
        "DHighschool": "Count High School Degree",
        "DBachelors": "Count Bachelors Degree",
        "DGraduate": "Count Graduate Degree",
        "D65Plus": "Count Age 65+",
        "DHouseholds": "Number of Households",
        "D0_25k": "Count Income <25k",
        "D25k_50k": "Count Income 25-50k",
        "D50k_100k": "Count Income 50-100k",
        "D100k_200k": "Count Income 100-200k",
        "D200kPlus": "Count Income >200k",
        "DArea": "Area (Square KM)",
    }

    xlsx_path = Path(f"data/raw/{args.state}_{args.denomination.lower()}_raw.xlsx")
    if not xlsx_path.exists():
        print(f"[File not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    excel_file = pd.ExcelFile(xlsx_path)
    sheet_name = "Data" if "Data" in excel_file.sheet_names else excel_file.sheet_names[0]
    df = excel_file.parse(sheet_name)

    def resolve_identifier_column():
        normalized = {col.strip().lower(): col for col in df.columns}
        candidates = [
            args.denomination,
            args.denomination.strip().lower(),
            args.denomination.strip().upper(),
            "name",
            "county",
            "parish",
            "city",
            "district",
            "countydistrict",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            key = candidate.strip().lower()
            if key in normalized:
                return normalized[key]
        return None

    id_column_name = resolve_identifier_column()
    if id_column_name is None:
        print("Could not find an id column.", file=sys.stderr)
        sys.exit(1)
    id_col = df[id_column_name]

    gbuild_r = df["GBuildR"] if "GBuildR" in df.columns else None
    gbuild_d = df["GBuildD"] if "GBuildD" in df.columns else None
    if gbuild_r is None or gbuild_d is None:
        print("Could not find both GBuild R and GBuild D columns.", file=sys.stderr)
        sys.exit(1)
    
    historical_r = {}
    historical_d = {}
    historical_o = {}
    for column in df.columns:
        name = column.strip()
        if name[0] == 'G':
            m = re.match(r'G(.*)(R|O|D)$', name)            
            if m:
                key = m.group(1)
                party = m.group(2)
                if key == 'Build':
                    continue
                if party == 'R':
                    historical_r[key] = column
                elif party == 'D':
                    historical_d[key] = column
                elif party == 'O':
                    historical_o[key] = column
    
    clean_df = pd.DataFrame()
    clean_df[args.denomination] = id_col
    clean_df['Baseline_R'] = gbuild_r.apply(to_numeric_clean)
    clean_df['Baseline_D'] = gbuild_d.apply(to_numeric_clean)

    third_party_shares = None
    election_counts = 0
    for key, col in historical_r.items():
        if key in historical_d:
            year = key[:2]
            race = key[2:5]
            r_col = historical_r[key]
            d_col = historical_d[key]
            new_col_name_d = f"20{year}_{race}_D"
            clean_df[new_col_name_d] = df[d_col].apply(to_numeric_clean)
            new_col_name_r = f"20{year}_{race}_R"
            clean_df[new_col_name_r] = df[r_col].apply(to_numeric_clean)
            if key in historical_o:
                o_col = historical_o[key]
                new_col_name_o = f"20{year}_{race}_O"
                clean_df[new_col_name_o] = df[o_col].apply(to_numeric_clean)
                r_series = clean_df[new_col_name_r]
                d_series = clean_df[new_col_name_d]
                o_series = clean_df[new_col_name_o]
                total = r_series + d_series + o_series
                third_party_share = o_series / total
                if third_party_shares is None:
                    third_party_shares = third_party_share
                else:
                    third_party_shares = (third_party_shares*(election_counts) + third_party_share.fillna(0))/(election_counts+1)
                election_counts += 1
    if third_party_shares is not None:
        import numpy as np
        np.random.seed(42)
        baseline_r = clean_df['Baseline_R']
        baseline_d = clean_df['Baseline_D']
        baseline_other = (third_party_shares * (baseline_r + baseline_d)).fillna(0).apply(np.floor)
        adjustments = baseline_other.apply(lambda x: np.random.randint(0, int(x)+1) if x > 0 else 0)
        clean_df['Baseline_R'] = baseline_r - adjustments
        clean_df['Baseline_D'] = baseline_d - (baseline_other - adjustments)
        clean_df['Baseline_O'] = baseline_other

    for raw_name, clean_name in demographics.items():
        if raw_name not in df.columns:
            continue
        clean_df[clean_name] = df[raw_name].apply(to_numeric_clean)
    
    for race in ["White", "Black", "Asian", "Native", "Latino"]:
        old_name = f"Citizen Voting Age Population {race}"
        new_name = f"Percent {race} VAP"
        if old_name in clean_df.columns and "VAP" in clean_df.columns:
            clean_df[new_name] = (clean_df[old_name] / clean_df["VAP"]).replace([float('inf'), -float('inf')], pd.NA)
            clean_df.drop(columns=[old_name], inplace=True)
    for education in ["High School", "Bachelors", "Graduate"]:
        old_name = f"Count {education} Degree"
        new_name = f"Percent {education} Attainment"
        if old_name in clean_df.columns and "VAP" in clean_df.columns:
            clean_df[new_name] = (clean_df[old_name] / clean_df["VAP"]).replace([float('inf'), -float('inf')], pd.NA)
            clean_df.drop(columns=[old_name], inplace=True)
    for age in ["Age 65+"]:
        old_name = f"Count {age}"
        new_name = f"Percent Senior VAP"
        if old_name in clean_df.columns and "VAP" in clean_df.columns:
            clean_df[new_name] = (clean_df[old_name] / clean_df["VAP"]).replace([float('inf'), -float('inf')], pd.NA)
            clean_df.drop(columns=[old_name], inplace=True)
    for income in ["Income <25k", "Income 25-50k", "Income 50-100k", "Income 100-200k", "Income >200k"]:
        old_name = f"Count {income}"
        new_name = f"Percent {income} Households"
        if old_name in clean_df.columns and "Number of Households" in clean_df.columns:
            clean_df[new_name] = (clean_df[old_name] / clean_df["Number of Households"]).replace([float('inf'), -float('inf')], pd.NA)
            clean_df.drop(columns=[old_name], inplace=True)
    old_name = "Area (Square KM)"
    new_name = "Population Density (Per Square KM)"
    if old_name in clean_df.columns and "Population 2020" in clean_df.columns:
        clean_df[new_name] = (clean_df["Population 2020"] / clean_df[old_name]).replace([float('inf'), -float('inf')], pd.NA)
        clean_df.drop(columns=[old_name], inplace=True)

    election_pattern = re.compile(r'^(\d{4}_[A-Za-z0-9]+)_(D|R|O)$')
    election_groups = {}
    for col in clean_df.columns:
        match = election_pattern.match(col)
        if not match:
            continue
        label = match.group(1)
        party = match.group(2)
        election_groups.setdefault(label, {})[party] = col
    drop_columns = []
    for parties in election_groups.values():
        drop = False
        for party in ("D", "R"):
            col = parties.get(party)
            if col is None:
                continue
            series = clean_df[col]
            if series.isna().any() or (series == 0).any():
                drop = True
                break
        if drop:
            drop_columns.extend(parties.values())
    if drop_columns:
        clean_df.drop(columns=list(set(drop_columns)), inplace=True)

    output_path = Path(f"data/clean/{args.state}_{args.denomination.lower()}_cleaned.csv")
    clean_df.to_csv(output_path)
    print(f"Cleaned data written to {output_path}")

if __name__ == "__main__":
    main()
