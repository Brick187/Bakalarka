import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque

DATA_PATH = "data/processed/"
FORM_N = 5

def safe_read_csv(path):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise ValueError(f"Nepodařilo se načíst {path}")

def parse_result_points(result, is_home):
    if result == "D":
        return 1
    if result == "H":
        return 3 if is_home else 0
    if result == "A":
        return 0 if is_home else 3
    return np.nan

def avg_or_nan(values):
    return np.mean(values) if len(values) > 0 else np.nan

def get_available_odds_columns(df):
    home_cols = [
        "1XBH","B365H","BFH","BFDH","BMGMH","BVH","BSH","BWH","CLH","GBH",
        "IWH","LBH","PSH","PH","SOH","SBH","SJH","SYH","VCH","WHH"
    ]
    draw_cols = [
        "1XBD","B365D","BFD","BFDD","BMGMD","BVD","BSD","BWD","CLD","GBD",
        "IWD","LBD","PSD","PD","SOD","SBD","SJD","SYD","VCD","WHD"
    ]
    away_cols = [
        "1XBA","B365A","BFA","BFDA","BMGMA","BVA","BSA","BWA","CLA","GBA",
        "IWA","LBA","PSA","PA","SOA","SBA","SJA","SYA","VCA","WHA"
    ]

    available_home = [c for c in home_cols if c in df.columns]
    available_draw = [c for c in draw_cols if c in df.columns]
    available_away = [c for c in away_cols if c in df.columns]

    return available_home, available_draw, available_away

def to_float_safe(value):
    if pd.isna(value):
        return np.nan

    # když je to string, očisti ho
    if isinstance(value, str):
        value = value.strip().replace(",", ".")

        # prázdný string
        if value == "":
            return np.nan

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def extract_odds_features(row, home_cols, draw_cols, away_cols):
    home_odds = [to_float_safe(row[c]) for c in home_cols if pd.notna(row[c])]
    draw_odds = [to_float_safe(row[c]) for c in draw_cols if pd.notna(row[c])]
    away_odds = [to_float_safe(row[c]) for c in away_cols if pd.notna(row[c])]

    if len(home_odds) == 0 or len(draw_odds) == 0 or len(away_odds) == 0:
        return None

    avg_h = float(np.mean(home_odds))
    avg_d = float(np.mean(draw_odds))
    avg_a = float(np.mean(away_odds))

    min_h, max_h = float(np.min(home_odds)), float(np.max(home_odds))
    min_d, max_d = float(np.min(draw_odds)), float(np.max(draw_odds))
    min_a, max_a = float(np.min(away_odds)), float(np.max(away_odds))

    # implied probabilities
    p_h = 1.0 / avg_h
    p_d = 1.0 / avg_d
    p_a = 1.0 / avg_a

    total = p_h + p_d + p_a
    if total <= 0:
        return None

    p_h /= total
    p_d /= total
    p_a /= total

    return {
        "OddsAvgH": avg_h,
        "OddsAvgD": avg_d,
        "OddsAvgA": avg_a,

        "OddsProbHome": p_h,
        "OddsProbDraw": p_d,
        "OddsProbAway": p_a,

        "OddsMaxHome": max_h,
        "OddsMaxDraw": max_d,
        "OddsMaxAway": max_a,

        "OddsMinHome": min_h,
        "OddsMinDraw": min_d,
        "OddsMinAway": min_a,

        "OddsSpreadHome": max_h - min_h,
        "OddsSpreadDraw": max_d - min_d,
        "OddsSpreadAway": max_a - min_a,

        "OddsCountHome": len(home_odds),
        "OddsCountDraw": len(draw_odds),
        "OddsCountAway": len(away_odds),
    }

def load_all_matches(data_path):
    all_dfs = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if not file.lower().endswith(".csv"):
                continue

            path = os.path.join(root, file)
            df = safe_read_csv(path)

            needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
            if not all(col in df.columns for col in needed):
                print(f"Přeskakuju {path}, chybí základní sloupce")
                continue

            season = os.path.basename(os.path.dirname(path))
            div = file.replace(".csv", "")

            df["Season"] = season
            df["Div"] = div
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
            df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
            df["EloHome"] = df["EloHome"].astype(float)
            df["EloAway"] = df["EloAway"].astype(float)

            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "EloHome", "EloAway"]).copy()
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("Nenačetly se žádné zápasy")

    matches = pd.concat(all_dfs, ignore_index=True)
    matches = matches.sort_values(["Date", "Season", "Div"]).reset_index(drop=True)
    return matches

def build_feature_dataset(matches, form_n=5):
    last_season = None
    table_points = defaultdict(int)
    overall_points = defaultdict(lambda: deque(maxlen=form_n))
    last_match_date = {}
    streaks = defaultdict(int)

    shots_hist = defaultdict(lambda: deque(maxlen=form_n))
    shots_target_hist = defaultdict(lambda: deque(maxlen=form_n))
    overall_gf = defaultdict(lambda: deque(maxlen=form_n))
    overall_ga = defaultdict(lambda: deque(maxlen=form_n))

    home_points = defaultdict(lambda: deque(maxlen=form_n))
    away_points = defaultdict(lambda: deque(maxlen=form_n))

    home_odds_cols, draw_odds_cols, away_odds_cols = get_available_odds_columns(matches)

    if not home_odds_cols or not draw_odds_cols or not away_odds_cols:
        print("Nenalezen žádný kompletní set odds sloupců, nelze extrahovat features z kurzů.")
        return pd.DataFrame()

    print("Nalezené odds sloupce:")
    print("Home:", home_odds_cols)
    print("Draw:", draw_odds_cols)
    print("Away:", away_odds_cols)

    rows = []

    for _, row in matches.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        ftr = row["FTR"]
        fthg = row["FTHG"]
        ftag = row["FTAG"]

        elo_home = row["EloHome"] 
        elo_away = row["EloAway"] 

        odds_features = extract_odds_features(
            row,
            home_odds_cols,
            draw_odds_cols,
            away_odds_cols
        )

        if odds_features is None:
            continue

        # =====================
        # REST DAYS
        # =====================
        if last_season is not None and row["Season"] != last_season:
            # reset tabulky a formy na začátku nové sezóny
            table_points = defaultdict(int)
            last_match_date = {}
        date = row["Date"]

        days_rest_home = (
            (date - last_match_date[home]).days
            if home in last_match_date else np.nan
        )

        days_rest_away = (
            (date - last_match_date[away]).days
            if away in last_match_date else np.nan
        )

        # =====================
        # GOAL DIFFERENCE
        # =====================
        goal_diff_home = avg_or_nan(overall_gf[home]) - avg_or_nan(overall_ga[home])
        goal_diff_away = avg_or_nan(overall_gf[away]) - avg_or_nan(overall_ga[away])

        # střely v aktuálním zápase – jen pokud existují ve vstupních datech
        hs = to_float_safe(row["HS"]) if "HS" in row.index else np.nan
        hst = to_float_safe(row["HST"]) if "HST" in row.index else np.nan
        as_ = to_float_safe(row["AS"]) if "AS" in row.index else np.nan
        ast = to_float_safe(row["AST"]) if "AST" in row.index else np.nan

        table_points[home]  
        table_points[away]  

        last_season = row["Season"]

        # =====================
        # FEATURE ROW
        # =====================
        feature_row = {
            "Season": row["Season"],
            "Div": row["Div"],
            "Date": date,

            # =====================
            # TÝMY
            # =====================
            "HomeTeam": home,
            "AwayTeam": away,

            # =====================
            # ELO
            # =====================
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away,

            # =====================
            # FORMA
            # =====================
            "form_points_home": avg_or_nan(overall_points[home]),
            "form_points_away": avg_or_nan(overall_points[away]),

            "shots_home": avg_or_nan(shots_hist[home]),
            "shots_away": avg_or_nan(shots_hist[away]),
            "shots_on_target_home": avg_or_nan(shots_target_hist[home]),
            "shots_on_target_away": avg_or_nan(shots_target_hist[away]),

            "goals_scored_home": avg_or_nan(overall_gf[home]),
            "goals_conceded_home": avg_or_nan(overall_ga[home]),
            "goals_scored_away": avg_or_nan(overall_gf[away]),
            "goals_conceded_away": avg_or_nan(overall_ga[away]),

            "goal_diff_home": goal_diff_home,
            "goal_diff_away": goal_diff_away,

            # =====================
            # DOMA / VENKU
            # =====================
            "home_form_home": avg_or_nan(home_points[home]),
            "away_form_away": avg_or_nan(away_points[away]),

            # =====================
            # STREAK
            # =====================
            "streak_home": streaks[home],
            "streak_away": streaks[away],
            "table_points_home": table_points[home],
            "table_points_away": table_points[away],
            "table_points_diff": table_points[home] - table_points[away],

            # =====================
            # KONTEXT
            # =====================
            "B365H": to_float_safe(row["B365H"]) if "B365H" in row.index else np.nan,
            "B365D": to_float_safe(row["B365D"]) if "B365D" in row.index else np.nan,
            "B365A": to_float_safe(row["B365A"]) if "B365A" in row.index else np.nan,
            "is_home": 1,
            "days_rest_home": days_rest_home,
            "days_rest_away": days_rest_away,
            # =====================
            # TARGET
            # =====================
            "Target": ftr
        }

        feature_row.update(odds_features)
        rows.append(feature_row)

        home_pts = parse_result_points(ftr, is_home=True)
        away_pts = parse_result_points(ftr, is_home=False)

        overall_points[home].append(home_pts)
        overall_points[away].append(away_pts)

        overall_gf[home].append(fthg)
        overall_ga[home].append(ftag)

        overall_gf[away].append(ftag)
        overall_ga[away].append(fthg)

        # update střel jen pokud existují
        if not pd.isna(hs):
            shots_hist[home].append(hs)
        if not pd.isna(as_):
            shots_hist[away].append(as_)

            if not pd.isna(hst):
                shots_target_hist[home].append(hst)
            if not pd.isna(ast):
                shots_target_hist[away].append(ast)

        home_points[home].append(home_pts)
        away_points[away].append(away_pts)
        # =====================
        # UPDATE TABULKY
        # =====================
        if ftr == "H":
            table_points[home] += 3
        elif ftr == "A":
            table_points[away] += 3
        elif ftr == "D":
            table_points[home] += 1
            table_points[away] += 1

        # =====================
        # UPDATE STREAK
        # =====================
        if ftr == "H":
            streaks[home] += 1
            streaks[away] = 0
        elif ftr == "A":
            streaks[away] += 1
            streaks[home] = 0
        else:
            streaks[home] = 0
            streaks[away] = 0

        # =====================
        # UPDATE REST
        # =====================
        last_match_date[home] = date
        last_match_date[away] = date
    return pd.DataFrame(rows)

if __name__ == "__main__":
    matches = load_all_matches(DATA_PATH)
    feature_df = build_feature_dataset(matches, form_n=5)

    target_map = {"H": 0, "D": 1, "A": 2}
    feature_df = feature_df[feature_df["Target"].isin(target_map)].copy()
    feature_df["TargetEncoded"] = feature_df["Target"].map(target_map)

    feature_df["form_points_home"] = feature_df["form_points_home"].fillna(1.5)
    feature_df["form_points_away"] = feature_df["form_points_away"].fillna(1.5)

    for col in [
        "goals_scored_home", "goals_conceded_home",
        "goals_scored_away", "goals_conceded_away",
        "shots_home", "shots_away",
        "shots_on_target_home", "shots_on_target_away",
        "goal_diff_home", "goal_diff_away"
    ]:
        feature_df[col] = feature_df[col].fillna(feature_df[col].mean())

    for col in [
        "home_form_home", "away_form_away",
        "days_rest_home", "days_rest_away",
        "streak_home", "streak_away"
    ]:
        feature_df[col] = feature_df[col].fillna(0)
    feature_df.to_csv("matches_with_features.csv", index=False)
    print("Hotovo: matches_with_features.csv")
    print(feature_df.head())