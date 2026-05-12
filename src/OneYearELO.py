import math
import numpy as np
import pandas as pd
from pathlib import Path
import Teams

DATA_ROOT = Path("data/processed/")
OUTPUT_DIR = Path("results")

HISTORY_MODES = ["1season", "5season", "all"]  # "1season" = pouze předchozí sezóna, "5season" = předchozích 5 sezón, "all" = všechny předchozí sezóny

K = 25.0
HOME_ADV = 100.0
DRAW_BASE = 0.28
DRAW_SCALE = 300
FILTER_FILES = set()

BASE_ELO = {
    "E0": 1600, "E1": 1480, "E2": 1360, "E3": 1240,
    "D1": 1600, "D2": 1480,
    "SP1": 1600, "SP2": 1480,
    "I1": 1600, "I2": 1480,
    "F1": 1600, "F2": 1480,
    "SC0": 1500, "SC1": 1380, "SC2": 1260, "SC3": 1140,
    "N1": 1500,
    "B1": 1500,
    "P1": 1400,
    "G1": 1400,
    "T1": 1400,
    "EC": 1120,
}

COUNTRY_BY_DIV = {
    "E0": "England", "E1": "England", "E2": "England", "E3": "England",
    "D1": "Germany", "D2": "Germany",
    "SP1": "Spain", "SP2": "Spain",
    "I1": "Italy", "I2": "Italy",
    "F1": "France", "F2": "France",
    "SC0": "Scotland", "SC1": "Scotland", "SC2": "Scotland", "SC3": "Scotland",
    "N1": "Netherlands",
    "B1": "Belgium",
    "P1": "Portugal",
    "G1": "Greece",
    "T1": "Turkey",
    "EC": "England",
}


# =========================
# METRIKY
# =========================

def probs_from_elo(rh, ra):
    p_home_binary = 1.0 / (1.0 + 10 ** (-(((rh + HOME_ADV) - ra) / 400.0)))

    p_draw = DRAW_BASE * math.exp((-abs(rh + HOME_ADV - ra) / 400.0)/DRAW_SCALE)
    p_home = (1.0 - p_draw) * p_home_binary
    p_away = (1.0 - p_draw) * (1.0 - p_home_binary)

    total = p_home + p_draw + p_away

    return {
        "H": p_home / total,
        "D": p_draw / total,
        "A": p_away / total,
    }


def predict_from_probs(probs):
    return max(probs, key=probs.get)


def logloss_one(true, probs):
    return -math.log(max(probs[true], 1e-15))


def brier_one(true, probs):
    return sum((probs[c] - (1 if c == true else 0)) ** 2 for c in ["H", "D", "A"])


def rps_one(true, probs):
    y = np.array([1 if c == true else 0 for c in ["H", "D", "A"]])
    p = np.array([probs[c] for c in ["H", "D", "A"]])
    return float(np.sum((np.cumsum(p) - np.cumsum(y)) ** 2) / 2)


# =========================
# ELO + TEAM CLASS
# =========================

def expected_home(r_home, r_away):
    return 1.0 / (1.0 + 10 ** (-(((r_home + HOME_ADV) - r_away) / 400.0)))


def outcome_home(hg, ag):
    if hg > ag:
        return 1.0
    if hg == ag:
        return 0.5
    return 0.0


def get_or_create_team(teams, name, div, season, alpha=0.6):
    base = BASE_ELO.get(div, 1500)

    # nový tým
    if name not in teams:
        team = Teams.Team(name, base, div)
        team.mark_seen(season)
        teams[name] = team
        return team

    team = teams[name]

    # tým se vrací po neaktivitě
    if not team.active or team.get_elo() is None or team.get_division() is None:
        last = team.last_elo if team.last_elo is not None else base
        new_elo = base + 0.5 * (last - base)
        team.reactivate(div, round(new_elo, 2), season)
        team.mark_seen(season)
        return team

    # změna divize
    if team.get_division() != div:
        old_div = team.get_division()

        old_base = BASE_ELO.get(old_div, base)
        new_base = BASE_ELO.get(div, base)

        current_elo = team.get_elo()
        if current_elo is None:
            current_elo = old_base

        new_elo = current_elo + alpha * (new_base - old_base)

        team.set_division(div)
        team.set_elo(round(new_elo, 2))

    team.mark_seen(season)
    return team


def mark_missing_teams(teams, current_season):
    for team in teams.values():
        if team.last_seen_season != current_season:
            team.mark_missing()


def elo_from_matches(df_train):
    teams = {}

    df_train = df_train.sort_values(["Season", "Date"]).copy()

    for row in df_train.itertuples(index=False):
        season = row.Season

        home_team = get_or_create_team(teams, row.HomeTeam, row.Div, season)
        away_team = get_or_create_team(teams, row.AwayTeam, row.Div, season)

        rh = home_team.get_elo()
        ra = away_team.get_elo()

        if rh is None:
            rh = BASE_ELO.get(row.Div, 1500)
            home_team.set_elo(rh)

        if ra is None:
            ra = BASE_ELO.get(row.Div, 1500)
            away_team.set_elo(ra)

        p = expected_home(rh, ra)
        s = outcome_home(row.FTHG, row.FTAG)

        home_post = rh + K * (s - p)
        away_post = ra + K * ((1.0 - s) - (1.0 - p))

        home_team.set_elo(round(home_post, 2))
        away_team.set_elo(round(away_post, 2))

        home_team.mark_seen(season)
        away_team.mark_seen(season)

    return teams


# =========================
# DATA
# =========================

def load_match_file(path):
    df = pd.read_csv(path)

    needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Div"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"{path}: chybí sloupce {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    df = df.dropna(subset=["Date", "FTHG", "FTAG"]).copy()

    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTR"] = df["FTR"].astype(str).str.strip()
    df["Div"] = df["Div"].astype(str).str.strip()

    df["Season"] = path.parent.name
    df["File"] = path.name
    df["Country"] = df["Div"].map(COUNTRY_BY_DIV)

    df = df.dropna(subset=["Country"]).copy()

    return df


def infer_round_per_season(df):
    if "Round" in df.columns:
        return df

    df = df.sort_values(["Country", "Season", "Date"]).copy()
    rounds = np.zeros(len(df), dtype=int)

    for (country, season, div), idx in df.groupby(["Country", "Season", "Div"]).groups.items():
        played = {}
        r_list = []

        for i in idx:
            h = df.at[i, "HomeTeam"]
            a = df.at[i, "AwayTeam"]

            r = max(played.get(h, 0), played.get(a, 0)) + 1
            r_list.append(r)

            played[h] = played.get(h, 0) + 1
            played[a] = played.get(a, 0) + 1

        rounds[list(idx)] = r_list

    df["Round"] = rounds
    return df


def load_all_matches(root):
    parts = []

    for season_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for csv_path in sorted(season_dir.glob("*.csv")):
            if FILTER_FILES and csv_path.name not in FILTER_FILES:
                continue

            try:
                part = load_match_file(csv_path)
                if not part.empty:
                    parts.append(part)
            except Exception as e:
                print(f"ERR load {csv_path} -> {e}")

    if not parts:
        raise FileNotFoundError(f"Nenašel jsem žádná použitelná CSV v {root.resolve()}")

    df = pd.concat(parts, ignore_index=True)
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)
    df = infer_round_per_season(df)

    return df


def get_previous_seasons(all_seasons, current_season, history_mode):
    seasons_sorted = sorted(all_seasons)
    idx = seasons_sorted.index(current_season)

    if history_mode == "1season":
        return seasons_sorted[max(0, idx - 1):idx]

    if history_mode == "5season":
        return seasons_sorted[max(0, idx - 5):idx]

    if history_mode == "all":
        return seasons_sorted[:idx]

    raise ValueError("history_mode musí být '1season', '5season' nebo 'all'")


# =========================
# EVALUACE
# =========================

def evaluate_country(df_country, country, history_mode):
    results = []
    all_seasons = sorted(df_country["Season"].unique())

    for season, df_season in df_country.groupby("Season", sort=True):
        df_season = df_season.sort_values("Date")

        for div, df_div_season in df_season.groupby("Div", sort=True):
            for r in sorted(df_div_season["Round"].unique()):
                df_round = df_div_season[df_div_season["Round"] == r].copy()
                round_start = df_round["Date"].min()

                previous_seasons = get_previous_seasons(
                    all_seasons,
                    season,
                    history_mode
                )

                df_train = df_country[
                    (df_country["Season"].isin(previous_seasons)) |
                    ((df_country["Season"] == season) & (df_country["Date"] < round_start))
                ].copy()

                df_train = df_train.sort_values(["Season", "Date"])

                if df_train.empty:
                    continue

                teams = elo_from_matches(df_train)

                total = 0
                correct = 0
                logloss_sum = 0.0
                brier_sum = 0.0
                rps_sum = 0.0

                for row in df_round.itertuples(index=False):
                    true = row.FTR
                    base = BASE_ELO.get(row.Div, 1500)

                    home_team = teams.get(row.HomeTeam)
                    away_team = teams.get(row.AwayTeam)

                    rh = home_team.get_elo() if home_team is not None else base
                    ra = away_team.get_elo() if away_team is not None else base

                    probs = probs_from_elo(rh, ra)
                    pred = predict_from_probs(probs)

                    total += 1
                    correct += int(pred == true)

                    logloss_sum += logloss_one(true, probs)
                    brier_sum += brier_one(true, probs)
                    rps_sum += rps_one(true, probs)

                if total == 0:
                    continue

                results.append({
                    "country": country,
                    "season": season,
                    "file": div,
                    "round": int(r),
                    "round_start": round_start.date().isoformat(),
                    "history_mode": history_mode,
                    "matches": int(total),
                    "correct": int(correct),
                    "accuracy": correct / total,
                    "logloss": logloss_sum / total,
                    "brier": brier_sum / total,
                    "rps": rps_sum / total,
                })

    return pd.DataFrame(results)


def make_summary_by_country(res):
    rows = []

    for country, g in res.groupby("country", sort=True):
        matches = g["matches"].sum()

        rows.append({
            "country": country,
            "matches": int(matches),
            "correct": int(g["correct"].sum()),
            "accuracy": g["correct"].sum() / matches,
            "logloss": (g["logloss"] * g["matches"]).sum() / matches,
            "brier": (g["brier"] * g["matches"]).sum() / matches,
            "rps": (g["rps"] * g["matches"]).sum() / matches,
        })

    return pd.DataFrame(rows)


def print_summary(res, history_mode):
    total_matches = res["matches"].sum()

    overall_accuracy = res["correct"].sum() / total_matches
    weighted_logloss = (res["logloss"] * res["matches"]).sum() / total_matches
    weighted_brier = (res["brier"] * res["matches"]).sum() / total_matches
    weighted_rps = (res["rps"] * res["matches"]).sum() / total_matches

    print("\n=== SUMMARY ===")
    print(f"History mode: {history_mode}")
    print(f"Total matches: {total_matches}")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"LogLoss: {weighted_logloss:.4f}")
    print(f"Brier: {weighted_brier:.4f}")
    print(f"RPS: {weighted_rps:.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_matches(DATA_ROOT)

    for history_mode in HISTORY_MODES:
        print("\n" + "=" * 70)
        print(f"SPOUŠTÍM HISTORY_MODE = {history_mode}")
        print("=" * 70)

        all_results = []

        for country, df_country in df.groupby("Country", sort=True):
            try:
                out = evaluate_country(df_country.copy(), country, history_mode)

                if not out.empty:
                    all_results.append(out)

                print(f"OK: {country} rows={len(out)}")

            except Exception as e:
                print(f"ERR: {country} -> {e}")

        if not all_results:
            print(f"Nic nebylo vyhodnoceno pro {history_mode}.")
            continue

        res = pd.concat(all_results, ignore_index=True)
        summary_country = make_summary_by_country(res)

        output_csv = OUTPUT_DIR / f"results_country_elo_{history_mode}.csv"
        summary_csv = OUTPUT_DIR / f"results_country_elo_{history_mode}_summary_by_country.csv"

        res.to_csv(output_csv, index=False, encoding="utf-8")
        summary_country.to_csv(summary_csv, index=False, encoding="utf-8")

        print_summary(res, history_mode)

        print(f"\nUloženo do: {output_csv.resolve()}")
        print(f"Souhrn podle států: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()