import pandas as pd
from pathlib import Path
import Teams

# --- Parametry Elo ---
INITIAL_ELO = 1600        # startovní Elo pro úvodní sezónu (v rámci dané ligy)
K = 25                    # citlivost aktualizace Elo
HOME_ADV = 100             # domácí výhoda v Elo bodech (použita jen pro očekávání)
SAVE_SUFFIX = "_elo.csv"  # výstupní přípona
league_elo_dict = {
    "E0": 1600,
    "E1": 1480,
    "E2": 1360,
    "E3": 1240,
    "D1": 1600,
    "D2": 1480,
    "SP1": 1600,
    "N1": 1500,
    "I1": 1600,
    "F1": 1600,
    "B1": 1500,
    "EC": 1120,
    "SC0": 1500,
    "SC1": 1380,
    "SC2": 1260,
    "SC3": 1140,
    "F2": 1480,
    "I2": 1480,
    "P1": 1400,
    "G1": 1400,
    "SP2": 1480,
    "T1": 1400,
}

def expected_result(elo_a, elo_b):
    """Očekávaný bodový zisk A (domácí) proti B (hosté) v 0..1."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def match_outcome(fthg, ftag):
    """Vrátí 1/0.5/0 pro domácí tým podle výsledku."""
    if fthg > ftag:
        return 1.0
    elif fthg == ftag:
        return 0.5
    else:
        return 0.0

def create_teams_dict(csv_path, global_teams, season):
    """
    Vytvoří/aktualizuje týmy z CSV souboru podle globálního slovníku.
    global_teams = slovník všech týmů napříč sezónami.
    """
    df = pd.read_csv(csv_path)

    # kontrola sloupců
    required = ["HomeTeam", "AwayTeam", "Div"]
    if not set(required).issubset(df.columns):
        print("Chybí sloupce v:", csv_path)
        print("mám:", list(df.columns))
        return global_teams

    for _, row in df.iterrows():
        h = row["HomeTeam"]
        a = row["AwayTeam"]
        div = row["Div"]

        # přeskoč ligy, které nejsou sledované
        if div not in league_elo_dict:
            continue
        team = process_team(global_teams, h, div, league_elo_dict, season)
        team.mark_seen(season)
        team = process_team(global_teams, a, div, league_elo_dict, season)
        team.mark_seen(season)

    return global_teams

def process_team(global_teams, name, new_div, baselines, season, alpha=0.6):
    """
    Zpracuje jeden tým v sezóně:
    - nový tým
    - návrat po vypadnutí
    - změna divize (postup/sestup)
    - normální pokračování
    """
    # NOVÝ TÝM
    if name not in global_teams:
        start_elo = baselines[new_div]
        team = Teams.Team(name, start_elo, new_div)
        team.last_seen_season = season
        global_teams[name] = team
        return team

    team = global_teams[name]

    # NÁVRAT TÝMU
    if not team.active:
        handle_team_return(team, new_div, baselines, season)
        return team

    # ŽÁDNÁ ZMĚNA DIVIZE
    if team.get_division() == new_div:
        team.last_seen_season = season
        return team

    # ZMĚNA DIVIZE (POSTUP/SESTUP)
    old_div = team.get_division()
    old_base = baselines[old_div]
    new_base = baselines[new_div]

    new_elo = team.get_elo() + alpha * (new_base - old_base)

    team.set_division(new_div)
    team.set_elo(round(new_elo, 2))
    team.last_seen_season = season

    return team

def handle_team_return(team, new_div, baselines, season, alpha=0.5):
    """
    Tým se vrací do sledované ligy.
    Nastaví mu Elo regresí z jeho posledního Elo.
    """
    base = baselines[new_div]
    last = team.last_elo
    if last is None:
        # bezpečný fallback – chovej se, jako by začínal od baseline
        last = base
    # regrese směrem k síle ligy
    new_elo = base + alpha * (last - base)

    team.reactivate(new_div, round(new_elo, 2), season)
    return new_elo

def process_league_csv(csv_path, teams, k=K, home_adv=HOME_ADV, save_suffix=SAVE_SUFFIX):
    """
    Zpracuje jeden league CSV soubor (např. E0.csv) z úvodní sezóny:
    - doplní Elo před a po zápase
    - uloží nový soubor *_elo.csv
    """
    df = pd.read_csv(csv_path)
    # Ověření klíčových sloupců
    required_cols_any = [
        ("HomeTeam", "AwayTeam", "FTHG", "FTAG"),
        # některé staré soubory mívají zkratky HG/AG; doplň si sem další varianty, pokud narazíš
    ]
    ok = any(all(col in df.columns for col in variant) for variant in required_cols_any)
    if not ok:
        raise ValueError(f"{csv_path.name}: chybí povinné sloupce (HomeTeam, AwayTeam, FTHG, FTAG).")


    # Ujisti se, že skóre je numerické
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")


    for i, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]
        if h not in teams or a not in teams:
            continue
        # Pokud chybí skóre (např. prázdné řádky), přeskoč aktualizaci, ale ulož pre-match Elo
        home_pre = teams[h].get_elo()
        away_pre = teams[a].get_elo()

        # Očekávané body domácích s domácí výhodou
        exp_home = expected_result(home_pre + home_adv, away_pre)
        exp_away = 1.0 - exp_home

        # Zapiš pre-match hodnoty
        df.at[i, "EloHome"] = round(home_pre, 2)
        df.at[i, "EloAway"] = round(away_pre, 2)

        # Pokud máme výsledek, aktualizuj Elo
        if pd.notna(hg) and pd.notna(ag):
            outcome_home = match_outcome(int(hg), int(ag))
            # update domácích proti hostům s HA přidanou do očekávání (ne do ratingu)
            home_post = home_pre + k * (outcome_home - exp_home)
            away_post = away_pre + k * ((1.0 - outcome_home) - exp_away)
            home_post = round(home_post, 2)
            away_post = round(away_post, 2)
            teams[h].set_elo(home_post)
            teams[a].set_elo(away_post)

    # Uložení
    out_path = csv_path.with_name(csv_path.stem + save_suffix)
    df.to_csv(out_path, index=False)
    return out_path

def mark_missing_teams(global_teams, current_season):
    for t in global_teams.values():
        if t.last_seen_season != current_season:
            t.mark_missing()

# -----------------------------
# Hlavní běh programu
# -----------------------------
input_root = Path("data/raw/")
output_root = Path("data/processed/")

list_teams = {}

for season in input_root.glob("*/"):
    season_name = season.name
    output_season = output_root / season_name
    output_season.mkdir(parents=True, exist_ok=True)

    for csv_file in season.glob("*.csv"):
        try:
            # vytvoření týmů
            list_teams.update(
                create_teams_dict(csv_file, list_teams, season_name)
            )

            # výpočet Elo
            out = process_league_csv(csv_file, list_teams)

            # nový název souboru
            new_name = csv_file.stem + "_elo.csv"

            # přesun do správné složky
            final_path = output_season / new_name

            Path(out).rename(final_path)

            print(f"OK: {csv_file} -> {final_path}")

        except Exception as e:
            print(f"CHYBA {csv_file.name}: {e}")

    mark_missing_teams(list_teams, season_name)


