import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# =========================
# NASTAVENÍ
# =========================
DATA_ROOT = Path("data/processed/")
OUTPUT_CSV = Path("results/results_dumb_sazkar.csv")

FILTER_FILES = set()  # např. {"E0.csv"} nebo prázdné = všechny
EPS = 1e-15
CLASS_ORDER = ["H", "D", "A"]

# =========================
# NAČTENÍ + PŘÍPRAVA
# =========================
def load_match_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = {"Date", "HomeTeam", "AwayTeam", "FTR"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: chybí sloupce {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["Season"] = path.parent.name
    df["File"] = path.name
    return df


def infer_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pokud chybí Round, dopočítá ho pro JEDNU sezónu.
    """
    if "Round" in df.columns:
        return df

    df = df.sort_values("Date").copy()
    played = {}
    rounds = []

    for row in df.itertuples(index=False):
        h = row.HomeTeam
        a = row.AwayTeam
        ph = played.get(h, 0)
        pa = played.get(a, 0)
        r = max(ph, pa) + 1
        rounds.append(r)
        played[h] = ph + 1
        played[a] = pa + 1

    df["Round"] = rounds
    return df


# =========================
# TABULKA + PREDIKCE
# =========================
def build_points_table(df_train: pd.DataFrame) -> dict:
    points = {}

    def add(team, p):
        points[team] = points.get(team, 0) + p

    for row in df_train.itertuples(index=False):
        h = row.HomeTeam
        a = row.AwayTeam
        r = str(row.FTR).strip()

        points.setdefault(h, 0)
        points.setdefault(a, 0)

        if r == "H":
            add(h, 3)
        elif r == "A":
            add(a, 3)
        elif r == "D":
            add(h, 1)
            add(a, 1)

    return points


def dumb_bettor_predict(points: dict, home: str, away: str) -> str:
    ph = points.get(home, 0)
    pa = points.get(away, 0)

    if ph > pa:
        return "H"
    if ph < pa:
        return "A"
    return "D"

def probs_from_prediction(pred: str, main_prob: float = 0.80) -> dict:
    rest = (1.0 - main_prob) / 2
    probs = {"H": rest, "D": rest, "A": rest}
    probs[pred] = main_prob
    return probs

def log_loss_one(true: str, probs: dict) -> float:
    return -np.log(max(probs[true], EPS))


def brier_one(true: str, probs: dict) -> float:
    return sum((probs[c] - (1 if c == true else 0)) ** 2 for c in CLASS_ORDER)


def rps_one(true: str, probs: dict) -> float:
    y = np.array([1 if c == true else 0 for c in CLASS_ORDER])
    p = np.array([probs[c] for c in CLASS_ORDER])
    return float(np.sum((np.cumsum(p) - np.cumsum(y)) ** 2) / (len(CLASS_ORDER) - 1))

# =========================
# EVALUACE
# =========================
def evaluate_file(path: Path) -> pd.DataFrame:
    df = load_match_file(path)
    df = infer_round(df)

    results = []

    for r in sorted(df["Round"].unique()):
        df_round = df[df["Round"] == r]
        df_train = df[df["Round"] < r]

        # pro r=1 bude df_train prázdné, points = {}
        points = build_points_table(df_train)

        total = 0
        correct = 0
        logloss_sum = 0.0
        brier_sum = 0.0
        rps_sum = 0.0
        for row in df_round.itertuples(index=False):
            true = str(row.FTR).strip()
            if true not in {"H", "D", "A"}:
                continue

            pred = dumb_bettor_predict(points, row.HomeTeam, row.AwayTeam)
            probs = probs_from_prediction(pred)

            logloss_sum += log_loss_one(true, probs)
            brier_sum += brier_one(true, probs)
            rps_sum += rps_one(true, probs)
            total += 1
            if pred == true:
                correct += 1

        if total == 0:
            continue

        results.append({
            "season": df["Season"].iloc[0],
            "file": path.name,
            "round": int(r),
            "matches": int(total),
            "correct": int(correct),
            "accuracy": float(correct / total),
            "logloss": float(logloss_sum / total),
            "brier": float(brier_sum / total),
            "rps": float(rps_sum / total) 
        })

    return pd.DataFrame(results)


def main():
    all_parts = []

    for season_dir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()]):
        for csv_path in sorted(season_dir.glob("*.csv")):
            if FILTER_FILES and csv_path.name not in FILTER_FILES:
                continue
            try:
                part = evaluate_file(csv_path)
                if not part.empty:
                    all_parts.append(part)
                print(f"OK: {season_dir.name}/{csv_path.name}")
            except Exception as e:
                print(f"ERR: {season_dir.name}/{csv_path.name} -> {e}")

    if not all_parts:
        print("Nic nebylo vyhodnoceno.")
        return

    out = pd.concat(all_parts, ignore_index=True)
    overall = out["correct"].sum() / out["matches"].sum()

    print("\n=== SUMMARY ===")
    print(f"Total matches: {out['matches'].sum()}")
    print(f"Overall accuracy: {overall:.4f}")

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Uloženo do: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()