import pandas as pd
import random
from pathlib import Path
import numpy as np

# =========================
# NASTAVENÍ
# =========================
DATA_ROOT = Path("data/processed/")
OUTPUT_CSV = Path("results/results_random_empirical.csv")

FILTER_FILES = set()  # např. {"E0.csv"} nebo prázdné = vše
EPS = 1e-15
CLASS_ORDER = ["H", "D", "A"]


# =========================
# NAČTENÍ + ROUND
# =========================
def load_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = {"Date", "HomeTeam", "AwayTeam", "FTR"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path}: chybí sloupce {needed}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df["Season"] = path.parent.name
    return df


def infer_round(df: pd.DataFrame) -> pd.DataFrame:
    if "Round" in df.columns:
        return df

    played = {}
    rounds = []

    for row in df.itertuples(index=False):
        h, a = row.HomeTeam, row.AwayTeam
        r = max(played.get(h, 0), played.get(a, 0)) + 1
        rounds.append(r)
        played[h] = played.get(h, 0) + 1
        played[a] = played.get(a, 0) + 1

    df["Round"] = rounds
    return df


# =========================
# EMPIRICKÉ PRAVDĚPODOBNOSTI
# =========================
def empirical_probs(df_train: pd.DataFrame) -> dict:
    """
    Spočítá P(H), P(D), P(A) z četností v df_train.
    """
    counts = {"H": 0, "D": 0, "A": 0}

    for r in df_train["FTR"]:
        r = str(r).strip()
        if r in counts:
            counts[r] += 1

    total = sum(counts.values())
    if total == 0:
        # fallback – rovnoměrně
        return {"H": 1/3, "D": 1/3, "A": 1/3}

    return {k: v / total for k, v in counts.items()}


def random_from_probs(probs: dict) -> str:
    return random.choices(
        population=["H", "D", "A"],
        weights=[probs["H"], probs["D"], probs["A"]],
        k=1
    )[0]


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
    df = infer_round(load_file(path))
    results = []

    for r in sorted(df["Round"].unique()):
        df_round = df[df["Round"] == r]

        # trénink = předchozí kola téže sezóny
        df_train = df[df["Round"] < r]

        # pro 1. kolo df_train prázdné → fallback 1/3–1/3–1/3
        probs = empirical_probs(df_train)

        total = 0
        correct = 0
        logloss_sum = 0.0
        brier_sum = 0.0
        rps_sum = 0.0

        for row in df_round.itertuples(index=False):
            true = str(row.FTR).strip()
            if true not in {"H", "D", "A"}:
                continue

            pred = random_from_probs(probs)
            logloss_sum += log_loss_one(true, probs)
            brier_sum += brier_one(true, probs)
            rps_sum += rps_one(true, probs)
            total += 1
            if pred == true:
                correct += 1

        if total:
            results.append({
                "season": df["Season"].iloc[0],
                "file": path.name,
                "round": int(r),
                "p_H": probs["H"],
                "p_D": probs["D"],
                "p_A": probs["A"],
                "matches": total,
                "correct": correct,
                "accuracy": correct / total,
                "logloss": logloss_sum / total,
                "brier": brier_sum / total,
                "rps": rps_sum / total
            })

    return pd.DataFrame(results)


def main():
    out = []

    for season_dir in DATA_ROOT.iterdir():
        if not season_dir.is_dir():
            continue
        for csv in season_dir.glob("*.csv"):
            if FILTER_FILES and csv.name not in FILTER_FILES:
                continue
            out.append(evaluate_file(csv))

    res = pd.concat(out, ignore_index=True)
    overall = res["correct"].sum() / res["matches"].sum()

    print("Overall accuracy:", round(overall, 4))
    res.to_csv(OUTPUT_CSV, index=False)
    print(f"Uloženo: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()