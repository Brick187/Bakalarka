import pandas as pd
from pathlib import Path
import numpy as np

HOME_ADV = 100
DRAW_BASE = 0.28
DRAW_SCALE = 300

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    

def elo_probabilities(eh: float, ea: float):
    diff = (eh + HOME_ADV) - ea

    # základní Elo pravděpodobnost domácí výhry vs. prohry
    p_home_win = 1.0 / (1.0 + 10 ** (-diff / 400.0))

    # remíza vyšší u vyrovnaných týmů, nižší u velkých rozdílů
    p_draw = DRAW_BASE * np.exp(-abs(diff) / DRAW_SCALE)

    # zbytek rozděl mezi H/A
    p_home = (1.0 - p_draw) * p_home_win
    p_away = (1.0 - p_draw) * (1.0 - p_home_win)

    # jistota, že součet je 1
    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total

    return p_home, p_draw, p_away

def probs_from_row(row):
        try:
            eh = float(row["EloHome"])
            ea = float(row["EloAway"])
        except Exception:
            return None, None, None
        return elo_probabilities(eh, ea)

def match_metrics(true_result: str, p_home: float, p_draw: float, p_away: float):
    o_home = 1 if true_result == "H" else 0
    o_draw = 1 if true_result == "D" else 0
    o_away = 1 if true_result == "A" else 0

    # Brier
    brier = (
        (p_home - o_home) ** 2 +
        (p_draw - o_draw) ** 2 +
        (p_away - o_away) ** 2
    )

    # LogLoss
    eps = 1e-15
    p_home_c = np.clip(p_home, eps, 1 - eps)
    p_draw_c = np.clip(p_draw, eps, 1 - eps)
    p_away_c = np.clip(p_away, eps, 1 - eps)

    if true_result == "H":
        logloss = -np.log(p_home_c)
    elif true_result == "D":
        logloss = -np.log(p_draw_c)
    else:
        logloss = -np.log(p_away_c)

    # RPS
    rps = 0.5 * (
        (p_home - o_home) ** 2 +
        (p_home + p_draw - o_home - o_draw) ** 2
    )

    return brier, logloss, rps

def predict_strict_draw(row) -> str | None:
    try:
        eh = float(row["EloHome"])
        ea = float(row["EloAway"])
    except Exception:
        return None
    

    diff = (eh + HOME_ADV) - ea
    if diff > 0:
        return "H"
    elif diff == 0:
        return "D"
    else:
        return "A"

def predict_threshold_draw(row, threshold=10):
    try:
        eh = float(row["EloHome"])
        ea = float(row["EloAway"])
    except Exception:
        return None

    diff = (eh + HOME_ADV) - ea
    if abs(diff) <= threshold:
        return "D"
    return "H" if diff > 0 else "A"

def week_start_monday(dt: pd.Timestamp) -> pd.Timestamp:
    # pondělí daného týdne
    return (dt - pd.Timedelta(days=int(dt.weekday()))).normalize()

def evaluate_file_by_week_roundstyle(path: Path, predictor):
    df = safe_read_csv(path)

    needed = {"Date", "FTR", "EloHome", "EloAway"}
    if not needed.issubset(df.columns):
        print(f"{path}: chybí sloupce {needed - set(df.columns)}")
        return None

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        print(f"{path}: žádná validní data.")
        return None

    # tvrdá predikce
    df["Pred"] = df.apply(predictor, axis=1)
    df["True"] = df["FTR"].astype(str).str.strip()

    # pravděpodobnosti
    probs = df.apply(probs_from_row, axis=1, result_type="expand")
    probs.columns = ["p_home", "p_draw", "p_away"]
    df = pd.concat([df, probs], axis=1)

    df = df[
        df["Pred"].notna()
        & df["True"].isin(["H", "D", "A"])
        & df["p_home"].notna()
        & df["p_draw"].notna()
        & df["p_away"].notna()
    ].copy()

    if df.empty:
        print(f"{path}: žádné použitelné zápasy.")
        return None

    df["WeekStart"] = df["Date"].apply(week_start_monday)

    week_order = (
        df[["WeekStart"]]
        .drop_duplicates()
        .sort_values("WeekStart")
        .reset_index(drop=True)
    )
    week_order["round"] = week_order.index + 1
    df = df.merge(week_order, on="WeekStart", how="left")

    df["correct"] = (df["Pred"] == df["True"]).astype(int)

    metric_vals = df.apply(
        lambda row: match_metrics(row["True"], row["p_home"], row["p_draw"], row["p_away"]),
        axis=1,
        result_type="expand"
    )
    metric_vals.columns = ["brier", "logloss", "rps"]
    df = pd.concat([df, metric_vals], axis=1)

    out = (
        df.groupby("round", as_index=False)
          .agg(
              matches=("correct", "count"),
              correct=("correct", "sum"),
              brier=("brier", "mean"),
              logloss=("logloss", "mean"),
              rps=("rps", "mean"),
          )
    )
    out["accuracy"] = (out["correct"] / out["matches"]).round(6)

    out.insert(0, "file", path.name)
    out.insert(0, "season", path.parent.name)

    return out

def evaluate_tree_by_week_roundstyle(root_dir: str, predictor, out_csv: str):
    root = Path(root_dir)
    all_rows = []

    for csv_path in sorted(root.rglob("*.csv")):
        res = evaluate_file_by_week_roundstyle(csv_path, predictor)
        if res is not None and not res.empty:
            all_rows.append(res)

    if not all_rows:
        print("Nenašel jsem žádná použitelná data.")
        return None

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    overall_accuracy = df["correct"].sum() / df["matches"].sum()
    overall_brier = np.average(df["brier"], weights=df["matches"])
    overall_logloss = np.average(df["logloss"], weights=df["matches"])
    overall_rps = np.average(df["rps"], weights=df["matches"])

    print(f"Uloženo: {out_csv}")
    print(f"Celková accuracy: {overall_accuracy:.4f}")
    print(f"Celkový Brier score: {overall_brier:.4f}")
    print(f"Celkový LogLoss: {overall_logloss:.4f}")
    print(f"Celkový RPS: {overall_rps:.4f}")

    return df

if __name__ == "__main__":
    evaluate_tree_by_week_roundstyle("data/processed/", predict_strict_draw, "results/elo_week_roundstyle_strict.csv")
    evaluate_tree_by_week_roundstyle("data/processed/", predict_threshold_draw, "results/elo_week_roundstyle_threshold.csv")