from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTCOME_TO_INDEX = {"H": 0, "D": 1, "A": 2}
VALID_OUTCOMES = set(OUTCOME_TO_INDEX.keys())

OUT_DIR = Path("results_betting")
OUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = Path("results_ml/machine_learning_match_predictions.csv")

INITIAL_BANKROLL = 100000.0
FIXED_STAKE = 1000.0
KELLY_FRACTION_COEF = 0.25
KELLY_MAX_FRACTION = 0.05
SELECTIONS = ["all", "top3", "top5"]

FILTER_SEASON = "2024-25"
FILTER_METHOD = None

REQUIRED_COLS = [
    "method", "model", "season", "week", "Date", "HomeTeam", "AwayTeam",
    "Target", "BetOn", "BetOdds", "BetModelProb"
]
OPTIONAL_COLS = [
    "week_start", "Div", "Predicted", "Correct", "Prob_H", "Prob_D", "Prob_A",
    "PredictedProb", "B365H", "B365D", "B365A", "BetImpliedProb", "BetEdge",
    "WeekStart", "WeekIndex", "train_matches", "valid_matches", "train_seasons"
]

def slugify(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")

def expected_value(prob: float, odds: float) -> float:
    return prob * odds - 1.0


def kelly_fraction(prob: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * odds - 1.0) / b
    return max(0.0, f)

def settle_bet_row(row: pd.Series, stake: float) -> float:
    predicted = str(row["BetOn"]).strip()
    actual = str(row["Target"]).strip()
    odds = float(row["BetOdds"])

    if predicted == actual:
        return stake * (odds - 1.0)
    return -stake

def max_drawdown(bankroll_series) -> float:
    bankroll = np.asarray(bankroll_series, dtype=float)
    if len(bankroll) == 0:
        return np.nan
    running_max = np.maximum.accumulate(bankroll)
    running_max[running_max == 0] = np.nan
    drawdowns = (running_max - bankroll) / running_max
    return np.nanmax(drawdowns)

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {path.resolve()}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")

    keep_cols = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]
    df = df[keep_cols].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BetOdds"] = pd.to_numeric(df["BetOdds"], errors="coerce")
    df["BetModelProb"] = pd.to_numeric(df["BetModelProb"], errors="coerce")

    if "BetImpliedProb" in df.columns:
        df["BetImpliedProb"] = pd.to_numeric(df["BetImpliedProb"], errors="coerce")
    else:
        df["BetImpliedProb"] = 1.0 / df["BetOdds"]

    if "BetEdge" in df.columns:
        df["BetEdge"] = pd.to_numeric(df["BetEdge"], errors="coerce")
    else:
        df["BetEdge"] = df["BetModelProb"] - df["BetImpliedProb"]

    df = df.dropna(subset=["Date", "BetOdds", "BetModelProb"]).copy()
    df = df[df["Target"].astype(str).str.strip().isin(VALID_OUTCOMES)].copy()
    df = df[df["BetOn"].astype(str).str.strip().isin(VALID_OUTCOMES)].copy()
    df = df[np.isfinite(df["BetOdds"]) & np.isfinite(df["BetModelProb"])].copy()
    df = df[df["BetOdds"] > 1.0].copy()
    df = df[df["BetModelProb"] > 0.0].copy()

    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["week"]).copy()
    df["week"] = df["week"].astype(int)

    df["sort_week_start"] = pd.to_datetime(
        df["week_start"] if "week_start" in df.columns else df.get("WeekStart", pd.NaT),
        errors="coerce"
    )

    df = df.sort_values(["method", "model", "season", "week", "sort_week_start", "Date", "BetModelProb"],
                        ascending=[True, True, True, True, True, True, False]).reset_index(drop=True)
    return df

def select_bets_for_group(group: pd.DataFrame, selection: str) -> pd.DataFrame:
    group = group.sort_values(["BetModelProb", "BetEdge", "Date"], ascending=[False, False, True]).copy()
    if selection == "all":
        return group
    if selection == "top3":
        return group.head(3)
    if selection == "top5":
        return group.head(5)
    raise ValueError(f"Neznámá selection: {selection}")

def group_selections(df: pd.DataFrame, selection: str) -> pd.DataFrame:
    grouped = []
    for _, grp in df.groupby(["method", "model", "season", "week"], sort=False):
        picked = select_bets_for_group(grp, selection)
        if not picked.empty:
            grouped.append(picked)
    if not grouped:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(grouped, ignore_index=True)

def simulate_fixed_stake(selected_df: pd.DataFrame, stake: float, initial_bankroll: float):
    bankroll = float(initial_bankroll)
    bet_rows = []
    round_rows = []

    grouped = selected_df.groupby(["method", "model", "season", "week"], sort=False)

    for (method, model, season, week), grp in grouped:
        grp = grp.sort_values(["Date", "BetModelProb"], ascending=[True, False]).copy()
        bankroll_before = bankroll
        round_profit = 0.0
        round_staked = 0.0
        round_bets = 0

        for _, row in grp.iterrows():
            profit = settle_bet_row(row, stake)
            bankroll += profit
            round_profit += profit
            round_staked += stake
            round_bets += 1

            bet_rows.append({
                "method": method,
                "model": model,
                "strategy": "fixed_stake",
                "selection": grp["selection"].iloc[0],
                "season": season,
                "week": int(week),
                "Date": row["Date"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "Div": row["Div"] if "Div" in row.index else pd.NA,
                "Target": row["Target"],
                "BetOn": row["BetOn"],
                "BetOdds": row["BetOdds"],
                "BetModelProb": row["BetModelProb"],
                "BetImpliedProb": row["BetImpliedProb"],
                "BetEdge": row["BetEdge"],
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
            })

        round_rows.append({
            "method": method,
            "model": model,
            "strategy": "fixed_stake",
            "selection": grp["selection"].iloc[0],
            "season": season,
            "week": int(week),
            "round_start": grp["sort_week_start"].dropna().iloc[0] if grp["sort_week_start"].notna().any() else grp["Date"].min(),
            "n_bets": round_bets,
            "round_staked": round_staked,
            "round_profit": round_profit,
            "bankroll_before": bankroll_before,
            "bankroll_after": bankroll,
        })

    return pd.DataFrame(bet_rows), pd.DataFrame(round_rows)


def simulate_fractional_kelly(selected_df: pd.DataFrame, initial_bankroll: float, kelly_fraction_coef: float, max_fraction: float):
    bankroll = float(initial_bankroll)
    bet_rows = []
    round_rows = []

    grouped = selected_df.groupby(["method", "model", "season", "week"], sort=False)

    for (method, model, season, week), grp in grouped:
        grp = grp.sort_values(["Date", "BetModelProb"], ascending=[True, False]).copy()
        bankroll_before = bankroll
        round_profit = 0.0
        round_staked = 0.0
        round_bets = 0

        for _, row in grp.iterrows():
            prob = float(row["BetModelProb"])
            odds = float(row["BetOdds"])
            full_kelly = kelly_fraction(prob, odds)
            frac_kelly = min(kelly_fraction_coef * full_kelly, max_fraction)
            stake = bankroll * frac_kelly

            if stake <= 0:
                continue

            profit = settle_bet_row(row, stake)
            bankroll += profit
            round_profit += profit
            round_staked += stake
            round_bets += 1

            bet_rows.append({
                "method": method,
                "model": model,
                "strategy": "fractional_kelly",
                "selection": grp["selection"].iloc[0],
                "season": season,
                "week": int(week),
                "Date": row["Date"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "Div": row["Div"] if "Div" in row.index else pd.NA,
                "Target": row["Target"],
                "BetOn": row["BetOn"],
                "BetOdds": row["BetOdds"],
                "BetModelProb": row["BetModelProb"],
                "BetImpliedProb": row["BetImpliedProb"],
                "BetEdge": row["BetEdge"],
                "full_kelly": full_kelly,
                "frac_kelly": frac_kelly,
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
            })

        round_rows.append({
            "method": method,
            "model": model,
            "strategy": "fractional_kelly",
            "selection": grp["selection"].iloc[0],
            "season": season,
            "week": int(week),
            "round_start": grp["sort_week_start"].dropna().iloc[0] if grp["sort_week_start"].notna().any() else grp["Date"].min(),
            "n_bets": round_bets,
            "round_staked": round_staked,
            "round_profit": round_profit,
            "bankroll_before": bankroll_before,
            "bankroll_after": bankroll,
        })

    return pd.DataFrame(bet_rows), pd.DataFrame(round_rows)

def compute_betting_metrics(bets_df: pd.DataFrame, rounds_df: pd.DataFrame) -> dict:
    total_profit = bets_df["profit"].sum() if not bets_df.empty else 0.0
    total_staked = bets_df["stake"].sum() if not bets_df.empty else 0.0
    n_bets = int(len(bets_df))

    roi = total_profit / total_staked if total_staked > 0 else np.nan

    if not bets_df.empty and (bets_df["stake"] > 0).any():
        bet_returns = (bets_df["profit"] / bets_df["stake"]).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        bet_returns = pd.Series(dtype=float)

    volatility = bet_returns.std(ddof=1) if len(bet_returns) > 1 else np.nan
    sharpe = bet_returns.mean() / volatility if len(bet_returns) > 1 and volatility > 0 else np.nan
    win_rate = (bets_df["profit"] > 0).mean() if n_bets > 0 else np.nan

    bankroll_series = rounds_df["bankroll_after"] if not rounds_df.empty else pd.Series(dtype=float)
    mdd = max_drawdown(bankroll_series.values) if len(bankroll_series) > 0 else np.nan
    final_bankroll = bankroll_series.iloc[-1] if len(bankroll_series) > 0 else np.nan

    return {
        "n_bets": n_bets,
        "n_rounds": int(len(rounds_df)),
        "total_profit": total_profit,
        "total_staked": total_staked,
        "roi": roi,
        "volatility": volatility,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "final_bankroll": final_bankroll,
    }

def plot_bankroll_curves_by_round(rounds_df: pd.DataFrame, title: str, output_path: Path):
    if rounds_df.empty:
        return

    plt.figure(figsize=(12, 7))

    for (method, model), grp in rounds_df.groupby(["method", "model"], sort=False):
        grp = grp.sort_values(["round_start", "season", "week"]).reset_index(drop=True)

        # počáteční bod před 1. kolem
        y = np.concatenate([[INITIAL_BANKROLL], grp["bankroll_after"].values])
        x = np.arange(0, len(grp) + 1)

        label = f"{method} | {model}"
        plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Round (week)")
    plt.ylabel("Bankroll")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    #plt.savefig(output_path, dpi=150, format="eps", bbox_inches="tight")
    plt.close()


def main():
    df = load_predictions(INPUT_FILE)
    print(f"Načteno sázek/predikcí před filtrem: {len(df)}")

    if FILTER_SEASON is not None:
        df = df[df["season"].astype(str) == str(FILTER_SEASON)].copy()
    if FILTER_METHOD is not None:
        df = df[df["method"].astype(str) == str(FILTER_METHOD)].copy()

    print(f"Po filtru season={FILTER_SEASON!r}, method={FILTER_METHOD!r}: {len(df)}")
    if df.empty:
        raise ValueError(
            f"Po filtrování nezůstala žádná data pro season={FILTER_SEASON!r} a method={FILTER_METHOD!r}."
        )

    metrics_rows = []
    all_bets = []
    all_rounds = []

    for selection in SELECTIONS:
        selected = group_selections(df, selection)
        if selected.empty:
            print(f"Selection {selection}: žádné vybrané sázky.")
            continue

        selected = selected.copy()
        selected["selection"] = selection

        for (method, model), grp in selected.groupby(["method", "model"], sort=False):
            grp = grp.sort_values(["sort_week_start", "Date", "season", "week"]).reset_index(drop=True)

            fixed_bets, fixed_rounds = simulate_fixed_stake(
                grp,
                stake=FIXED_STAKE,
                initial_bankroll=INITIAL_BANKROLL,
            )
            kelly_bets, kelly_rounds = simulate_fractional_kelly(
                grp,
                initial_bankroll=INITIAL_BANKROLL,
                kelly_fraction_coef=KELLY_FRACTION_COEF,
                max_fraction=KELLY_MAX_FRACTION,
            )
            fixed_metrics = compute_betting_metrics(fixed_bets, fixed_rounds)
            kelly_metrics = compute_betting_metrics(kelly_bets, kelly_rounds)

            metrics_rows.append({
                "method": method,
                "model": model,
                "strategy": "fixed_stake",
                "selection": selection,
                **fixed_metrics,
            })
            metrics_rows.append({
                "method": method,
                "model": model,
                "strategy": "fractional_kelly",
                "selection": selection,
                **kelly_metrics,
            })

            all_bets.append(fixed_bets)
            all_bets.append(kelly_bets)
            all_rounds.append(fixed_rounds)
            all_rounds.append(kelly_rounds)

    if not metrics_rows:
        print("Nevznikly žádné výsledky.")
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["strategy", "selection", "roi"],
        ascending=[True, True, False],
    )
    metrics_df.to_csv(OUT_DIR / "betting_metrics.csv", index=False)

    bets_df = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    rounds_df = pd.concat(all_rounds, ignore_index=True) if all_rounds else pd.DataFrame()

    if not bets_df.empty:
        bets_df.to_csv(OUT_DIR / "all_bets.csv", index=False)
    if not rounds_df.empty:
        rounds_df.to_csv(OUT_DIR / "all_round_histories.csv", index=False)

        for strategy in ["fixed_stake", "fractional_kelly"]:
            for selection in SELECTIONS:
                subset = rounds_df[(rounds_df["strategy"] == strategy) & (rounds_df["selection"] == selection)].copy()
                if subset.empty:
                    continue
                plot_bankroll_curves_by_round(
                    subset,
                    title=f"Bankroll progress after rounds – {strategy} – {selection}",
                    output_path=OUT_DIR / f"bankroll_rounds_{strategy}_{selection}.png",
                )

    print("\n=== BETTING SUMMARY ===")
    print(metrics_df)
    print(f"\nUloženo do: {OUT_DIR.resolve()}")
    print("- betting_metrics.csv")
    print("- all_bets.csv")
    print("- all_round_histories.csv")
    print("- bets_fixed_<method>_<model>_<selection>.csv")
    print("- bets_kelly_<method>_<model>_<selection>.csv")
    print("- history_fixed_<method>_<model>_<selection>.csv")
    print("- history_kelly_<method>_<model>_<selection>.csv")
    print("- bankroll_rounds_<strategy>_<selection>.png")


if __name__ == "__main__":
    main()