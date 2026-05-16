from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# =========================
# NASTAVENÍ
# =========================
DATA_FILE = Path("matches_with_features.csv")
OUTPUT_DIR = Path("results_ml")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

TARGET_COL = "Target"
TARGET_MAP = {"H": 0, "D": 1, "A": 2}
INV_TARGET_MAP = {0: "H", 1: "D", 2: "A"}
CLASS_NAMES = ["HomeWin", "Draw", "AwayWin"]

# PONECHÁNO PODLE PŮVODNÍHO SOUBORU
FEATURE_COLS = [

    # odds
    "OddsProbHome",
    "OddsProbDraw",
    "OddsProbAway",
    "OddsSpreadHome",
    "OddsSpreadDraw",
    "OddsSpreadAway",

    # forma
    "form_points_home",
    "form_points_away",
    "goals_scored_home",
    "goals_conceded_home",
    "goals_scored_away",
    "goals_conceded_away",

    # střely
    "shots_home",
    "shots_away",
    "shots_on_target_home",
    "shots_on_target_away",

    # elo
    "elo_home",
    "elo_away",
    "elo_diff",

    # kontext
    "is_home",
    "days_rest_home",
    "days_rest_away",

    # tabulka
    "table_points_home",
    "table_points_away",
    "table_points_diff",

    # bonus
    "home_form_home",
    "away_form_away",
    "goal_diff_home",
    "goal_diff_away",
    "streak_home",
    "streak_away",
]

# metoda 1: průběžné přeučování po týdnu
MIN_HISTORY_MATCHES = 200
WF_TRAIN_WINDOW_DAYS = 1460
VALIDATION_WEEKS = 4

# metoda 2: trénink na 4 sezónách, test po týdnu na další sezóně
STATIC_TRAIN_SEASONS = 4

BET_ODDS_COLS = {
    "H": "B365H",
    "D": "B365D",
    "A": "B365A",
}


# =========================
# METRIKY
# =========================
def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    y_cum = np.cumsum(y_onehot, axis=1)
    p_cum = np.cumsum(probs, axis=1)
    return float(np.mean(np.sum((p_cum - y_cum) ** 2, axis=1) / (n_classes - 1)))


def evaluate_prob_model(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, probs, labels=np.arange(probs.shape[1]))),
        "brier": multiclass_brier_score(y_true, probs),
        "rps": ranked_probability_score(y_true, probs),
    }


# =========================
# DATA
# =========================
def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Soubor {DATA_FILE} neexistuje.")

    df = pd.read_csv(DATA_FILE)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    missing = [c for c in FEATURE_COLS + [TARGET_COL, "Season", "Date"] if c not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce v datasetu: {missing}")

    df = df[df[TARGET_COL].isin(TARGET_MAP)].copy()
    df["TargetEncoded"] = df[TARGET_COL].map(TARGET_MAP)
    df = df.dropna(subset=["Season", "Date", "TargetEncoded"] + FEATURE_COLS).copy()

    if df.empty:
        raise ValueError("Po odfiltrování chybějících hodnot nezůstala žádná data.")

    df = df.sort_values(["Date", "Season"]).reset_index(drop=True)
    df["WeekStart"] = (df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")).dt.normalize()

    week_map = (
        df[["Season", "WeekStart"]]
        .drop_duplicates()
        .sort_values(["Season", "WeekStart"])
        .copy()
    )
    week_map["WeekIndex"] = week_map.groupby("Season").cumcount() + 1
    df = df.merge(week_map, on=["Season", "WeekStart"], how="left")

    return df


# =========================
# MODELY
# =========================
def fit_models(train_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["TargetEncoded"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model_lr = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model_lr.fit(X_train_scaled, y_train)

    model_gbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model_gbm.fit(X_train, y_train)

    calibrated_gbm = CalibratedClassifierCV(
        estimator=LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
        method="isotonic",
        cv=3,
    )
    calibrated_gbm.fit(X_train, y_train)
    model_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    )

    model_mlp.fit(X_train_scaled, y_train)

    return {
        "lr": {
            "model": model_lr,
            "scaler": scaler,
            "needs_scaling": True,
        },
        "gbm": {
            "model": model_gbm,
            "scaler": None,
            "needs_scaling": False,
        },
        "gbm_calibrated": {
            "model": calibrated_gbm,
            "scaler": None,
            "needs_scaling": False,
        },
        "mlp": {
            "model": model_mlp,
            "scaler": scaler,
            "needs_scaling": True,
        },
    }


def predict_probs(payload: Dict[str, object], eval_df: pd.DataFrame) -> np.ndarray:
    X = eval_df[FEATURE_COLS].to_numpy(dtype=float)
    if payload["needs_scaling"]:
        X = payload["scaler"].transform(X)
    probs = payload["model"].predict_proba(X)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


# =========================
# INTERPRETACE
# =========================
def logistic_coefficients_table(model, feature_names, class_names=None):
    if class_names is None:
        class_names = [f"class_{i}" for i in range(model.coef_.shape[0])]

    rows = []
    for class_idx, class_name in enumerate(class_names):
        for feat, coef in zip(feature_names, model.coef_[class_idx]):
            rows.append({
                "class": class_name,
                "feature": feat,
                "coefficient": coef,
                "abs_coefficient": abs(coef)
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["class", "abs_coefficient"], ascending=[True, False])


def plot_feature_importance(model, feature_names, top_n=20, output_path=None):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return importance


def reliability_curve_binary(y_true_binary, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    prob_means = []
    true_freqs = []
    counts = []

    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            prob_means.append(np.mean(y_prob[mask]))
            true_freqs.append(np.mean(y_true_binary[mask]))
            counts.append(np.sum(mask))

    return np.array(prob_means), np.array(true_freqs), np.array(counts)


def plot_reliability_diagram_multiclass(y_true, probs, class_names=None, n_bins=10, output_prefix=None):
    n_classes = probs.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)
        prob_mean, true_freq, _ = reliability_curve_binary(y_binary, probs[:, c], n_bins=n_bins)

        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(prob_mean, true_freq, marker="o")
        plt.xlabel("Průměrná predikovaná pravděpodobnost")
        plt.ylabel("Skutečná četnost")
        plt.title(f"Reliability diagram – {class_names[c]}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_prefix is not None:
            fname = f"{output_prefix}_class_{c}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()


# =========================
# SPLITY
# =========================
def split_past_into_train_valid(all_past: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_weeks = sorted(all_past["WeekStart"].unique())
    if len(unique_weeks) <= VALIDATION_WEEKS:
        return pd.DataFrame(), pd.DataFrame()

    valid_weeks = unique_weeks[-VALIDATION_WEEKS:]
    train_df = all_past[~all_past["WeekStart"].isin(valid_weeks)].copy()
    valid_df = all_past[all_past["WeekStart"].isin(valid_weeks)].copy()
    return train_df, valid_df

def build_match_predictions(
    eval_df: pd.DataFrame,
    probs: np.ndarray,
    method: str,
    model_name: str,
    season: str,
    week: int,
    week_start,
    train_matches: int,
    valid_matches: int,
    train_seasons: str | None = None,
    valid_seasons: str | None = None,
) -> pd.DataFrame:
    out = eval_df.copy()
    out["method"] = method
    out["model"] = model_name
    out["season"] = season
    out["week"] = int(week)
    out["week_start"] = pd.Timestamp(week_start).date().isoformat()
    out["train_matches"] = int(train_matches)
    out["valid_matches"] = int(valid_matches)
    if train_seasons is not None:
        out["train_seasons"] = train_seasons
    if valid_seasons is not None:
        out["valid_seasons"] = valid_seasons

    out["Prob_H"] = probs[:, 0]
    out["Prob_D"] = probs[:, 1]
    out["Prob_A"] = probs[:, 2]
    pred_idx = np.argmax(probs, axis=1)
    out["Predicted"] = [INV_TARGET_MAP[i] for i in pred_idx]
    out["PredictedProb"] = probs[np.arange(len(out)), pred_idx]
    out["Correct"] = (out["Predicted"] == out[TARGET_COL]).astype(int)

    for outcome, col in BET_ODDS_COLS.items():
        prob_col = f"Prob_{outcome}"
        implied_col = f"ImpliedProb_{outcome}"
        if col in out.columns:
            odds = pd.to_numeric(out[col], errors="coerce")
            out[implied_col] = np.where(odds > 0, 1.0 / odds, np.nan)
        else:
            out[implied_col] = np.nan

    def pick_row(row: pd.Series):
        pick = row["Predicted"]
        odds_col = BET_ODDS_COLS.get(pick)
        prob_col = f"Prob_{pick}"
        implied_col = f"ImpliedProb_{pick}"
        odds_val = row.get(odds_col, np.nan)
        model_prob = row.get(prob_col, np.nan)
        implied_prob = row.get(implied_col, np.nan)
        edge = model_prob - implied_prob if pd.notna(model_prob) and pd.notna(implied_prob) else np.nan
        return pd.Series({
            "BetOn": pick,
            "BetOdds": odds_val,
            "BetModelProb": model_prob,
            "BetImpliedProb": implied_prob,
            "BetEdge": edge,
        })

    bet_info = out.apply(pick_row, axis=1)
    out = pd.concat([out, bet_info], axis=1)

    preferred_order = [
        "method", "model", "season", "week", "week_start",
        "train_matches", "valid_matches", "train_seasons", "valid_seasons",
        "Date", "HomeTeam", "AwayTeam", "Div",
        TARGET_COL, "Predicted", "Correct",
        "Prob_H", "Prob_D", "Prob_A", "PredictedProb",
        "B365H", "B365D", "B365A",
        "BetOn", "BetOdds", "BetModelProb", "BetImpliedProb", "BetEdge",
    ]
    existing = [c for c in preferred_order if c in out.columns]
    others = [c for c in out.columns if c not in existing]
    return out[existing + others]

# =========================
# METODA 1
# průběžné přeučování po týdnu
# =========================
def run_weekly_walk_forward(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    results = []
    match_rows = []
    last_models = None

    for season in sorted(df["Season"].dropna().unique()):
        season_df = df[df["Season"] == season].sort_values(["WeekStart", "Date"]).copy()
        week_starts = sorted(season_df["WeekStart"].unique())

        for week_start in week_starts:
            test_df = season_df[season_df["WeekStart"] == week_start].copy()
            if test_df.empty:
                continue

            test_start_date = test_df["Date"].min()
            train_from = test_start_date - pd.Timedelta(days=WF_TRAIN_WINDOW_DAYS)

            all_past = df[
                (df["Date"] >= train_from) &
                (df["Date"] < test_start_date)
            ].copy()
            if len(all_past) < MIN_HISTORY_MATCHES:
                continue

            train_df, valid_df = split_past_into_train_valid(all_past)
            if train_df.empty or valid_df.empty:
                continue

            models = fit_models(train_df)
            last_models = models

            for model_name, payload in models.items():
                valid_probs = predict_probs(payload, valid_df)
                test_probs = predict_probs(payload, test_df)

                valid_metrics = evaluate_prob_model(valid_df["TargetEncoded"].to_numpy(dtype=int), valid_probs)
                test_metrics = evaluate_prob_model(test_df["TargetEncoded"].to_numpy(dtype=int), test_probs)
                week_idx = int(test_df["WeekIndex"].iloc[0])
                results.append({
                    "method": "weekly_walk_forward",
                    "model": model_name,
                    "season": season,
                    "week": week_idx,
                    "week_start": pd.Timestamp(week_start).date().isoformat(),
                    "train_matches": len(train_df),
                    "valid_matches": len(valid_df),
                    "test_matches": len(test_df),
                    "valid_accuracy": valid_metrics["accuracy"],
                    "valid_log_loss": valid_metrics["log_loss"],
                    "valid_brier": valid_metrics["brier"],
                    "valid_rps": valid_metrics["rps"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_log_loss": test_metrics["log_loss"],
                    "test_brier": test_metrics["brier"],
                    "test_rps": test_metrics["rps"],
                })
                match_rows.append(
                    build_match_predictions(
                        test_df,
                        test_probs,
                        method="weekly_walk_forward",
                        model_name=model_name,
                        season=season,
                        week=week_idx,
                        week_start=week_start,
                        train_matches=len(train_df),
                        valid_matches=len(valid_df),
                    )
                )
        print(f"Processed: Season {season}")

    return pd.DataFrame(results), pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame(), last_models


# =========================
# METODA 2
# train na 4 sezónách, test po týdnu na další sezóně
# =========================
def run_train4_test1_weekly(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    seasons = sorted(df["Season"].dropna().unique())
    results = []
    match_rows = []
    last_models = None

    needed = STATIC_TRAIN_SEASONS + 1
    if len(seasons) < needed:
        raise ValueError(f"Potřebuješ alespoň {needed} sezón pro metodu train-4-test-1.")

    for start_idx in range(0, len(seasons) - needed + 1):
        train_seasons = seasons[start_idx:start_idx + STATIC_TRAIN_SEASONS]
        test_season = seasons[start_idx + STATIC_TRAIN_SEASONS]

        train_df = df[df["Season"].isin(train_seasons)].copy()
        train_valid_source = df[df["Season"].isin(train_seasons)].sort_values(["Date"]).copy()
        train_df, valid_df = split_past_into_train_valid(train_valid_source)
        if train_df.empty or valid_df.empty:
            continue
        test_season_df = df[df["Season"] == test_season].sort_values(["WeekStart", "Date"]).copy()

        models = fit_models(train_df)
        last_models = models
        week_starts = sorted(test_season_df["WeekStart"].unique())
        train_seasons_str = ", ".join(train_seasons)
        for model_name, payload in models.items():
            valid_probs = predict_probs(payload, valid_df)
            valid_metrics = evaluate_prob_model(valid_df["TargetEncoded"].to_numpy(dtype=int), valid_probs)

            for week_start in week_starts:
                test_df = test_season_df[test_season_df["WeekStart"] == week_start].copy()
                if test_df.empty:
                    continue

                test_probs = predict_probs(payload, test_df)
                test_metrics = evaluate_prob_model(test_df["TargetEncoded"].to_numpy(dtype=int), test_probs)
                week_idx = int(test_df["WeekIndex"].iloc[0])
                results.append({
                    "method": "train_4seasons_test_1season_weekly",
                    "model": model_name,
                    "season": test_season,
                    "week": week_idx,
                    "week_start": pd.Timestamp(week_start).date().isoformat(),
                    "train_seasons": ", ".join(train_seasons),
                    "train_matches": len(train_df),
                    "valid_matches": len(valid_df),
                    "test_matches": len(test_df),
                    "valid_accuracy": valid_metrics["accuracy"],
                    "valid_log_loss": valid_metrics["log_loss"],
                    "valid_brier": valid_metrics["brier"],
                    "valid_rps": valid_metrics["rps"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_log_loss": test_metrics["log_loss"],
                    "test_brier": test_metrics["brier"],
                    "test_rps": test_metrics["rps"],
                })
                match_rows.append(
                    build_match_predictions(
                        test_df,
                        test_probs,
                        method="train_4seasons_test_1season_weekly",
                        model_name=model_name,
                        season=test_season,
                        week=week_idx,
                        week_start=week_start,
                        train_matches=len(train_df),
                        valid_matches=len(valid_df),
                        train_seasons=train_seasons_str
                    )
                )

    return pd.DataFrame(results), pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame(), last_models


# =========================
# SHRNUTÍ A ULOŽENÍ
# =========================
def save_summary(results_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    summary = (
        results_df.groupby(["method", "model"], as_index=False)
        .agg(
            weeks=("week", "count"),
            total_test_matches=("test_matches", "sum"),
            avg_valid_accuracy=("valid_accuracy", "mean"),
            avg_valid_log_loss=("valid_log_loss", "mean"),
            avg_valid_brier=("valid_brier", "mean"),
            avg_valid_rps=("valid_rps", "mean"),
            avg_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            avg_test_log_loss=("test_log_loss", "mean"),
            std_test_log_loss=("test_log_loss", "std"),
            avg_test_brier=("test_brier", "mean"),
            avg_test_rps=("test_rps", "mean"),
        )
        .sort_values(["method", "avg_test_log_loss", "avg_test_accuracy"], ascending=[True, True, False])
    )
    summary.to_csv(out_path, index=False)
    return summary


def save_predictions_for_last_week(df: pd.DataFrame, models: Dict[str, Dict[str, object]], prefix: str):
    if not models:
        return

    last_date = df["Date"].max()
    last_week_start = (last_date - pd.Timedelta(days=int(last_date.weekday()))).normalize()
    last_week_df = df[df["WeekStart"] == last_week_start].copy()
    if last_week_df.empty:
        return

    for model_name, payload in models.items():
        probs = predict_probs(payload, last_week_df)
        out = build_match_predictions(
            last_week_df,
            probs,
            method=prefix,
            model_name=model_name,
            season=str(last_week_df["Season"].iloc[0]),
            week=int(last_week_df["WeekIndex"].iloc[0]),
            week_start=last_week_start,
            train_matches=0,
            valid_matches=0,
        )
        out.to_csv(OUTPUT_DIR / f"{prefix}_predictions_{model_name}.csv", index=False)

def main():

    df = load_dataset()
    print(f"Načteno řádků po filtraci: {len(df)}")
    print(f"Použité feature: {len(FEATURE_COLS)}")

    wf_results, wf_matches, wf_models = run_weekly_walk_forward(df)
    static_results, static_matches, static_models = run_train4_test1_weekly(df)

    all_results = pd.concat([wf_results, static_results], ignore_index=True)
    if all_results.empty:
        raise ValueError("Nevznikly žádné výsledky. Zkontroluj množství dat a nastavení splitů.")

    all_match_predictions = pd.concat([wf_matches, static_matches], ignore_index=True)

    weekly_path = OUTPUT_DIR / "machine_learning_weekly_results.csv"
    summary_path = OUTPUT_DIR / "machine_learning_summary.csv"
    matches_path = OUTPUT_DIR / "machine_learning_match_predictions.csv"

    all_results.to_csv(weekly_path, index=False)
    summary_df = save_summary(all_results, summary_path)

    if not all_match_predictions.empty:
        all_match_predictions.to_csv(matches_path, index=False)
        print(f"Uloženo: {matches_path}")

    print("=== SOUHRN ===")
    print(summary_df)

    interp_models = static_models if static_models is not None else wf_models

    if interp_models is not None:
        coef_df = logistic_coefficients_table(
            interp_models["lr"]["model"],
            FEATURE_COLS,
            CLASS_NAMES,
        )
        coef_df.to_csv(OUTPUT_DIR / "logistic_coefficients.csv", index=False)

        fi_df = plot_feature_importance(
            interp_models["gbm"]["model"],
            FEATURE_COLS,
            top_n=20,
            output_path=OUTPUT_DIR / "gbm_feature_importance.png",
        )
        fi_df.to_csv(OUTPUT_DIR / "gbm_feature_importance.csv", index=False)


    print("Uloženo do složky:", OUTPUT_DIR.resolve())
    print("- machine_learning_weekly_results.csv")
    print("- machine_learning_summary.csv")
    print("- machine_learning_match_predictions.csv")
    print("- logistic_coefficients.csv")
    print("- gbm_feature_importance.csv")
    print("- gbm_feature_importance.png")


if __name__ == "__main__":
    main()