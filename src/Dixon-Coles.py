import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# =========================
# NASTAVENÍ
# =========================
DATA_ROOT = Path("data/processed/")

SEASONS_FILTER = {}       # např. {"2018-19","2019-20"} nebo None
FILES_FILTER = {}          # např. {"E0.csv"} nebo None

MAX_GOALS = 7                 # ořez skóre matice
L2 = 1e-3                      # regularizace
MIN_MATCHES_TRAIN = 100        # minimální počet zápasů pro fit
WINDOW_DAYS = 365   # 365 = rok, 1825 = 5 let, None = celá historie

# =========================
# POMOCNÉ FUNKCE
# =========================
def log_poisson_pmf(k: int, lam: float) -> float:
    return -lam + k * math.log(lam) - gammaln(k + 1)

def tau_dc(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - rho * lam * mu
    if x == 0 and y == 1:
        return 1 + rho * lam
    if x == 1 and y == 0:
        return 1 + rho * mu
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0

# =========================
# MODEL
# =========================
class DixonColes:
    def __init__(self, teams: List[str]):
        self.teams = teams
        self.n = len(teams)
        self.idx = {t: i for i, t in enumerate(teams)}
        self.params = None

    def _unpack(self, x):
        home_adv = x[0]
        a = np.zeros(self.n)
        d = np.zeros(self.n)

        a[:-1] = x[1:1+self.n-1]
        d[:-1] = x[1+self.n-1:1+2*(self.n-1)]

        a[-1] = -a[:-1].sum()
        d[-1] = -d[:-1].sum()

        rho = 0.5 * math.tanh(x[-1])
        return home_adv, a, d, rho

    def fit(self, df: pd.DataFrame):
        h_idx = df["HomeTeam"].map(self.idx).to_numpy()
        a_idx = df["AwayTeam"].map(self.idx).to_numpy()
        hg = df["FTHG"].to_numpy()
        ag = df["FTAG"].to_numpy()

        x0 = np.zeros(1 + 2*(self.n-1) + 1)

        def nll(x):
            ha, att, defn, rho = self._unpack(x)
            ll = 0.0

            for i in range(len(df)):
                lam = math.exp(ha + att[h_idx[i]] + defn[a_idx[i]])
                mu  = math.exp(att[a_idx[i]] + defn[h_idx[i]])

                lam = min(max(lam, 0.05), 8.0)
                mu  = min(max(mu, 0.05), 8.0)

                tau = tau_dc(hg[i], ag[i], lam, mu, rho)
                if tau <= 0:
                    return 1e9

                ll -= (
                    math.log(tau)
                    + log_poisson_pmf(hg[i], lam)
                    + log_poisson_pmf(ag[i], mu)
                )

            penalty = L2 * (ha**2 + np.sum(att**2) + np.sum(defn**2))
            return ll + penalty

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 3000, "maxfun": 300000}
        )

        if not res.success:
            raise RuntimeError(res.message)

        self.params = self._unpack(res.x)

    def probs_1x2(self, home: str, away: str) -> Dict[str, float]:
        ha, att, defn, rho = self.params
        hi, ai = self.idx[home], self.idx[away]

        lam = math.exp(ha + att[hi] + defn[ai])
        mu  = math.exp(att[ai] + defn[hi])

        lam = min(max(lam, 0.05), 8.0)
        mu  = min(max(mu, 0.05), 8.0)

        px = np.exp([-lam + x*np.log(lam) - gammaln(x+1) for x in range(MAX_GOALS+1)])
        py = np.exp([-mu  + y*np.log(mu)  - gammaln(y+1) for y in range(MAX_GOALS+1)])

        mat = np.outer(px, py)

        mat[0,0] *= max(0, 1 - rho*lam*mu)
        mat[0,1] *= max(0, 1 + rho*lam)
        mat[1,0] *= max(0, 1 + rho*mu)
        mat[1,1] *= max(0, 1 - rho)

        mat /= mat.sum()

        pH = mat[np.tril_indices(MAX_GOALS+1, -1)].sum()
        pD = np.trace(mat)
        pA = mat[np.triu_indices(MAX_GOALS+1, 1)].sum()

        return {"H": pH, "D": pD, "A": pA}

# =========================
# DATA LOAD
# =========================
def load_all():
    rows = []
    for sdir in DATA_ROOT.iterdir():
        if not sdir.is_dir():
            continue
        if SEASONS_FILTER and sdir.name not in SEASONS_FILTER:
            continue

        for f in sdir.glob("*.csv"):
            if FILES_FILTER and f.name not in FILES_FILTER:
                continue

            df = pd.read_csv(f)
            if not {"Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"}.issubset(df.columns):
                continue
            df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
            df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
            df["FTR"] = df["FTR"].astype(str).str.strip()
            df = df[df["FTR"].isin(["H", "D", "A"])].copy()
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date"])
            df["Season"] = sdir.name
            df["File"] = f.name
            df = df[[
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
                "Season",
                "File"
            ]]
            rows.append(df)

    return pd.concat(rows, ignore_index=True)

# =========================
# WALK-FORWARD PO TÝDNECH
# =========================
def main():
    df = load_all()
    results = []

    # seskupit jen podle File (liga) = napříč sezónami
    for file, df_file in df.groupby("File", sort=False):
        df_file = df_file.sort_values("Date").reset_index(drop=True)

        # stejná definice týdne jako u dumb sázkaře: pondělí
        df_file["WeekStart"] = df_file["Date"].dt.to_period("W").astype(str)

        # vyhodnocuj po sezónách (round se resetuje každou sezónu)
        for season, df_season in df_file.groupby("Season", sort=False):
            df_season = df_season.sort_values("Date").reset_index(drop=True)

            # týdny v rámci sezóny (1..N)
            weeks = sorted(df_season["WeekStart"].unique())

            for r, ws in enumerate(weeks, start=1):
                test = df_season[df_season["WeekStart"] == ws]
                if test.empty:
                    continue

                test_start = test["Date"].min()

                # trénink ber z df_file (napříč sezónami), ale jen posledních WINDOW_DAYS
                if WINDOW_DAYS is None:
                    train = df_file[df_file["Date"] < test_start]
                else:
                    train_from = test_start - pd.Timedelta(days=WINDOW_DAYS)
                    train = df_file[(df_file["Date"] >= train_from) & (df_file["Date"] < test_start)]

                if len(train) < MIN_MATCHES_TRAIN:
                    continue

                # model a indexy dělej na všech týmech v df_file (aby nevznikaly KeyError)
                all_teams = sorted(set(df_file["HomeTeam"]) | set(df_file["AwayTeam"]))
                model = DixonColes(all_teams)

                try:
                    model.fit(train)
                except RuntimeError:
                    continue

                correct = 0
                logloss = 0.0
                brier_sum = 0.0
                rps_sum = 0.0
                used = 0

                for row in test.itertuples(index=False):
                    true = str(row.FTR).strip()
                    if true not in {"H", "D", "A"}:
                        continue

                    probs = model.probs_1x2(row.HomeTeam, row.AwayTeam)

                    p_home = probs["H"]
                    p_draw = probs["D"]
                    p_away = probs["A"]

                    pred = max(probs, key=probs.get)

                    used += 1
                    if pred == true:
                        correct += 1

                    # --- LOGLOSS ---
                    logloss -= math.log(max(probs[true], 1e-12))

                    # --- ONE HOT ---
                    o_home = 1 if true == "H" else 0
                    o_draw = 1 if true == "D" else 0
                    o_away = 1 if true == "A" else 0

                    # --- BRIER ---
                    brier = (
                        (p_home - o_home)**2 +
                        (p_draw - o_draw)**2 +
                        (p_away - o_away)**2
                    )
                    brier_sum += brier

                    # --- RPS ---
                    rps = 0.5 * (
                        (p_home - o_home)**2 +
                        (p_home + p_draw - o_home - o_draw)**2
                    )
                    rps_sum += rps
                if used == 0:
                    continue

                results.append({
                    "season": season,
                    "file": file,
                    "round": r,
                    "matches": used,
                    "correct": correct,
                    "accuracy": correct / used,
                    "logloss": logloss / used,
                    "brier": brier_sum / used,
                    "rps": rps_sum / used
                })

        print(f"OK: {file}")

    out = pd.DataFrame(results)
    out.to_csv("results_dixon_coles_weekly_S.csv", index=False)
    print("Uloženo: results_dixon_coles_weekly_S.csv")
    overall_acc = out["correct"].sum() / out["matches"].sum()
    overall_logloss = np.average(out["logloss"], weights=out["matches"])
    overall_brier = np.average(out["brier"], weights=out["matches"])
    overall_rps = np.average(out["rps"], weights=out["matches"])

    print("\n=== CELKOVÉ METRIKY ===")
    print(f"Accuracy: {overall_acc:.4f}")
    print(f"LogLoss:  {overall_logloss:.4f}")
    print(f"Brier:    {overall_brier:.4f}")
    print(f"RPS:      {overall_rps:.4f}")

if __name__ == "__main__":
    main()