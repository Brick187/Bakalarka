import pandas as pd
import matplotlib.pyplot as plt

# =========================
# NASTAVENÍ
# =========================
DATA_FILE = "matches_with_features.csv"

SEASON = "2018-19"
LEAGUE = "E0_elo"

FEATURE = "form_points_home"
# např:
# "shots_home"
# "goal_diff_home"
# "goals_scored_home"
# "home_form_home"

N_TEAMS = 10

SELECTED_TEAMS = ["Chelsea", "Liverpool", "Man City", "Arsenal",
                  "Cardiff", "Wolves", "Leicester", "Fulham"]

# =========================
# NAČTENÍ
# =========================
df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# filtrace
df = df[
    (df["Season"] == SEASON) &
    (df["Div"] == LEAGUE)
].copy()

df = df.sort_values("Date")

# =========================
# VÝBĚR TÝMŮ
# =========================
if SELECTED_TEAMS is None:

    # vezme poslední známou hodnotu feature
    last_values = (
        df.groupby("HomeTeam")
        .tail(1)
        .sort_values(FEATURE, ascending=False)
    )

    selected_teams = last_values["HomeTeam"].head(N_TEAMS).tolist()

else:
    selected_teams = SELECTED_TEAMS

# =========================
# GRAF
# =========================
plt.figure(figsize=(14, 8))

for team in selected_teams:

    team_df = df[df["HomeTeam"] == team].copy()

    plt.plot(
        team_df["Date"],
        team_df[FEATURE],
        linewidth=2,
        marker="o",
        markersize=3,
        label=team
    )

plt.title(
    f"Vývoj feature '{FEATURE}' během sezóny {SEASON}\nPremier League"
)

plt.xlabel("Datum")
plt.ylabel(FEATURE)

plt.grid(True, alpha=0.3)

plt.legend(
    title="Tým",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    f"{FEATURE}_{SEASON}_{LEAGUE}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()