class Team:
    def __init__(self, name: str, opening_elo: float, division: str):
        self.name = name
        self.elo = opening_elo
        self.division = division
        self.last_elo = opening_elo      # poslední známé elo
        self.last_seen_season = None   # ← přidáno
        self.active = True             # ← hraje ve sledované lize

    def __repr__(self):
        return f"Team(name={self.name}, elo={self.elo}, div={self.division})"

    def get_division(self) -> str:
        return self.division
    
    def set_division(self, new_div: str):
        self.division = new_div

    def get_elo(self) -> float:
        return self.elo
    
    def set_elo(self, new_elo: float):
        self.elo = new_elo
        self.last_elo = new_elo         # vždy uložíme poslední známé elo

    def mark_seen(self, season):
        self.last_seen_season = season
        self.active = True

    def mark_missing(self):
        self.active = False
        self.division = None
        self.last_elo = self.elo  # uložit poslední elo
        self.elo = None
        
    def reactivate(self, new_div, new_elo, season):
        self.active = True
        self.division = new_div
        self.elo = new_elo
        self.last_seen_season = season