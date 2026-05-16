import csv
from pathlib import Path
import shutil

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

for season_dir in RAW_DIR.iterdir():

    if not season_dir.is_dir():
        continue

    out_season = PROCESSED_DIR / season_dir.name
    out_season.mkdir(parents=True, exist_ok=True)

    for csv_file in season_dir.glob("*.csv"):

        output_path = out_season / csv_file.name

        # kopie raw -> processed
        shutil.copy2(csv_file, output_path)

        print(f"Kopíruji: {csv_file} -> {output_path}")

def add_elo_headers(csv_path):
    p = Path(csv_path)

    try:
        with open(p, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
    except UnicodeDecodeError:
        with open(p, newline='', encoding='cp1250') as f:
            rows = list(csv.reader(f))

    # Pokud není žádný řádek, ukonči
    if not rows:
        print(f"{p.name}: prázdný soubor, přeskočeno.")
        return

    # První řádek je hlavička
    header = rows[0]

    # Přidej nové hlavičky, pokud už tam nejsou
    if "EloHome" not in header:
        header.append("EloHome")
    if "EloAway" not in header:
        header.append("EloAway")

    # Uprav hlavičku v první řádce
    rows[0] = header

    # Přidej prázdné hodnoty na konec každého dalšího řádku
    for i in range(1, len(rows)):
        # Zajisti, že každý řádek má stejný počet sloupců jako hlavička
        while len(rows[i]) < len(header):
            rows[i].append("")

    # Ulož zpět (přepíše původní soubor)
    with open(p, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Přidány sloupce EloHome a EloAway do {p.name}")

def process_all_seasons(root_dir):
    """Projdi všechny sezóny a všechny ligy v zadaném adresáři."""
    root = Path(root_dir)
    if not root.exists():
        print("Cesta neexistuje:", root)
        return

    for season_dir in sorted(root.iterdir()):
        if not season_dir.is_dir():
            continue
        print(f"\n Zpracovávám sezónu: {season_dir.name}")
        for csv_file in sorted(season_dir.glob("*.csv")):
            add_elo_headers(csv_file)

process_all_seasons("data/processed")
