import csv
from pathlib import Path

def fix_e1_file_exact_header(path: Path):
    print(f"\nKontroluji {path} ...")

    # Načti "syrově" celý CSV
    with open(path, newline="", encoding="latin-1", errors="replace") as f:
        rows = list(csv.reader(f))

    if not rows:
        print("Soubor je prázdný, přeskočeno.")
        return

    header = rows[0]
    header_len = len(header)

    print(f"  Hlavička má {header_len} sloupců.")

    fixed_rows = [header]   # hlavička zůstane stejná

    # Upravit každý další řádek
    for idx, r in enumerate(rows[1:], start=2):
        if len(r) == header_len:
            fixed_rows.append(r)
            continue

        if len(r) < header_len:
            # doplnit prázdné hodnoty
            new_r = r + [""] * (header_len - len(r))
            print(f"  → řádek {idx}: DOPLNĚN z {len(r)} na {header_len}")
        else:
            # oříznout nadbytečné hodnoty
            new_r = r[:header_len]
            print(f"  → řádek {idx}: OŘÍZNUT z {len(r)} na {header_len}")

        fixed_rows.append(new_r)

    # Zapsat zpět — přepisuje původní soubor
    with open(path, "w", newline="", encoding="latin-1") as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)

    print(" Opraveno – všechny řádky mají stejný počet sloupců jako hlavička.")


def fix_all_e1(root_dir: str):
    root = Path(root_dir)
    for file in root.rglob("*.csv"):
        fix_e1_file_exact_header(file)

fix_all_e1("data/")