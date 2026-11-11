"""Tikslus demonstracinių porų palyginimas; ne klinikinis sprendimų įrankis."""
import csv
import re
import unicodedata
from pathlib import Path

ALIASES = {
    "acetilsalicilo rugstis": "acetilsalicilo rugstis",
    "acetylsalicylic acid": "acetilsalicilo rugstis",
    "warfarin": "varfarinas", "warfarinum": "varfarinas",
    "ibuprofen": "ibuprofenas", "ibuprofenum": "ibuprofenas",
    "simvastatin": "simvastatinas", "clarithromycin": "klaritromicinas",
    "sildenafil": "sildenafilis", "nitroglycerin": "nitroglicerinas",
    "glyceryl trinitrate": "nitroglicerinas",
    "methotrexate": "metotreksatas", "trimethoprim": "trimetoprimas",
    "spironolactone": "spironolaktonas", "potassium chloride": "kalio chloridas",
    "lithium": "litis", "lithium carbonate": "litis", "licio karbonatas": "litis",
    "digoxin": "digoksinas", "amiodarone": "amjodaronas", "amiodaronas": "amjodaronas",
    "levothyroxine": "levotiroksinas", "levotiroksino natris": "levotiroksinas",
    "calcium carbonate": "kalcio karbonatas", "clopidogrel": "klopidogrelis",
    "omeprazole": "omeprazolas",
}

def normalize(value):
    if not isinstance(value, str):
        return ""
    value = "".join(c for c in unicodedata.normalize("NFKD", value.lower()) if not unicodedata.combining(c))
    value = re.sub(r"\s+", " ", value).strip()
    return ALIASES.get(value, value)

def find_demo_pair(first, second, csv_path=None):
    """Only exact ingredient names; never infer a class or mark missing pairs safe."""
    a, b = normalize(first), normalize(second)
    if not a or not b or a in {'—','nenurodyta'} or b in {'—','nenurodyta'}:
        return None
    path = Path(csv_path) if csv_path is not None else Path(__file__).with_name('demo_porines_saveikos.csv')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if {a,b} == {normalize(row['veiklioji_1']),normalize(row['veiklioji_2'])}:
                return row
    return None
