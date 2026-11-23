"""Papildomas (neklinikinis) kolegų preparatų informacijos katalogas.

Tikslus 4 laukų atitikimas; dviprasmiai įrašai ignoruojami.
Niekada neteikia klinikinio sprendimo dėl vaistų sąveikų.
"""
import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

FIELDS = {'pavadinimas','stiprumas','forma','veiklioji','vartojimas','indikacijos',
          'kontraindikacijos','ispejimai','mechanizmas','vvkt_url','dokumento_url','duomenu_busena'}
MISSING = {'', '—', '-', 'nan', 'none', 'null', 'nenurodyta'}

def _norm(value):
    if value is None:
        return ''
    s = unicodedata.normalize('NFKD', str(value).casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', s).strip()

def _key(name, ingredient, strength, form):
    result = tuple(_norm(v) for v in (name, ingredient, strength, form))
    if any(v in MISSING for v in result):
        return None
    return result

@lru_cache(maxsize=4)
def load_info_catalog(path):
    """Nuskaityti tik teisingos struktūros CSV; nesėkmė -> tuščias katalogas."""
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as fh:
            rd=csv.DictReader(fh)
            if not FIELDS.issubset(rd.fieldnames or []):
                return ()
            return tuple(dict(row) for row in rd)
    except (OSError, UnicodeError, csv.Error):
        return ()

def match_product(catalog, name, ingredient, strength, form):
    """Vienareikšmis įrašas pagal VVKT išsirinktos pakuotės 4 požymius.

    Nenaudojame tik pavadinimo, fuzzy paieškos arba vien veikliosios medžiagos.
    """
    k = _key(name, ingredient, strength, form)
    if k is None:
        return None
    matches = [row for row in catalog if
               _key(row.get('pavadinimas'), row.get('veiklioji'),
                    row.get('stiprumas'), row.get('forma')) == k]
    if not matches:
        return None
    # Dublikatai priimtini tik tada, jei visos reikšmės vienodos.
    first=matches[0]
    if any(row != first for row in matches[1:]):
        return None
    return first
