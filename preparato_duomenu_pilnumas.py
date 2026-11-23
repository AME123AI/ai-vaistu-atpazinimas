"""Konservatyvus 441 eilutės testinių ištraukų prieinamumo rodiklis.

NĖRA farmacinės / klinikinės validacijos ir nereiškia vaisto saugumo.
"""
import csv
import re
import unicodedata
from pathlib import Path


def _norm(value):
    value = unicodedata.normalize('NFKD', str(value or '').casefold())
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r'\s+', ' ', value).strip()


def _present(value):
    return _norm(value) not in ('', 'nerasta', 'nan', 'none', '—', '-')


def load_catalog(path):
    """Return list of source rows. Missing/broken file yields [] (fail closed)."""
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            required = {'Pavadinimas', 'Veiklioji_medziaga', 'Stiprumas', 'Forma',
                        'Duomenys_rasti', '4.5_Saveika', 'Dokumento_URL'}
            if not required.issubset(reader.fieldnames or []):
                return []
            return list(reader)
    except (OSError, UnicodeError, csv.Error):
        return []


def check_drug(name, ingredient, strength, form, records):
    """Return conservative text-availability status for exactly selected product.

    A match requires product name, ingredient, strength and form when known.
    Conflicting batch rows count as insufficient. No attempt to validate source identity.
    """
    if not records or not _present(name) or not _present(ingredient) or not _present(strength) or not _present(form):
        return {'enough': False, 'reason': 'Nepakanka duomenų tiksliai preparato formai ir stiprumui nustatyti.'}
    target = (_norm(name), _norm(ingredient), _norm(strength), _norm(form))
    candidates = [r for r in records if
        (_norm(r.get('Pavadinimas')), _norm(r.get('Veiklioji_medziaga')),
         _norm(r.get('Stiprumas')), _norm(r.get('Forma'))) == target]
    if not candidates:
        return {'enough': False, 'reason': 'Šio tikslaus preparato įrašo bandomojoje dokumentų bazėje nerasta.'}
    available = all(_norm(r.get('Duomenys_rasti')) == 'taip' and
                    _present(r.get('4.5_Saveika')) and
                    _present(r.get('Dokumento_URL')) for r in candidates)
    if not available:
        return {'enough': False, 'reason': 'Vienoje ar keliose šio preparato eilutėse nėra ištraukto 4.5 skyriaus arba dokumento nuorodos.'}
    return {'enough': True, 'reason': 'Rastas bandomasis 4.5 skyriaus tekstas. Jo tikslumas ir preparato tapatybė dar nepatikrinti.'}
