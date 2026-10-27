"""Edukacinės demonstracinės taisyklės; NENAUDOTI medicininiams sprendimams."""
import re
from collections import defaultdict
import unicodedata

LIMIT_DUPLICATE_MG = 500
LIMIT_DIFFERENT_MG = 700
EXCLUDED_FORMS = ('tepal', 'purškal', 'purskal', 'gelis', 'geliai', 'krem', 'tirpalas odai')

def _parts(text):
    return [p.strip() for p in re.split(r'\s*[/;+]\s*', str(text or '')) if p.strip()]

def _strengths(text):
    parts = _parts(text)
    values = []
    for p in parts:
        m = re.fullmatch(r'(\d+(?:[.,]\d+)?)\s*mg', p, re.I)
        if not m:
            return None
        values.append(float(m.group(1).replace(',', '.')))
    return values

def _ingredient_key(value):
    """Tik aiškiai apibrėžtas sinonimų grupes laikome susijusiomis.

    Druskos forma ir bazinė medžiaga nėra automatiškai lygiaverčiai mg.
    """
    key = unicodedata.normalize('NFKD', str(value or '').casefold())
    key = ''.join(ch for ch in key if not unicodedata.combining(ch))
    key = re.sub(r'\s+', ' ', key).strip()
    if key in {'ambroksolis', 'ambroksolio hidrochloridas',
               'ambroxol', 'ambroxol hydrochloride', 'ambroxoli hydrochloridum'}:
        return 'ambroksolis'
    return key


def _is_ambroxol_salt(value):
    key = _ingredient_key(value)
    return key == 'ambroksolis' and any(
        term in unicodedata.normalize('NFKD', str(value).casefold())
        for term in ('hidrochlorid', 'hydrochlorid')
    )


def analyze(items):
    """Grąžina (perspėjimai, pastabos, išskaidytos sudedamosios dalys)."""
    warnings, notes, components = [], [], []
    incomplete = False
    for item in items:
        name = str(item.get('name') or 'Preparatas')
        ingredients = _parts(item.get('ingredient'))
        amounts = _strengths(item.get('strength'))
        form = str(item.get('form') or '').casefold()
        qty = item.get('quantity', 1)
        excluded = any(s in form for s in EXCLUDED_FORMS)
        if not ingredients or not form or not isinstance(qty, int) or qty < 1:
            incomplete = True
            notes.append(f'{name}: nepakanka patvirtintų duomenų skaičiavimui')
            for ing in ingredients:
                components.append({'name':name,'ingredient':ing,'mg':None,'excluded':excluded})
            continue
        if amounts is None or len(amounts) != len(ingredients):
            incomplete = True
            notes.append(f'{name}: negalima patikimai susieti kiekvienos veikliosios medžiagos su stiprumu')
            for ing in ingredients:
                components.append({'name':name,'ingredient':ing,'mg':None,'excluded':excluded})
            continue
        for ing, amount in zip(ingredients, amounts):
            components.append({'name':name,'ingredient':ing,'mg':amount*qty if not excluded else None,'excluded':excluded})
        if excluded:
            notes.append(f'{name}: šiai farmacinei formai mg demonstracinės ribos netaikomos')
    grouped = defaultdict(list)
    for c in components:
        grouped[_ingredient_key(c['ingredient'])].append(c)
    for group in grouped.values():
        if len({c['name'] for c in group}) > 1:
            warnings.append(f'Galimas veikliosios medžiagos „{group[0]["ingredient"]}“ dubliavimas')
            # Skirtingų druskos / bazinės medžiagos formų mg nesumuojame.
            if len({_is_ambroxol_salt(c['ingredient']) for c in group}) > 1:
                notes.append('Ambroksolio formų mg kiekiai automatiškai nesulyginami; būtina patikrinti VVKT dokumentus')
            elif all(c['mg'] is not None for c in group):
                total = sum(c['mg'] for c in group)
                # Be medžiagai specifinių patvirtintų ribų mg sumos nevertiname kaip rizikos.
    return warnings, notes, components

def evaluate(items):
    warnings, notes, _ = analyze(items)
    return warnings, notes
