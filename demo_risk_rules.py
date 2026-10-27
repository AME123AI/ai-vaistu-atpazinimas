"""Edukacinės demonstracinės taisyklės; NENAUDOTI medicininiams sprendimams."""
import re
from collections import defaultdict

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
        grouped[c['ingredient'].casefold()].append(c)
    for group in grouped.values():
        if len({c['name'] for c in group}) > 1:
            warnings.append(f'Galimas veikliosios medžiagos „{group[0]["ingredient"]}“ dubliavimas')
            if all(c['mg'] is not None for c in group):
                total = sum(c['mg'] for c in group)
                if total > LIMIT_DUPLICATE_MG:
                    warnings.append(f'Demonstracinė 500 mg riba viršyta: {group[0]["ingredient"]} – {total:g} mg')
    # Skirtingų veikliųjų medžiagų suma yra tik programinis demonstravimas,
    # mediciniškai skirtingų medžiagų mg sudėti negalima saugumui vertinti.
    if len(grouped) > 1 and not incomplete:
        included = [c for c in components if not c['excluded']]
        if included and all(c['mg'] is not None for c in included):
            total = sum(c['mg'] for c in included)
            if total > LIMIT_DIFFERENT_MG:
                warnings.append(f'Demonstracinė 700 mg suma viršyta: {total:g} mg (ne medicininė riba)')
    return warnings, notes, components

def evaluate(items):
    warnings, notes, _ = analyze(items)
    return warnings, notes
