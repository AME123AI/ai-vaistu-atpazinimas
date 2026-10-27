"""Tik demonstracinės, mediciniškai nepatvirtintos perspėjimų taisyklės."""
import re
from collections import defaultdict

LIMIT_DUPLICATE_MG = 500
LIMIT_DIFFERENT_MG = 700
EXCLUDED_FORMS = ("tepal", "purškal", "purskal", "gelis", "geliai", "krem", "tirpalas odai")


def _mg(value):
    """Grąžina mg tik kai visas stiprumas vienareikšmis (ne mg/ml ir pan.)."""
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*mg\s*", str(value or ""), re.I)
    return float(match.group(1).replace(",", ".")) if match else None


def evaluate(items):
    """items: [{name, ingredient, strength, form, quantity}]."""
    warnings, notes = [], []
    grouped = defaultdict(list)
    for item in items:
        ingredient = str(item.get("ingredient") or "").strip()
        if not ingredient:
            notes.append(f"{item.get('name', 'Preparatas')}: nežinoma veiklioji medžiaga")
            continue
        # Sudėtinių medžiagų automatiškai neskaidome: reikia patvirtintos sudėties.
        if any(sep in ingredient for sep in ("+", ";", "/")):
            notes.append(f"{item.get('name', 'Preparatas')}: sudėtinę veikliąją medžiagą būtina patikrinti")
            continue
        grouped[ingredient.casefold()].append(item)
    for key, group in grouped.items():
        if len(group) > 1:
            warnings.append(f"Galimas veikliosios medžiagos „{group[0]['ingredient']}“ dubliavimas")
        total = 0
        complete = True
        for item in group:
            form = str(item.get("form") or "").casefold()
            strength = _mg(item.get("strength"))
            quantity = item.get("quantity", 1)
            if any(x in form for x in EXCLUDED_FORMS):
                complete = False
                continue
            if not form or strength is None or not isinstance(quantity, int) or quantity < 1:
                complete = False
                notes.append(f"{item.get('name')}: nepakanka duomenų mg ribai apskaičiuoti")
                continue
            total += strength * quantity
        if complete and len(group) > 1 and total > LIMIT_DUPLICATE_MG:
            warnings.append(f"Demonstracinė 500 mg riba viršyta: {group[0]['ingredient']} – {total:g} mg")
    # Skirtingų veikliųjų medžiagų taisyklė tik kai visi duomenys žinomi.
    if len(grouped) > 1 and not notes:
        all_items = [i for group in grouped.values() for i in group]
        if all(not any(x in str(i.get('form') or '').casefold() for x in EXCLUDED_FORMS) for i in all_items):
            values = [_mg(i.get('strength')) for i in all_items]
            if all(v is not None and i.get('form') and isinstance(i.get('quantity', 1), int) for v, i in zip(values, all_items)):
                total = sum(v * i.get('quantity', 1) for v, i in zip(values, all_items))
                if total > LIMIT_DIFFERENT_MG:
                    warnings.append(f"Demonstracinė skirtingų medžiagų 700 mg suma viršyta: {total:g} mg")
    return warnings, notes
