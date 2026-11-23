"""Edukaciniai sąveikų pavyzdžiai; OCR / VVKT kodo nekeičia."""
import csv
from pathlib import Path
from demo_saveikos import normalize

DATA_FILE = Path(__file__).with_name('saltiniais_pagristos_15_poru.csv')
UNKNOWN = {'', '—', 'nenurodyta', 'nan', 'none'}


def find_documented_pairs(first, second, data_path=None):
    a, b = normalize(first), normalize(second)
    if a in UNKNOWN or b in UNKNOWN or a == b:
        return []
    path = Path(data_path) if data_path else DATA_FILE
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [row for row in csv.DictReader(f)
                if {a, b} == {normalize(row['veiklioji_1']), normalize(row['veiklioji_2'])}]


def evaluate_pair(first, second, data_path=None):
    """Grąžina informacinę išvadą; neprognozuoja nežinomų vaistų saugumo."""
    a, b = normalize(first), normalize(second)
    if a in UNKNOWN or b in UNKNOWN:
        return {'type': 'missing', 'records': []}
    if a == b:
        return {'type': 'duplicate', 'records': []}
    matched = find_documented_pairs(first, second, data_path)
    return {'type': 'documented' if matched else 'unknown', 'records': matched}


def show_pair_result(first, second):
    import streamlit as st
    result = evaluate_pair(first, second)
    st.markdown('### 🧩 Dviejų vaistų sąveikos įvertinimas')
    if result['type'] == 'missing':
        st.warning('Nepavyko nustatyti abiejų veikliųjų medžiagų. Patikrinkite pasirinktus preparatus.')
        return False
    if result['type'] == 'duplicate':
        st.warning('⚠️ **Dubliuojama veiklioji medžiaga.** Abu pasirinkti preparatai turi tą pačią veikliąją medžiagą. Vartojant abu gali būti viršyta rekomenduojama bendra dozė.')
        st.info('Svarbu patikrinti preparatų stiprumą, suvartojamų vienetų skaičių, vartojimo dažnumą ir preparato lapelį. Vien iš pakuotės stiprumo perdozavimo nustatyti negalima.')
        return True
    if result['type'] == 'unknown':
        st.info('**Šios poros mokomojoje bazėje neįvertinome.** Sąveika nei patvirtinta, nei paneigta; tai nereiškia, kad vartoti kartu saugu.')
        st.caption('Duomenų bazėje yra 15 konkrečių porų. Tai nėra visų vaistų saugumo patikros sistema.')
        return False
    for r in result['records']:
        category = r['kategorija']
        if category.startswith('Paprastai galima'):
            st.success('✅ **' + category + '**')
        elif 'Kontraindikuot' in category or 'Nevartoti' in category:
            st.error('⛔ **' + category + '**')
        else:
            st.warning('⚠️ **' + category + '**')
        st.markdown('**Kas žinoma:** ' + r['saveika'])
        st.write('**Kodėl:**', r['mechanizmas'])
        st.write('**Galimos pasekmės:**', r['galimos_pasekmes'])
        st.write('**Svarbios sąlygos:**', r['salygos'])
        url = r.get('saltinis_url', '')
        if url.startswith('https://'):
            st.markdown('**Šaltinis:** [Atidaryti medicininį šaltinį](' + url + ')')
    st.caption('Universiteto mokomasis prototipas: informacija apibendrinta iš viešų šaltinių; konkrečių preparatų PCS ir individualios sveikatos būklės nevertiname.')
    return True
