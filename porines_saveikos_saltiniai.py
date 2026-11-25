"""Platus edukacinis sąveikų modulis. Keičia TIK sąveikos pateikimą.

Suderinamas su esamu app.py: show_pair_result(first_ingredient, second_ingredient).
Jokie nežinomi deriniai nepriskiriami saugiems; grupių taisyklės aiškiai ženklinamos.
"""
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'saltiniais_pagristos_placios_poros.csv'
FALLBACK = ROOT / 'saltiniais_pagristos_15_poru.csv'


def norm(value):
    s = unicodedata.normalize('NFKD', str(value or '').strip().lower())
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r'\s+', ' ', s).strip()

# Forma „hidrochloridas“, „hidrobromidas“ neturi trukdyti medžiagų paieškai.
ALIASES = {
    'ibuprofenas': ('ibuprofenas', 'ibuprofen'),
    'paracetamolis': ('paracetamolis', 'paracetamol', 'acetaminophen'),
    'pseudoefedrinas': ('pseudoefedrinas','pseudoefedrino hidrochloridas','pseudoephedrine'),
    'dekstrometorfanas': ('dekstrometorfanas','dekstrometorfano hidrobromidas','dextromethorphan'),
    'acetilsalicilo rūgštis': ('acetilsalicilo rugstis','acetilsalicilo rūgštis','aspirinas','acetylsalicylic acid'),
    'naproksenas': ('naproksenas', 'naproxen'),
    'diklofenakas': ('diklofenakas','diklofenako', 'diklofenako natrio druska','diclofenac', 'diclofenacum natricum'),
    'deksketoprofenas': ('deksketoprofenas','dexketoprofen'),
    'ketoprofenas': ('ketoprofenas','ketoprofen'),
    'varfarinas': ('varfarinas','warfarin'),
    'prednizolonas': ('prednizolonas','prednisolone'),
    'sertralinas': ('sertralinas','sertraline'),
    'klaritromicinas': ('klaritromicinas','clarithromycin'),
    'simvastatinas': ('simvastatinas','simvastatin'),
    'lovastatinas': ('lovastatinas','lovastatin'),
    'levotiroksinas': ('levotiroksinas','levothyroxine'),
    'kalcio karbonatas': ('kalcio karbonatas','calcium carbonate'),
    'sildenafilis': ('sildenafilis','sildenafil'),
    'nitroglicerinas': ('nitroglicerinas','nitroglycerin','glicerilio trinitratas'),
    'digoksinas': ('digoksinas','digoxin'),
    'amjodaronas': ('amjodaronas','amiodarone'),
    'metotreksatas': ('metotreksatas','methotrexate'),
    'trimetoprimas': ('trimetoprimas','trimethoprim'),
    'spironolaktonas': ('spironolaktonas','spironolactone'),
    'kalio chloridas': ('kalio chloridas','potassium chloride'),
    'litis': ('litis','lithium'),
    'klopidogrelis': ('klopidogrelis','clopidogrel'),
    'omeprazolas': ('omeprazolas','omeprazole'),
}

PROFILES = {
    'ibuprofenas': ('NVNU', 'Mažina skausmą, uždegimą ir karščiavimą.'),
    'paracetamolis': ('Skausmą ir karščiavimą mažinantis vaistas', 'Mažina skausmą ir karščiavimą; kartais yra sudėtiniuose peršalimo preparatuose.'),
    'pseudoefedrinas': ('Nosies gleivinės paburkimą mažinantis preparatas', 'Gali palengvinti nosies užgulimą; svarbios kraujospūdžio ir širdies būklės atsargumo priemonės.'),
    'dekstrometorfanas': ('Kosulį slopinanti medžiaga', 'Slopina kosulio refleksą; kai kurių kitų vaistų deriniai reikalauja papildomos patikros.'),
    'naproksenas': ('NVNU', 'Mažina skausmą ir uždegimą.'),
    'diklofenakas': ('NVNU', 'Mažina skausmą ir uždegimą.'),
    'ketoprofenas': ('NVNU', 'Mažina skausmą ir uždegimą.'),
    'deksketoprofenas': ('NVNU', 'Mažina skausmą ir uždegimą.'),
    'acetilsalicilo rūgštis': ('Salicilatas / NVNU', 'Poveikis priklauso nuo dozės ir vartojimo tikslo, įskaitant trombocitų slopinimą.'),
    'varfarinas': ('Antikoaguliantas', 'Mažina kraujo krešėjimą; svarbi kitų vaistų įtaka kraujavimo rizikai.'),
    'sertralinas': ('SSRI antidepresantas', 'Veikia serotonino apykaitą; galimos sąveikos su kraujavimą didinančiais vaistais.'),
    'prednizolonas': ('Gliukokortikoidas', 'Mažina uždegimą ir imuninį atsaką.'),
    'klaritromicinas': ('Makrolidų antibiotikas', 'Veikia bakterijas; gali keisti kai kurių kitų vaistų koncentraciją.'),
    'simvastatinas': ('Statinas', 'Mažina cholesterolio koncentraciją.'),
    'pseudoefedrinas': ('Simpatomimetikas', 'Mažina nosies užgulimą; svarbu įvertinti kraujospūdį.'),
}
# Tik aiškiai atpažintos sisteminių NVNU grupės veikliosios medžiagos.
# Mažų acetilsalicilo rūgšties antitrombocitinių dozių situacijos atskiros,
# todėl šiame automatiniame sąraše jos sąmoningai nėra.
SYSTEMIC_NSAIDS = {
    'ibuprofenas', 'naproksenas', 'diklofenakas',
    'deksketoprofenas', 'ketoprofenas',
}
NSAID_SOURCE = 'https://www.medicines.org.uk/emc/product/159/smpc'
DICLOFENAC_SOURCE = 'https://www.medicines.org.uk/emc/product/2662/smpc'


def nsaid_pair_risk(a, b):
    """Vienas konkretus grupės perspėjimas; tai nėra saugumo išvada.

    Forma ir vartojimo kelias čia nežinomi, tad perspėjimas kalba apie
    sisteminį vartojimą ir nepaverčia vietinio preparato sisteminiu.
    """
    found = sorted({(x, y) for x in a for y in b
                    if x in SYSTEMIC_NSAIDS and y in SYSTEMIC_NSAIDS
                    and x != y})
    return bool(found)


BOTANICAL_TERMS=('verbenu','verbena','gencijon','gentiana','rugstyni','rumex','seivamed','sambucus','raktazol','primula','eteriniu alieju','eukalipt','miros','mirtu')


def substances(value):
    """Atsargiai išgauna žinomus komponentus iš sudėtinių VVKT pavadinimų.

    Nežinomo viso augalinio mišinio nepaverčia išgalvota veikliąja medžiaga.
    """
    text=norm(value)
    if text in ('','—','nan','none','nenurodyta'):
        return []
    found=[]
    for canonical, aliases in ALIASES.items():
        for name in aliases:
            t=norm(name)
            if re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)', text):
                found.append(canonical)
                break
    if found:
        return list(dict.fromkeys(found))
    return [text]


def records(data_path=None):
    path=Path(data_path) if data_path else (DATA if DATA.is_file() else FALLBACK)
    if not path.is_file(): return []
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def evaluate_pair(first, second, data_path=None):
    a,b=substances(first),substances(second)
    if not a or not b:
        return dict(type='missing',records=[],a=a,b=b,shared=[],profiles=[],nsaid_group_warning=False)
    shared=sorted(set(a)&set(b))
    matches=[]
    pairs={frozenset((x,y)) for x in a for y in b if x!=y}
    for row in records(data_path):
        key=frozenset((substances(row.get('veiklioji_1')) or [''])[0:1] + (substances(row.get('veiklioji_2')) or [''])[0:1])
        if len(key)==2 and key in pairs:
            matches.append(row)
    # Konkreti pora prioritetinė, tik po jos bendroji grupės taisyklė.
    matches.sort(key=lambda r: 0 if r.get('pagrindimo_lygis','konkreti_pora')=='konkreti_pora' else 1)
    profiles=[]
    for name in dict.fromkeys(a+b):
        if name in PROFILES:
            profiles.append((name,)+PROFILES[name])
    return dict(type='documented' if matches else 'screening', records=matches,
                a=a, b=b, shared=shared, profiles=profiles,
                nsaid_group_warning=nsaid_pair_risk(a,b))


def find_documented_pairs(first,second,data_path=None):
    return evaluate_pair(first,second,data_path)['records']


def explain_pair(first, second, data_path=None):
    """Duomenų pagrindu gautas rezultatas, nenaudojantis Streamlit."""
    result = evaluate_pair(first, second, data_path)
    if result['type'] == 'missing':
        status = 'truksta_sudeties'
        summary = 'Nepavyko nustatyti vieno ar abiejų preparatų sudėties.'
    elif result['shared']:
        status = 'dubliavimas'
        summary = 'Abiejuose preparatuose kartojasi ta pati veiklioji medžiaga.'
    elif result.get('nsaid_group_warning'):
        status = 'nvnu_grupes_rizika'
        summary = 'Abiejų preparatų veikliosios medžiagos priklauso NVNU grupei.'
    elif result['records']:
        status = 'zinoma_saveika'
        summary = 'Pagal veikliųjų medžiagų porą rasta sąveikos informacija.'
    else:
        status = 'sudeties_apzvalga'
        summary = 'Vienodų veikliųjų medžiagų nenustatyta; kitų sąveikų ši automatinė patikra neatmeta.'
    return dict(result, status=status, summary=summary)


def show_pair_result(first, second):
    """Trumpas mokomasis poros rezultatas; medikamentų atpažinimo nekeičia."""
    import streamlit as st
    result = explain_pair(first, second)
    st.markdown('## 🔬 Kaip šie vaistai veikia kartu?')

    if result['status'] == 'truksta_sudeties':
        st.warning('Nepakanka sudėties duomenų abiem vaistams palyginti.')
        return False

    if result['shared']:
        shared = ', '.join(result['shared'])
        st.warning('⚠️ **Kartojasi veiklioji medžiaga:** ' + shared)
        if any('ambroksol' in x or 'ambroxol' in x for x in result['shared']):
            st.write('**Bendras poveikis.** Abu preparatai turi ambroksolio, kuris padeda skystinti kvėpavimo takų sekretą ir palengvina atsikosėjimą. Tai tas pats, o ne du vienas kitą papildantys veikimo mechanizmai.')
        else:
            st.write('**Bendras poveikis.** Abu preparatai turi tą pačią veikliąją medžiagą, todėl jų poveikis dubliuojasi.')
        st.write('**Kas vyksta vartojant kartu?** Suvartoti tos pačios veikliosios medžiagos kiekiai susideda. Vien tik skirtingi preparatų pavadinimai nereiškia papildomos naudos.')
        st.write('**Į ką atkreipti dėmesį?** Tikrinti bendrą dozę ir vartojimo dažnį. Vien dubliavimas dar neįrodo perdozavimo, tačiau gali padidinti jo tikimybę.')
        st.caption('Rezultatas nustatytas iš sudėties, ne iš individualiai patvirtintos klinikinės išvados.')

    if result.get('nsaid_group_warning'):
        st.error(
            '⛔ **NVNU grupės sąveikos perspėjimas.** Jei abu preparatai '
            'vartojami sistemiškai, jų derinys paprastai nerekomenduojamas. '
            'Gali didėti virškinamojo trakto opų ir kraujavimo rizika. '
            'Vien skirtingi veikliųjų medžiagų pavadinimai nereiškia, kad '
            'tai nepriklausomi ar saugūs kartu vartoti vaistai.'
        )
        st.caption(
            'Tai oficialių preparatų charakteristikų pagrindu pateikta '
            'farmakologinės grupės atsargumo taisyklė, ne individualus '
            'klinikinis derinio įvertinimas. Vietinio ir sisteminio vartojimo '
            'situacijos gali skirtis – reikia patikrinti vaisto formą ir kelią.'
        )
        st.markdown(
            '[Deksketoprofeno preparato charakteristika](' + NSAID_SOURCE + ')'
            ' · [Diklofenako preparato charakteristika](' + DICLOFENAC_SOURCE + ')'
        )

    if result['records']:
        st.markdown('**Žinomos sąveikos ir atsargumo sąlygos**')
        for r in result['records'][:8]:
            category = r.get('kategorija', 'Reikia atsargumo')
            level = r.get('pagrindimo_lygis', 'konkreti_pora')
            pair = str(r.get('veiklioji_1', '')) + ' + ' + str(r.get('veiklioji_2', ''))
            if 'Paprastai galima' in category:
                st.success('✅ **' + pair + ': ' + category + '**')
            elif 'Kontraindikuot' in category or 'Nevartoti' in category:
                st.error('⛔ **' + pair + ': ' + category + '**')
            else:
                st.warning('⚠️ **' + pair + ': ' + category + '**')
            if r.get('saveika'):
                st.write('**Kas žinoma:** ' + str(r['saveika']))
            if r.get('mechanizmas'):
                st.write('**Kodėl:** ' + str(r['mechanizmas']))
            if r.get('salygos'):
                st.write('**Svarbios sąlygos:** ' + str(r['salygos']))
            if level == 'farmakologines_grupes_taisykle':
                st.caption('Tai bendroji farmakologinės grupės taisyklė; konkrečios poros klinikinė rizika individualiai nepatvirtinta.')
            url = str(r.get('saltinis_url', '')).strip()
            if url.startswith('https://'):
                st.markdown('[Peržiūrėti informacijos šaltinį](' + url + ')')
        if len(result['records']) > 8:
            st.caption('Rodomi 8 aktualiausi įrašai iš mokomosios duomenų bazės.')
    elif not result['shared'] and not result.get('nsaid_group_warning'):
        st.info('**Sudėties palyginimas:** tų pačių veikliųjų medžiagų nenustatyta.')
        st.write('**Sąveikos duomenys:** šiai konkrečiai porai mūsų mokomojoje bazėje dokumentuotos sąveikos įrašo nėra. Tai nėra patvirtinimas, kad derinys saugus.')

    # Išskleidžiamą skiltį rodome tik tada, kai joje yra realios informacijos.
    if result['profiles']:
        with st.expander('ℹ️ Papildoma veikliųjų medžiagų informacija'):
            for name, group, desc in result['profiles']:
                st.write('**' + name.capitalize() + ' (' + group + '):** ' + desc)
    if any(any(term in x for term in BOTANICAL_TERMS) for x in result['a'] + result['b']):
        st.caption('Augalinių medžiagų derinių sąveikos gali būti nepakankamai ištirtos; įrašo nebuvimas nepatvirtina saugumo.')
    st.caption('Edukacinė sudėties ir šaltinių analizė; nėra individuali medicininė rekomendacija.')
    return bool(result['records'] or result['shared'] or result['profiles'] or result.get('nsaid_group_warning'))
