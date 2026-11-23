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
    'diklofenakas': ('diklofenakas','diclofenac'),
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
        return dict(type='missing',records=[],a=a,b=b,shared=[],profiles=[])
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
    return dict(type='documented' if matches else 'screening',records=matches,a=a,b=b,shared=shared,profiles=profiles)


def find_documented_pairs(first,second,data_path=None):
    return evaluate_pair(first,second,data_path)['records']


def show_pair_result(first,second):
    import streamlit as st
    result=evaluate_pair(first,second)
    st.markdown('### 🧩 Dviejų vaistų sąveikos ir sudėties analizė')
    if result['type']=='missing':
        st.warning('Nepakanka informacijos apie vieno ar abiejų preparatų veikliąsias medžiagas.')
        return False
    if result['shared']:
        st.warning('⚠️ **Kartojasi veiklioji medžiaga:** '+', '.join(result['shared'])+'. Dviejų preparatų vartojimas gali lemti tos pačios medžiagos dozės sumavimą.')
        st.caption('Faktinei perdozavimo rizikai įvertinti reikia žinoti vartojamą kiekį, dažnį ir konkrečių preparatų dozavimo rekomendacijas.')
    if result['records']:
        st.caption(f'Rasta {len(result["records"])} sąveikų / grupės atsargumo taisyklių pagal pasirinktų preparatų sudėtį.')
        for r in result['records'][:12]:
            category=r.get('kategorija','Reikia atsargumo')
            lev=r.get('pagrindimo_lygis','konkreti_pora')
            heading=f'**{r.get("veiklioji_1")} + {r.get("veiklioji_2")}** — {category}'
            if 'Paprastai galima' in category:
                st.success('✅ '+heading)
            elif 'Kontraindikuot' in category or 'Nevartoti' in category:
                st.error('⛔ '+heading)
            else:
                st.warning('⚠️ '+heading)
            st.write('**Kas žinoma:**',r.get('saveika',''))
            st.write('**Kodėl / kokia rizika:**',r.get('mechanizmas',''))
            st.write('**Į ką atkreipti dėmesį:**',r.get('salygos',''))
            if lev=='farmakologines_grupes_taisykle':
                st.caption('Tai farmakologinės grupės atsargumo taisyklė, o ne individualiai patvirtinta preparatų poros sąveika.')
            url=r.get('saltinis_url','')
            if url.startswith('https://'):
                st.markdown('**Viešas šaltinis:** [skaityti]('+url+')')
        if len(result['records'])>12:
            st.caption('Rodomi pirmieji 12 radinių; visus duomenis galima peržiūrėti CSV faile.')
    else:
        st.info('**Konkretaus šių vaistų derinio sąveikos mūsų bazėje neaprašyta.** Žemiau pateikiama sudėties analizė, bet tai nėra įrodymas, kad derinys saugus.')
    with st.expander('🔬 Ką daro atpažintos veikliosios medžiagos?', expanded=not bool(result['records'])):
        if result['profiles']:
            for name,group,desc in result['profiles']:
                st.write(f'**{name.capitalize()} — {group}.** {desc}')
        else:
            st.write('Šių medžiagų farmakologinių profilių mokomojoje bazėje kol kas nėra.')
        for src,parts in (('Pirmasis preparatas',result['a']),('Antrasis preparatas',result['b'])):
            st.caption(src+': '+', '.join(parts))
        if any(any(term in x for term in BOTANICAL_TERMS) for x in result['a']+result['b']):
            st.info('Viename preparate yra augalinių medžiagų. Duomenų apie augalinių preparatų sąveikas dažnai mažiau; jų nebuvimas šioje bazėje nėra saugumo patvirtinimas.')
    st.caption('Mokomasis AI prototipas. Sąveikos pateikiamos pagal viešų šaltinių ištraukas arba atskirai pažymėtas grupių taisykles. Be paciento duomenų negalima teigti, kad konkretų derinį saugu vartoti.')
    return bool(result['records'] or result['shared'] or result['profiles'])
