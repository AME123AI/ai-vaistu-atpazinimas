from pathlib import Path
import json
import re
import unicodedata
from difflib import get_close_matches, SequenceMatcher

import joblib
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st

from PIL import Image, ImageOps, ImageEnhance
from skimage.feature import hog


# =========================================================
# NUSTATYMAI
# =========================================================

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="AI vaistų atpažinimas",
    page_icon="💊",
    layout="wide"
)


# =========================================================
# DUOMENŲ ĮKĖLIMAS
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "hog_logreg.joblib")


@st.cache_data
def load_json(name):
    path = ROOT / name

    if not path.exists():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8")
    )


@st.cache_data
def load_csv(name):
    path = ROOT / name

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        encoding="utf-8-sig"
    )


@st.cache_data
def load_vvkt():
    return pd.read_csv(
        ROOT / "PreparatasPakuote.csv",
        low_memory=False,
        encoding="utf-8-sig"
    )


model = load_model()

drug_data = load_json("drug_profiles.json")
ingredient_profiles = load_json("ingredient_profiles.json")

interactions = load_csv("interactions.csv")
ingredient_knowledge = load_csv("ingredient_knowledge.csv")
interaction_rules = load_csv("interaction_rules.csv")

vvkt = load_vvkt()


# =========================================================
# VVKT PARUOŠIMAS
# =========================================================

vvkt = vvkt[
    vvkt["preparato_pav"].notna()
].copy()

vvkt["preparato_pav"] = (
    vvkt["preparato_pav"]
    .astype(str)
    .str.strip()
)

vvkt_names = sorted(
    vvkt["preparato_pav"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)


# =========================================================
# SESSION STATE
# =========================================================

if "first_vvkt_drug" not in st.session_state:
    st.session_state.first_vvkt_drug = None


# =========================================================
# BENDROSIOS FUNKCIJOS
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(
        r"[^a-z0-9+\-/.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_value(value):

    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass

    value = str(value).strip()

    return value if value else "—"


def similarity(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# =========================================================
# VVKT FUNKCIJOS
# =========================================================

def get_vvkt_rows(name):

    if not name:
        return pd.DataFrame()

    return vvkt[
        vvkt["preparato_pav"] == name
    ]


def get_vvkt_row(name):

    rows = get_vvkt_rows(name)

    if rows.empty:
        return None

    return rows.iloc[0]


def get_vvkt_ingredient(name):

    row = get_vvkt_row(name)

    if row is None:
        return None

    value = row.get(
        "veiklioji_medz_lt"
    )

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    return value if value else None


def show_vvkt_info(name):

    row = get_vvkt_row(name)

    if row is None:

        st.warning(
            "VVKT informacijos šiam preparatui "
            "rasti nepavyko."
        )

        return

    st.write(
        "**Pavadinimas:**",
        clean_value(row.get("preparato_pav"))
    )

    st.write(
        "**Veiklioji medžiaga:**",
        clean_value(row.get("veiklioji_medz_lt"))
    )

    st.write(
        "**Stiprumas:**",
        clean_value(row.get("stiprumas"))
    )

    st.write(
        "**Farmacinė forma:**",
        clean_value(row.get("farmacine_forma_lt"))
    )

    st.write(
        "**Vartojimo būdas:**",
        clean_value(row.get("vartojimo_budas"))
    )

    st.write(
        "**Recepto poreikis:**",
        clean_value(row.get("recepto_poreikis"))
    )


def find_names(query):

    if not query:
        return []

    q = normalize_text(query)

    results = []

    for name in vvkt_names:

        if q in normalize_text(name):
            results.append(name)

        if len(results) >= 30:
            break

    return results


# =========================================================
# FARMAKOLOGINĖ ŽINIŲ BAZĖ
# =========================================================

def get_ingredient_knowledge(ingredient):

    if not ingredient:
        return None

    if ingredient_knowledge.empty:
        return None

    if "ingredient" not in ingredient_knowledge.columns:
        return None

    target = normalize_text(
        ingredient
    )

    matches = ingredient_knowledge[
        ingredient_knowledge["ingredient"]
        .fillna("")
        .astype(str)
        .apply(normalize_text)
        == target
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def show_ingredient_explanation(
    drug_name,
    ingredient
):

    info = get_ingredient_knowledge(
        ingredient
    )

    st.markdown(
        f"#### {drug_name} – "
        f"{ingredient or 'veiklioji medžiaga nenustatyta'}"
    )

    if info is None:

        st.caption(
            "Šios veikliosios medžiagos "
            "farmakologinis profilis dar "
            "neįtrauktas į patikrintą "
            "prototipo bazę."
        )

        return False

    st.write(
        "**Farmakologinė grupė:**",
        clean_value(info.get("group"))
    )

    st.write(
        "**Veikimo mechanizmas:**",
        clean_value(info.get("mechanism"))
    )

    st.write(
        "**Pagrindinis poveikis:**",
        clean_value(info.get("effect"))
    )

    return True


# =========================================================
# AUTOMATINIS FARMAKOLOGINĖS GRUPĖS NUSTATYMAS
# =========================================================

INGREDIENT_GROUP_RULES = {

    "Nesteroidinis vaistas nuo uždegimo (NVNU)": [
        "ibuprofen",
        "naproksen",
        "deksketoprofen",
        "dexketoprofen",
        "ketoprofen",
        "diklofenak",
        "diclofenac",
        "celekoksib",
        "celecoxib",
        "meloksikam",
        "meloxicam",
        "indometacin",
        "etoricoxib",
        "etorikoksib",
        "ketorolak",
    ],

    "Antikoaguliantas": [
        "warfarin",
        "varfarin",
        "apiksaban",
        "apixaban",
        "rivaroksaban",
        "rivaroxaban",
        "dabigatran",
        "edoksaban",
        "edoxaban",
        "heparin",
        "enoksaparin",
        "enoxaparin",
    ],

    "Antitrombocitinis vaistas": [
        "klopidogrel",
        "clopidogrel",
        "tikagrelor",
        "ticagrelor",
        "prasugrel",
        "acetilsalicilo rugst",
        "acetylsalicylic acid",
    ],

    "SSRI antidepresantas": [
        "sertralin",
        "escitalopram",
        "citalopram",
        "fluoksetin",
        "fluoxetine",
        "paroksetin",
        "paroxetine",
        "fluvoksamin",
        "fluvoxamine",
    ],

    "AKF inhibitorius": [
        "ramipril",
        "perindopril",
        "enalapril",
        "lisinopril",
        "kaptopril",
        "captopril",
        "fosinopril",
        "trandolapril",
    ],

    "Angiotenzino II receptorių blokatorius (ARB)": [
        "kandesartan",
        "candesartan",
        "losartan",
        "valsartan",
        "telmisartan",
        "irbesartan",
        "olmesartan",
    ],

    "Kalį sulaikantis diuretikas": [
        "spironolakton",
        "spironolacton",
        "eplerenon",
        "eplerenone",
        "amilorid",
        "triamteren",
    ],

    "Diuretikas": [
        "hidrochlorotiazid",
        "hydrochlorothiazide",
        "indapamid",
        "furosemid",
        "torasemid",
        "chlortalidon",
        "chlorthalidone",
    ],

    "Litis": [
        "licio karbonat",
        "lithium carbonate",
        "lithium",
    ],

    "Metotreksatas": [
        "metotreksat",
        "methotrexate",
    ],

    "Širdies glikozidas": [
        "digoksin",
        "digoxin",
    ],

    "PDE5 inhibitorius": [
        "sildenafil",
        "tadalafil",
        "vardenafil",
        "avanafil",
    ],

    "Nitratas": [
        "nitroglicerin",
        "glicerilio trinitrat",
        "glyceryl trinitrate",
        "izosorbido mononitrat",
        "isosorbide mononitrate",
        "izosorbido dinitrat",
        "isosorbide dinitrate",
    ],

    "Simvastatinas": [
        "simvastatin",
    ],

    "Stiprus CYP3A4 inhibitorius": [
        "klaritromicin",
        "clarithromycin",
        "eritromicin",
        "erythromycin",
        "itrakonazol",
        "itraconazole",
        "ketokonazol",
        "ketoconazole",
        "posakonazol",
        "posaconazole",
        "vorikonazol",
        "voriconazole",
        "ritonavir",
    ],

    "Tramadolis": [
        "tramadol",
    ],

    "Augalinis atsikosėjimą lengvinantis preparatas": [
        "raktažol",
        "primula",
        "ciobrel",
        "thymus",
        "eukalipt",
        "mirt",
        "eteriniu alieju",
    ],
}


def get_ingredient_group(ingredient):

    if not ingredient:
        return None

    # 1. Pirmiausia tikriname patikrintą ingredient_knowledge.csv
    info = get_ingredient_knowledge(ingredient)

    if info is not None:

        group = clean_value(
            info.get("group")
        )

        if group:
            return group

    # 2. Jei profilio nėra, bandome automatiškai nustatyti
    # grupę pagal VVKT veikliosios medžiagos pavadinimą.

    normalized_ingredient = normalize_text(
        ingredient
    )

    for group, keywords in INGREDIENT_GROUP_RULES.items():

        for keyword in keywords:

            normalized_keyword = normalize_text(
                keyword
            )

            if normalized_keyword in normalized_ingredient:
                return group

    return None


        # =========================================================
# GRUPIŲ SĄVEIKOS
# =========================================================

def find_group_interaction(
    ingredient_1,
    ingredient_2
):

    if interaction_rules.empty:
        return None

    group_1 = get_ingredient_group(
        ingredient_1
    )

    group_2 = get_ingredient_group(
        ingredient_2
    )

    if not group_1 or not group_2:
        return None

    group_1_norm = normalize_text(
        group_1
    )

    group_2_norm = normalize_text(
        group_2
    )

    for _, row in interaction_rules.iterrows():

        rule_group_a = normalize_text(
            row.get("group_a")
        )

        rule_group_b = normalize_text(
            row.get("group_b")
        )

        direct_match = (
            group_1_norm == rule_group_a
            and
            group_2_norm == rule_group_b
        )

        reverse_match = (
            group_1_norm == rule_group_b
            and
            group_2_norm == rule_group_a
        )

        if direct_match or reverse_match:

            result = row.to_dict()

            result["detected_group_1"] = group_1
            result["detected_group_2"] = group_2

            return result

    return None

# =========================================================
# APIE SISTEMĄ
# =========================================================

st.divider()

with st.expander(
    "ℹ️ Apie sistemą"
):

    st.write(
        "Pakuotės tekstas nuskaitomas OCR "
        "ir lyginamas su VVKT registruotų "
        "preparatų katalogu."
    )

    st.write(
        "Atpažinimo algoritmas vertina "
        "preparato pavadinimą, veikliąją "
        "medžiagą, stiprumą ir farmacinę formą."
    )

    st.write(
        "Farmakologinei analizei naudojami "
        "patikrinti veikliųjų medžiagų profiliai "
        "ir farmakologinių grupių sąveikos "
        "taisyklės."
    )

    st.write(
        "HOG + Logistic Regression modelis "
        "naudojamas kaip bazinis tyrimo modelis "
        "ir mokytas tik su 24 preparatų klasėmis."
    )


st.caption(
    "Edukacinis AI prototipas – "
    "vaistų pakuočių atpažinimas ir "
    "farmakologinės informacijos demonstravimas."
)
