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


        # =================================================
        # GRUPIŲ SĄVEIKOS TAISYKLĖ
        # =================================================

        group_rule = find_group_interaction(
            ingredient_1,
            ingredient_2
        )

        group_1 = get_ingredient_group(
            ingredient_1
        )

        group_2 = get_ingredient_group(
            ingredient_2
        )

        # =================================================
        # RODOME ATPAŽINTAS FARMAKOLOGINES GRUPES
        # =================================================

        st.markdown(
            "### 🧬 Farmakologinės grupės"
        )

        col_group_1, col_group_2 = st.columns(2)

        with col_group_1:

            st.markdown(
                f"**{first}**"
            )

            if group_1:

                st.write(
                    group_1
                )

            else:

                st.caption(
                    "Farmakologinė grupė šiame "
                    "prototipe automatiškai "
                    "nenustatyta."
                )

        with col_group_2:

            st.markdown(
                f"**{second}**"
            )

            if group_2:

                st.write(
                    group_2
                )

            else:

                st.caption(
                    "Farmakologinė grupė šiame "
                    "prototipe automatiškai "
                    "nenustatyta."
                )

        # =================================================
        # JEI RASTA PATVIRTINTA GRUPINĖ TAISYKLĖ
        # =================================================

        if group_rule is not None:

            st.markdown(
                "### ⚠️ Galima farmakologinė sąveika"
            )

            interaction_text = clean_value(
                group_rule.get(
                    "interaction"
                )
            )

            st.warning(
                interaction_text
            )

            st.markdown(
                "#### Galimas poveikis / rizika"
            )

            effect_text = clean_value(
                group_rule.get(
                    "effect"
                )
            )

            st.write(
                effect_text
            )

            st.markdown(
                "#### 📋 Išvada"
            )

            conclusion_text = clean_value(
                group_rule.get(
                    "conclusion"
                )
            )

            st.error(
                conclusion_text
            )

            source = clean_value(
                group_rule.get(
                    "source"
                )
            )

            if source != "—":

                st.markdown(
                    "#### 📚 Sąveikos taisyklės šaltinis"
                )

                st.write(
                    source
                )

        # =================================================
        # JEI PATVIRTINTOS GRUPINĖS TAISYKLĖS NĖRA
        # =================================================

        else:

            st.markdown(
                "### ℹ️ Sąveikos vertinimas"
            )

            if group_1 and group_2:

                st.info(
                    "Abiejų preparatų farmakologinės "
                    "grupės šiame prototipe nustatytos, "
                    "tačiau šiai grupių porai "
                    "patvirtinta sąveikos taisyklė "
                    "dar neįtraukta. "
                    "Tai nėra išvada, kad preparatus "
                    "saugu vartoti kartu."
                )

            elif group_1 or group_2:

                st.info(
                    "Pavyko nustatyti tik vieno iš "
                    "preparatų farmakologinę grupę. "
                    "Todėl patikima automatinė "
                    "grupių sąveikos taisyklė "
                    "negali būti pritaikyta. "
                    "Tai nėra išvada, kad preparatus "
                    "saugu vartoti kartu."
                )

            else:

                st.info(
                    "Šių veikliųjų medžiagų "
                    "farmakologinių grupių prototipas "
                    "automatiškai nenustatė. "
                    "Todėl automatinė sąveikos "
                    "taisyklė nepateikiama. "
                    "Tai nėra išvada, kad preparatus "
                    "saugu vartoti kartu."
                )

            st.caption(
                "Prototipas pateikia tik į jo "
                "patikrintą taisyklių bazę "
                "įtrauktas farmakologines sąveikas. "
                "Individualų vaistų derinį reikia "
                "vertinti pagal oficialias preparatų "
                "charakteristikų santraukas ir "
                "sveikatos priežiūros specialisto "
                "rekomendacijas."
            )

        # =================================================
        # VARTOJIMO BŪDAS
        # =================================================

        st.markdown(
            "### 💉 Vartojimo būdas"
        )

        if first_row is not None:

            st.write(
                f"**{first}:**",
                clean_value(
                    first_row.get(
                        "vartojimo_budas"
                    )
                )
            )

        if second_row is not None:

            st.write(
                f"**{second}:**",
                clean_value(
                    second_row.get(
                        "vartojimo_budas"
                    )
                )
            )

        # =================================================
        # FARMAKOLOGINIŲ PROFILIŲ ŠALTINIAI
        # =================================================

        st.markdown(
            "### 📚 Farmakologinių profilių šaltiniai"
        )

        shown_sources = set()

        for info in [
            info_1,
            info_2
        ]:

            if info is None:
                continue

            source = clean_value(
                info.get(
                    "source"
                )
            )

            if (
                source != "—"
                and source not in shown_sources
            ):

                st.write(
                    f"• {source}"
                )

                shown_sources.add(
                    source
                )

        if not shown_sources:

            st.caption(
                "Atskiri farmakologiniai profiliai "
                "šiems ingredientams patikrintoje "
                "prototipo bazėje dar neaprašyti."
            )

        st.caption(
            "Preparatų pavadinimai, veikliosios "
            "medžiagos, stiprumas, farmacinė forma "
            "ir vartojimo būdas gaunami iš projekte "
            "esančio VVKT duomenų rinkinio."
        )


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
