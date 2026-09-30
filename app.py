from pathlib import Path
import json
import re
import unicodedata
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
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
    return json.loads(
        (ROOT / name).read_text(encoding="utf-8")
    )


@st.cache_data
def load_interactions():
    return pd.read_csv(
        ROOT / "interactions.csv"
    )


@st.cache_data
def load_vvkt():
    return pd.read_csv(
        ROOT / "PreparatasPakuote.csv",
        low_memory=False
    )


model = load_model()
classes = list(model.classes_)

drug_data = load_json(
    "drug_profiles.json"
)

ingredient_profiles = load_json(
    "ingredient_profiles.json"
)

interactions = load_interactions()

vvkt = load_vvkt()

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
    .drop_duplicates()
    .tolist()
)


# =========================================================
# SESSION STATE
# =========================================================

if "first_vvkt_drug" not in st.session_state:
    st.session_state.first_vvkt_drug = None

if "second_vvkt_drug_confirmed" not in st.session_state:
    st.session_state.second_vvkt_drug_confirmed = None


# =========================================================
# PAGALBINĖS FUNKCIJOS
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

    return text


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

    if pd.isna(value):
        return None

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
        clean_value(
            row.get("preparato_pav")
        )
    )

    st.write(
        "**Veiklioji medžiaga:**",
        clean_value(
            row.get("veiklioji_medz_lt")
        )
    )

    st.write(
        "**Stiprumas:**",
        clean_value(
            row.get("stiprumas")
        )
    )

    st.write(
        "**Farmacinė forma:**",
        clean_value(
            row.get("farmacine_forma_lt")
        )
    )

    st.write(
        "**Vartojimo būdas:**",
        clean_value(
            row.get("vartojimo_budas")
        )
    )

    st.write(
        "**Recepto poreikis:**",
        clean_value(
            row.get("recepto_poreikis")
        )
    )


def find_names(query):
    if not query:
        return []

    q = normalize_text(query)

    results = []

    for name in vvkt_names:
        if q in normalize_text(name):
            results.append(name)

        if len(results) >= 20:
            break

    return results


# =========================================================
# SENŲ SĄVEIKOS TAISYKLIŲ SUSIEJIMAS
# =========================================================
#
# interactions.csv šiuo metu turi tik kelias
# patikrintas taisykles pagal senų preparatų pavadinimus.
#
# Todėl jas susiejame su VVKT veikliosiomis medžiagomis.
# Sąrašas sąmoningai ribotas – nebandome išgalvoti
# neegzistuojančių sąveikų.
# =========================================================

LEGACY_ALIASES = {
    "Aggrex": [
        "acetilsalicilo rugst",
        "aspirin"
    ],

    "CELEBREX": [
        "celekoksib",
        "celecoxib"
    ],

    "Candalkan": [
        "kandesartan",
        "candesartan"
    ]
}


def ingredient_matches(
    ingredient,
    legacy_drug
):
    if not ingredient:
        return False

    ingredient_norm = normalize_text(
        ingredient
    )

    aliases = LEGACY_ALIASES.get(
        legacy_drug,
        []
    )

    for alias in aliases:

        if normalize_text(alias) in ingredient_norm:
            return True

    return False


def find_interactions(
    ingredient_1,
    ingredient_2
):
    found = []

    if not ingredient_1 or not ingredient_2:
        return found

    for _, rule in interactions.iterrows():

        drug_a = str(
            rule.get(
                "drug_a",
                ""
            )
        ).strip()

        drug_b = str(
            rule.get(
                "drug_b",
                ""
            )
        ).strip()

        direct = (
            ingredient_matches(
                ingredient_1,
                drug_a
            )
            and
            ingredient_matches(
                ingredient_2,
                drug_b
            )
        )

        reverse = (
            ingredient_matches(
                ingredient_1,
                drug_b
            )
            and
            ingredient_matches(
                ingredient_2,
                drug_a
            )
        )

        if direct or reverse:
            found.append(rule)

    return found


# =========================================================
# DIZAINAS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
    }

    h1 {
        text-align: center;
    }

    .step {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .result-box {
        padding: 18px;
        border: 1px solid rgba(150,150,150,.25);
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ANTRAŠTĖ
# =========================================================

st.title(
    "💊 AI vaistų atpažinimas ir sąveikų paaiškinimas"
)

st.caption(
    "Patobulintas prototipas: OCR + VVKT vaistų "
    "duomenys ir HOG + Logistic Regression baseline modelis"
)

st.warning(
    "⚠️ Edukacinis prototipas. Pateikiama informacija "
    "nėra individuali medicininė rekomendacija ir "
    "nepakeičia gydytojo ar vaistininko konsultacijos."
)


# =========================================================
# 1. NUOTRAUKA
# =========================================================

st.markdown(
    '<div class="step">'
    '📷 1. Įkelkite vaisto pakuotės nuotrauką'
    '</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Vaisto pakuotės nuotrauka",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# 2. AI ATPAŽINIMO REZULTATAS
# =========================================================

if uploaded:

    image = Image.open(
        uploaded
    ).convert("RGB")

    st.image(
        image,
        caption="Įkelta vaisto pakuotė",
        width=430
    )

    st.markdown(
        '<div class="step">'
        '🤖 2. AI atpažinimo rezultatas'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    ocr_text = pytesseract.image_to_string(
        image
    )

    normalized_ocr = normalize_text(
        ocr_text
    )

    with st.expander(
        "🔤 Peržiūrėti OCR nuskaitytą tekstą"
    ):
        st.text(
            ocr_text
            if ocr_text.strip()
            else "Teksto atpažinti nepavyko."
        )

    detected_ingredient = None
    detected_strength = None
    detected_form = None
    ocr_suggestions = []

    # -----------------------------------------------------
    # 2A. TIKSLUS PREPARATO PAVADINIMAS
    # -----------------------------------------------------

    exact_matches = []

    for name in vvkt_names:

        normalized_name = normalize_text(
            name
        )

        if (
            len(normalized_name) >= 4
            and normalized_name in normalized_ocr
        ):
            exact_matches.append(name)

    exact_matches = sorted(
        exact_matches,
        key=len,
        reverse=True
    )

    if exact_matches:
        ocr_suggestions = exact_matches

    # -----------------------------------------------------
    # 2B. VEIKLIOJI MEDŽIAGA
    # -----------------------------------------------------

    ingredient_names = sorted(
        vvkt["veiklioji_medz_lt"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    ingredient_lookup = {
        normalize_text(x): x
        for x in ingredient_names
    }

    ingredient_normalized = list(
        ingredient_lookup.keys()
    )

    ocr_lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if len(line.strip()) >= 4
    ]

    for line in ocr_lines:

        parts = [line]

        parts.extend(
            word
            for word in line.split()
            if len(word) >= 5
        )

        for part in parts:

            part_norm = normalize_text(
                part
            )

            matches = get_close_matches(
                part_norm,
                ingredient_normalized,
                n=1,
                cutoff=0.75
            )

            if matches:

                detected_ingredient = (
                    ingredient_lookup[
                        matches[0]
                    ]
                )

                break

        if detected_ingredient:
            break

    # -----------------------------------------------------
    # 2C. STIPRUMAS
    # -----------------------------------------------------

    strength_match = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*mg\b",
        normalized_ocr
    )

    if strength_match:

        detected_strength = (
            strength_match
            .group(1)
            .replace(",", ".")
            + " mg"
        )

    # -----------------------------------------------------
    # 2D. FARMACINĖ FORMA
    # -----------------------------------------------------

    if (
        "plevele dengtos tabletes"
        in normalized_ocr
    ):

        detected_form = (
            "plėvele dengtos tabletės"
        )

    elif "tabletes" in normalized_ocr:

        detected_form = "tabletės"

    elif "kapsules" in normalized_ocr:

        detected_form = "kapsulės"

    elif "injekcinis tirpalas" in normalized_ocr:

        detected_form = "injekcinis tirpalas"

    # -----------------------------------------------------
    # 2E. JEIGU PAVADINIMAS NERASTAS,
    # FILTRUOJAME VVKT PAGAL VEIKLIĄJĄ MEDŽIAGĄ
    # -----------------------------------------------------

    if (
        not ocr_suggestions
        and detected_ingredient
    ):

        candidates = vvkt[
            vvkt["veiklioji_medz_lt"]
            .fillna("")
            .astype(str)
            .apply(normalize_text)
            ==
            normalize_text(
                detected_ingredient
            )
        ].copy()

        if detected_strength:

            strength_candidates = candidates[
                candidates["stiprumas"]
                .fillna("")
                .astype(str)
                .apply(normalize_text)
                .str.contains(
                    normalize_text(
                        detected_strength
                    ),
                    regex=False
                )
            ]

            if not strength_candidates.empty:
                candidates = strength_candidates

        if detected_form:

            form_candidates = candidates[
                candidates["farmacine_forma_lt"]
                .fillna("")
                .astype(str)
                .apply(normalize_text)
                .str.contains(
                    normalize_text(
                        detected_form
                    ),
                    regex=False
                )
            ]

            if not form_candidates.empty:
                candidates = form_candidates

        ocr_suggestions = (
            candidates["preparato_pav"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

    # -----------------------------------------------------
    # 2F. ATSARGINĖ FUZZY PAIEŠKA
    # -----------------------------------------------------

    if not ocr_suggestions:

        fuzzy_matches = []

        normalized_name_map = {
            normalize_text(name): name
            for name in vvkt_names
        }

        normalized_names = list(
            normalized_name_map.keys()
        )

        for line in ocr_lines:

            line_norm = normalize_text(
                line
            )

            matches = get_close_matches(
                line_norm,
                normalized_names,
                n=5,
                cutoff=0.80
            )

            for match in matches:

                fuzzy_matches.append(
                    normalized_name_map[
                        match
                    ]
                )

        ocr_suggestions = fuzzy_matches

    # -----------------------------------------------------
    # PASIKARTOJIMŲ PAŠALINIMAS
    # -----------------------------------------------------

    ocr_suggestions = list(
        dict.fromkeys(
            ocr_suggestions
        )
    )

    ocr_suggestions = (
        ocr_suggestions[:20]
    )

    # -----------------------------------------------------
    # KĄ ATPAŽINO OCR
    # -----------------------------------------------------

    if (
        detected_ingredient
        or detected_strength
        or detected_form
    ):

        st.markdown(
            "#### 🧾 Iš pakuotės aptikta informacija"
        )

        if detected_ingredient:

            st.write(
                "• **Veiklioji medžiaga:**",
                detected_ingredient
            )

        if detected_strength:

            st.write(
                "• **Stiprumas:**",
                detected_strength
            )

        if detected_form:

            st.write(
                "• **Farmacinė forma:**",
                detected_form
            )

    # -----------------------------------------------------
    # VVKT REZULTATAS
    # -----------------------------------------------------

    if ocr_suggestions:

        st.markdown(
            "#### 🔎 VVKT atitinkantys preparatai"
        )

        if len(ocr_suggestions) == 1:

            st.success(
                "Pagal pakuotėje aptiktą informaciją "
                "rastas vienas VVKT preparatas."
            )

        else:

            st.info(
                f"Pagal pakuotėje aptiktą informaciją "
                f"rasti {len(ocr_suggestions)} galimi "
                "VVKT preparatai. "
                "Patvirtinkite preparatą pagal pakuotę."
            )

        ocr_selected = st.selectbox(
            "Pasirinkite preparatą",
            ocr_suggestions,
            key="ocr_vvkt_match"
        )

        show_vvkt_info(
            ocr_selected
        )

        if st.button(
            "✅ Patvirtinti atpažintą preparatą",
            type="primary",
            key="confirm_first"
        ):

            st.session_state.first_vvkt_drug = (
                ocr_selected
            )

            st.success(
                f"Patvirtintas preparatas: "
                f"{ocr_selected}"
            )

    else:

        st.warning(
            "Pagal OCR informaciją VVKT kataloge "
            "tinkamo preparato automatiškai "
            "parinkti nepavyko."
        )

    # =====================================================
    # BASELINE MODELIS
    # =====================================================

    with st.expander(
        "📊 Bazinio modelio rezultatas (tyrimui)"
    ):

        st.caption(
            "HOG + Logistic Regression modelis buvo "
            "mokytas tik su 24 preparatų klasėmis. "
            "Jeigu nuotraukoje yra kitas preparatas, "
            "modelis vis tiek priskiria jį vienai iš "
            "žinomų klasių. Todėl šis rezultatas "
            "naudojamas baseline palyginimui, o ne "
            "galutiniam preparato identifikavimui."
        )

        arr = np.asarray(
            image
            .convert("L")
            .resize((128, 128)),
            dtype=np.float32
        ) / 255.0

        features = hog(
            arr,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True
        ).reshape(1, -1)

        probs = model.predict_proba(
            features
        )[0]

        top = np.argsort(
            probs
        )[::-1][:3]

        for i, idx in enumerate(
            top,
            1
        ):

            st.write(
                f"**{i}. "
                f"{model.classes_[idx]}** — "
                f"{probs[idx] * 100:.1f}% "
                "modelio tikimybės įvertis"
            )

        st.caption(
            "Šie įverčiai nėra kalibruotas "
            "pasitikėjimo matas."
        )


# =========================================================
# RANKINĖ PIRMOS VAISTO PAIEŠKA
# =========================================================

st.divider()

st.markdown(
    "### 🔎 Jei reikia – raskite pirmą preparatą rankiniu būdu"
)

first_query = st.text_input(
    "Ieškoti pirmo preparato VVKT kataloge",
    placeholder="Pvz. IBUPROM, Atacand, Celebrex..."
)

if first_query:

    first_matches = find_names(
        first_query
    )

    if first_matches:

        manual_first = st.selectbox(
            "Rasti preparatai",
            first_matches,
            key="manual_first_select"
        )

        show_vvkt_info(
            manual_first
        )

        if st.button(
            "✅ Naudoti kaip pirmą preparatą",
            key="manual_first_button"
        ):

            st.session_state.first_vvkt_drug = (
                manual_first
            )

            st.rerun()

    else:

        st.info(
            "Pagal įvestą pavadinimą "
            "VVKT preparatų nerasta."
        )


# =========================================================
# 3. PIRMO PREPARATO PATVIRTINIMAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '✅ 3. Patvirtintas pirmasis preparatas'
    '</div>',
    unsafe_allow_html=True
)

first = (
    st.session_state.first_vvkt_drug
)

if first:

    st.success(
        f"💊 {first}"
    )

    show_vvkt_info(
        first
    )

    if st.button(
        "🔄 Keisti pirmą preparatą"
    ):

        st.session_state.first_vvkt_drug = None
        st.rerun()

else:

    st.info(
        "Dar nepatvirtintas pirmasis preparatas."
    )


# =========================================================
# 4. ANTRAS PREPARATAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '💊 4. Pasirinkite antrą vaistą'
    '</div>',
    unsafe_allow_html=True
)

second_query = st.text_input(
    "Ieškokite antro preparato VVKT kataloge",
    placeholder="Pvz. NO-SPA, Celebrex, Atacand..."
)

second = None

if second_query:

    second_matches = find_names(
        second_query
    )

    if second_matches:

        second = st.selectbox(
            "Pasirinkite antrą preparatą",
            second_matches,
            key="second_vvkt_select"
        )

        show_vvkt_info(
            second
        )

    else:

        st.info(
            "Pagal įvestą pavadinimą "
            "VVKT preparatų nerasta."
        )


# =========================================================
# 5. SĄVEIKOS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '🔬 5. Kaip šie vaistai veikia kartu'
    '</div>',
    unsafe_allow_html=True
)

if st.button(
    "🔬 Analizuoti veikliąsias medžiagas",
    type="primary"
):

    if not first:

        st.warning(
            "Pirmiausia patvirtinkite "
            "pirmą preparatą."
        )

    elif not second:

        st.warning(
            "Pirmiausia pasirinkite "
            "antrą preparatą."
        )

    elif first == second:

        st.warning(
            "Pasirinkite du skirtingus preparatus."
        )

    else:

        first_row = get_vvkt_row(
            first
        )

        second_row = get_vvkt_row(
            second
        )

        first_ingredient = (
            get_vvkt_ingredient(
                first
            )
        )

        second_ingredient = (
            get_vvkt_ingredient(
                second
            )
        )

        st.subheader(
            f"💊 {first} + {second}"
        )

        # -------------------------------------------------
        # VEIKLIOSIOS MEDŽIAGOS
        # -------------------------------------------------

        st.markdown(
            "### 🧪 Veikliosios medžiagos"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"#### {first}"
            )

            st.write(
                "**Veiklioji medžiaga:**",
                first_ingredient
                or "—"
            )

            if first_row is not None:

                st.write(
                    "**Stiprumas:**",
                    clean_value(
                        first_row.get(
                            "stiprumas"
                        )
                    )
                )

                st.write(
                    "**Farmacinė forma:**",
                    clean_value(
                        first_row.get(
                            "farmacine_forma_lt"
                        )
                    )
                )

        with col2:

            st.markdown(
                f"#### {second}"
            )

            st.write(
                "**Veiklioji medžiaga:**",
                second_ingredient
                or "—"
            )

            if second_row is not None:

                st.write(
                    "**Stiprumas:**",
                    clean_value(
                        second_row.get(
                            "stiprumas"
                        )
                    )
                )

                st.write(
                    "**Farmacinė forma:**",
                    clean_value(
                        second_row.get(
                            "farmacine_forma_lt"
                        )
                    )
                )

        # -------------------------------------------------
        # PORINĖS SĄVEIKOS
        # -------------------------------------------------

        st.markdown(
            "### 🔬 Sąveikos analizė"
        )

        rules = find_interactions(
            first_ingredient,
            second_ingredient
        )

        if rules:

            for rule in rules:

                description = clean_value(
                    rule.get(
                        "description"
                    )
                )

                source = clean_value(
                    rule.get(
                        "source"
                    )
                )

                st.warning(
                    description
                )

                if source != "—":

                    st.caption(
                        "Šaltinis: "
                        + source
                    )

            st.markdown(
                "### 📋 Išvada"
            )

            st.info(
                "Šiai veikliųjų medžiagų porai "
                "prototipo bazėje yra konkreti "
                "patikrinta sąveikos taisyklė. "
                "Vertinant realų vartojimą reikia "
                "atsižvelgti ir į dozę, vartojimo "
                "būdą, kitas ligas bei kitus "
                "vartojamus vaistus."
            )

        else:

            st.info(
                "Dabartinėje prototipo sąveikų "
                "bazėje šiai veikliųjų medžiagų "
                "porai nėra įrašytos konkrečios "
                "patikrintos porinės taisyklės."
            )

            st.warning(
                "Tai nereiškia, kad preparatus "
                "saugu vartoti kartu. Ši prototipo "
                "versija neturi pakankamai duomenų "
                "automatinei medicininei išvadai "
                "apie šią porą pateikti."
            )

            st.markdown(
                "### 📋 Išvada"
            )

            st.write(
                "Norint paaiškinti, kaip šios "
                "veikliosios medžiagos veikia kartu, "
                "reikia patikrinto farmakologinio "
                "šaltinio apie jų veikimo mechanizmus "
                "ir galimas tarpusavio sąveikas. "
                "Vien VVKT registracijos duomenų tam "
                "nepakanka."
            )

        # -------------------------------------------------
        # VARTOJIMO BŪDAS
        # -------------------------------------------------

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

        # -------------------------------------------------
        # DUOMENŲ ŠALTINIAI
        # -------------------------------------------------

        st.markdown(
            "### 📚 Naudojami duomenys"
        )

        st.write(
            "• Preparato pavadinimas, veiklioji "
            "medžiaga, stiprumas, farmacinė forma "
            "ir vartojimo būdas gaunami iš projekte "
            "esančio VVKT duomenų rinkinio."
        )

        st.write(
            "• Porinė sąveikos informacija rodoma "
            "tik tada, kai tokia taisyklė yra "
            "projekto patikrintame "
            "`interactions.csv` faile."
        )


# =========================================================
# INFORMACIJA APIE MODELĮ
# =========================================================

st.divider()

with st.expander(
    "ℹ️ Apie bazinį AI modelį"
):

    st.write(
        "Bazinis HOG + Logistic Regression modelis "
        "mokytas atpažinti 24 vaistų pakuočių klases."
    )

    st.write(
        "Vaizdas konvertuojamas į pilkumo skalę "
        "ir pakeičiamas į 128×128 pikselių dydį."
    )

    st.write(
        "HOG parametrai: 9 orientacijos, "
        "8×8 pikselių ląstelės ir 2×2 blokai."
    )

    st.write(
        "Patobulintoje sistemoje galutiniam "
        "preparato identifikavimui papildomai "
        "naudojamas OCR ir VVKT registruotų "
        "preparatų duomenų rinkinys."
    )


st.caption(
    "Edukacinis AI prototipas – "
    "vaistų pakuočių atpažinimas ir "
    "sąveikų informacijos demonstravimas."
)
