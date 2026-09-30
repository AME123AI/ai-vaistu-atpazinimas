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
    return joblib.load(
        ROOT / "hog_logreg.joblib"
    )


@st.cache_data
def load_json(name):
    return json.loads(
        (ROOT / name).read_text(
            encoding="utf-8"
        )
    )


@st.cache_data
def load_interactions():
    path = ROOT / "interactions.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_vvkt():
    return pd.read_csv(
        ROOT / "PreparatasPakuote.csv",
        low_memory=False
    )


@st.cache_data
def load_ingredient_knowledge():
    path = ROOT / "ingredient_knowledge.csv"

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "ingredient",
                "group",
                "mechanism",
                "effect",
                "source"
            ]
        )

    return pd.read_csv(path)


model = load_model()

drug_data = load_json(
    "drug_profiles.json"
)

ingredient_profiles = load_json(
    "ingredient_profiles.json"
)

interactions = load_interactions()

vvkt = load_vvkt()

ingredient_knowledge = (
    load_ingredient_knowledge()
)


# =========================================================
# VVKT DUOMENŲ PARUOŠIMAS
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
    .drop_duplicates()
    .tolist()
)


# =========================================================
# SESSION STATE
# =========================================================

if "first_vvkt_drug" not in st.session_state:
    st.session_state.first_vvkt_drug = None


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

    if not value:
        return "—"

    return value


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


def get_ingredient_knowledge(ingredient):
    if not ingredient:
        return None

    if ingredient_knowledge.empty:
        return None

    if (
        "ingredient"
        not in ingredient_knowledge.columns
    ):
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
    knowledge = get_ingredient_knowledge(
        ingredient
    )

    if knowledge is None:
        st.markdown(
            f"**{drug_name} – "
            f"{ingredient or 'veiklioji medžiaga nenustatyta'}:** "
            "šios veikliosios medžiagos "
            "farmakologinis profilis dar "
            "neįtrauktas į patikrintą "
            "prototipo bazę."
        )

        return False

    group = clean_value(
        knowledge.get("group")
    )

    mechanism = clean_value(
        knowledge.get("mechanism")
    )

    effect = clean_value(
        knowledge.get("effect")
    )

    st.markdown(
        f"**{drug_name} – {ingredient}:** "
        f"{group}. {mechanism} {effect}"
    )

    return True


def show_vvkt_info(name):
    row = get_vvkt_row(name)

    if row is None:
        st.warning(
            "VVKT informacijos šiam "
            "preparatui rasti nepavyko."
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

        if len(results) >= 30:
            break

    return results


# =========================================================
# SENŲ SĄVEIKŲ TAISYKLIŲ SUSIEJIMAS
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
        if (
            normalize_text(alias)
            in ingredient_norm
        ):
            return True

    return False


def find_interactions(
    ingredient_1,
    ingredient_2
):
    found = []

    if interactions.empty:
        return found

    if (
        not ingredient_1
        or not ingredient_2
    ):
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
# OCR PAGALBINĖS FUNKCIJOS
# =========================================================

def extract_strength(text):
    text_norm = normalize_text(text)

    match = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*mg\b",
        text_norm
    )

    if not match:
        return None

    number = (
        match.group(1)
        .replace(",", ".")
    )

    return f"{number} mg"


def detect_form(text):
    text_norm = normalize_text(text)

    if (
        "minkstosios kapsules"
        in text_norm
    ):
        return "minkštosios kapsulės"

    if (
        "plevele dengtos tabletes"
        in text_norm
    ):
        return "plėvele dengtos tabletės"

    if "kapsules" in text_norm:
        return "kapsulės"

    if "tabletes" in text_norm:
        return "tabletės"

    if (
        "injekcinis tirpalas"
        in text_norm
    ):
        return "injekcinis tirpalas"

    if "sirupas" in text_norm:
        return "sirupas"

    if "gelis" in text_norm:
        return "gelis"

    return None


def detect_ingredient_from_ocr(
    ocr_text
):
    ingredient_names = sorted(
        vvkt["veiklioji_medz_lt"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    ingredient_lookup = {
        normalize_text(item): item
        for item in ingredient_names
    }

    normalized_ingredients = list(
        ingredient_lookup.keys()
    )

    ocr_normalized = normalize_text(
        ocr_text
    )

    # Pirmiausia tikslus veikliosios
    # medžiagos pavadinimo radimas.
    exact_candidates = []

    for normalized_name in (
        normalized_ingredients
    ):
        if (
            len(normalized_name) >= 5
            and normalized_name
            in ocr_normalized
        ):
            exact_candidates.append(
                normalized_name
            )

    if exact_candidates:
        best = max(
            exact_candidates,
            key=len
        )

        return ingredient_lookup[best]

    # Jei tiksliai nerasta –
    # atsargus fuzzy palyginimas.
    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if len(line.strip()) >= 4
    ]

    for line in lines:

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
                normalized_ingredients,
                n=1,
                cutoff=0.78
            )

            if matches:
                return ingredient_lookup[
                    matches[0]
                ]

    return None


def find_vvkt_candidates_from_ocr(
    ocr_text,
    ingredient,
    strength,
    form
):
    normalized_ocr = normalize_text(
        ocr_text
    )

    # -----------------------------------------------------
    # 1. TIKSLUS PREPARATO PAVADINIMAS
    # -----------------------------------------------------

    exact_names = []

    for name in vvkt_names:

        normalized_name = normalize_text(
            name
        )

        if (
            len(normalized_name) >= 4
            and normalized_name
            in normalized_ocr
        ):
            exact_names.append(name)

    exact_names = list(
        dict.fromkeys(exact_names)
    )

    exact_names = sorted(
        exact_names,
        key=len,
        reverse=True
    )

    if exact_names:
        return exact_names[:20]

    # -----------------------------------------------------
    # 2. FILTRAVIMAS PAGAL VEIKLIĄJĄ MEDŽIAGĄ
    # -----------------------------------------------------

    if ingredient:

        candidates = vvkt[
            vvkt["veiklioji_medz_lt"]
            .fillna("")
            .astype(str)
            .apply(normalize_text)
            ==
            normalize_text(ingredient)
        ].copy()

        # ---------------------------------------------
        # STIPRUMAS
        # ---------------------------------------------

        if strength:

            strength_number_match = (
                re.search(
                    r"\d+(?:[.,]\d+)?",
                    strength
                )
            )

            if strength_number_match:

                strength_number = (
                    strength_number_match
                    .group(0)
                    .replace(",", ".")
                )

                strength_candidates = (
                    candidates[
                        candidates["stiprumas"]
                        .fillna("")
                        .astype(str)
                        .apply(normalize_text)
                        .str.contains(
                            strength_number,
                            regex=False
                        )
                    ]
                )

                if (
                    not
                    strength_candidates.empty
                ):
                    candidates = (
                        strength_candidates
                    )

        # ---------------------------------------------
        # FARMACINĖ FORMA
        # ---------------------------------------------

        if form:

            form_candidates = (
                candidates[
                    candidates[
                        "farmacine_forma_lt"
                    ]
                    .fillna("")
                    .astype(str)
                    .apply(normalize_text)
                    .str.contains(
                        normalize_text(form),
                        regex=False
                    )
                ]
            )

            if not form_candidates.empty:
                candidates = (
                    form_candidates
                )

        names = (
            candidates[
                "preparato_pav"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if names:
            return names[:20]

    # -----------------------------------------------------
    # 3. FUZZY PREPARATO PAVADINIMO PAIEŠKA
    # -----------------------------------------------------

    normalized_name_map = {
        normalize_text(name): name
        for name in vvkt_names
    }

    normalized_names = list(
        normalized_name_map.keys()
    )

    fuzzy_matches = []

    for line in ocr_text.splitlines():

        line = line.strip()

        if len(line) < 4:
            continue

        matches = get_close_matches(
            normalize_text(line),
            normalized_names,
            n=5,
            cutoff=0.82
        )

        for match in matches:
            fuzzy_matches.append(
                normalized_name_map[
                    match
                ]
            )

    fuzzy_matches = list(
        dict.fromkeys(fuzzy_matches)
    )

    return fuzzy_matches[:20]


# =========================================================
# DIZAINAS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        text-align: center;
    }

    .step {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .small-note {
        opacity: 0.8;
        font-size: 0.92rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ANTRAŠTĖ
# =========================================================

st.title(
    "💊 AI vaistų atpažinimas ir "
    "sąveikų paaiškinimas"
)

st.caption(
    "Patobulintas prototipas: OCR + VVKT "
    "duomenys ir HOG + Logistic Regression "
    "bazinis modelis"
)

st.warning(
    "⚠️ Edukacinis prototipas. "
    "Pateikiama informacija nėra "
    "individuali medicininė rekomendacija "
    "ir nepakeičia gydytojo ar "
    "vaistininko konsultacijos."
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
# 2. AI ATPAŽINIMAS
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

    detected_ingredient = (
        detect_ingredient_from_ocr(
            ocr_text
        )
    )

    detected_strength = (
        extract_strength(
            ocr_text
        )
    )

    detected_form = (
        detect_form(
            ocr_text
        )
    )

    ocr_suggestions = (
        find_vvkt_candidates_from_ocr(
            ocr_text,
            detected_ingredient,
            detected_strength,
            detected_form
        )
    )

    # -----------------------------------------------------
    # VARTOTOJUI RODOME AIŠKIĄ INFORMACIJĄ
    # -----------------------------------------------------

    st.markdown(
        "#### 🧾 Iš pakuotės atpažinta informacija"
    )

    if detected_ingredient:
        st.write(
            "**Veiklioji medžiaga:**",
            detected_ingredient
        )
    else:
        st.write(
            "**Veiklioji medžiaga:** "
            "automatiškai nenustatyta"
        )

    if detected_strength:
        st.write(
            "**Stiprumas:**",
            detected_strength
        )

    if detected_form:
        st.write(
            "**Farmacinė forma:**",
            detected_form
        )

    # Neapdorotą OCR slepiame.
    with st.expander(
        "🔧 Techninė OCR informacija"
    ):
        st.caption(
            "Šis tekstas skirtas prototipo "
            "testavimui. OCR gali turėti klaidų."
        )

        st.text(
            ocr_text
            if ocr_text.strip()
            else "Teksto atpažinti nepavyko."
        )

    # -----------------------------------------------------
    # VVKT KANDIDATAI
    # -----------------------------------------------------

    if ocr_suggestions:

        st.markdown(
            "#### 🔎 Galimi VVKT preparatai"
        )

        if len(ocr_suggestions) == 1:

            st.success(
                "Pagal pakuotės informaciją "
                "rastas vienas galimas preparatas."
            )

        else:

            st.info(
                f"Pagal pakuotės informaciją "
                f"rasti {len(ocr_suggestions)} "
                "galimi preparatai. "
                "Pasirinkite tą, kuris atitinka "
                "pakuotę."
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
            "✅ Patvirtinti preparatą",
            type="primary",
            key="confirm_first"
        ):

            st.session_state[
                "first_vvkt_drug"
            ] = ocr_selected

            st.success(
                f"Patvirtintas preparatas: "
                f"{ocr_selected}"
            )

    else:

        st.info(
            "Automatiškai tinkamo VVKT "
            "preparato pasirinkti nepavyko. "
            "Preparatą galite rasti rankine "
            "paieška žemiau."
        )

    # =====================================================
    # BASELINE
    # =====================================================

    with st.expander(
        "📊 Bazinio modelio rezultatas (tyrimui)"
    ):

        st.caption(
            "HOG + Logistic Regression modelis "
            "mokytas tik su 24 preparatų klasėmis. "
            "Jeigu nuotraukoje yra preparatas, "
            "kurio tarp šių klasių nėra, modelis "
            "vis tiek priskiria vaizdą vienai iš "
            "24 žinomų klasių. Todėl šis rezultatas "
            "naudojamas tik baseline palyginimui."
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
            start=1
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
# RANKINĖ PIRMO VAISTO PAIEŠKA
# =========================================================

st.divider()

st.markdown(
    "### 🔎 Jei reikia – raskite pirmą "
    "preparatą rankiniu būdu"
)

first_query = st.text_input(
    "Ieškoti pirmo preparato VVKT kataloge",
    placeholder=(
        "Pvz. IBUPROM, Atacand, Celebrex..."
    )
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

            st.session_state[
                "first_vvkt_drug"
            ] = manual_first

            st.rerun()

    else:

        st.info(
            "Pagal įvestą pavadinimą "
            "VVKT preparatų nerasta."
        )


# =========================================================
# 3. PATVIRTINTAS PIRMAS PREPARATAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '✅ 3. Patvirtintas pirmasis preparatas'
    '</div>',
    unsafe_allow_html=True
)

first = st.session_state.get(
    "first_vvkt_drug"
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

        st.session_state[
            "first_vvkt_drug"
        ] = None

        st.rerun()

else:

    st.info(
        "Dar nepatvirtintas "
        "pirmasis preparatas."
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
    placeholder=(
        "Pvz. NO-SPA, Celebrex, Atacand..."
    )
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
# 5. KAIP ŠIE VAISTAI VEIKIA KARTU
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
            "Pasirinkite du skirtingus "
            "preparatus."
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

        # =================================================
        # ANTRAŠTĖ
        # =================================================

        st.subheader(
            f"💊 {first} + {second}"
        )

        # =================================================
        # VEIKLIOSIOS MEDŽIAGOS
        # =================================================

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
                first_ingredient or "—"
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
                second_ingredient or "—"
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

        # =================================================
        # KAIP VEIKIA KARTU
        # =================================================

        st.markdown(
            "### 🔬 KAIP ŠIE VAISTAI VEIKIA KARTU"
        )

        known_first = (
            show_ingredient_explanation(
                first,
                first_ingredient
            )
        )

        known_second = (
            show_ingredient_explanation(
                second,
                second_ingredient
            )
        )

        first_info = (
            get_ingredient_knowledge(
                first_ingredient
            )
        )

        second_info = (
            get_ingredient_knowledge(
                second_ingredient
            )
        )

        # =================================================
        # BENDRAS POVEIKIS
        # =================================================

        if (
            first_info is not None
            and second_info is not None
        ):

            st.markdown(
                "#### Bendras poveikis"
            )

            first_effect = clean_value(
                first_info.get(
                    "effect"
                )
            )

            second_effect = clean_value(
                second_info.get(
                    "effect"
                )
            )

            st.write(
                f"**{first_ingredient}** ir "
                f"**{second_ingredient}** veikia "
                "skirtingais farmakologiniais "
                "mechanizmais. "
                f"{first} ({first_ingredient}): "
                f"{first_effect} "
                f"{second} ({second_ingredient}): "
                f"{second_effect}"
            )

        # =================================================
        # KONKRETI PORINĖ SĄVEIKA
        # =================================================

        rules = find_interactions(
            first_ingredient,
            second_ingredient
        )

        if rules:

            st.markdown(
                "#### ⚠️ Žinoma porinė sąveika"
            )

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

        else:

            st.caption(
                "Šiai konkrečiai veikliųjų "
                "medžiagų porai prototipo "
                "patikrintų porinių sąveikų "
                "bazėje atskira taisyklė "
                "neįrašyta. Tai nėra teiginys, "
                "kad derinį saugu vartoti kartu."
            )

        # =================================================
        # IŠVADA
        # =================================================

        st.markdown(
            "#### 📋 Išvada"
        )

        if known_first and known_second:

            st.info(
                "Pagal veikimo mechanizmus šių "
                "veikliųjų medžiagų poveikiai yra "
                "skirtingi ir gali būti susiję su "
                "skirtingomis simptomų grandimis. "
                "Vien veikimo mechanizmų palyginimas "
                "nepatvirtina, kad konkretų derinį "
                "saugu vartoti kartu. Vertinant "
                "vartojimą reikia atsižvelgti į "
                "dozes, vartojimo būdą, "
                "kontraindikacijas, kitus vartojamus "
                "vaistus ir konkrečių preparatų "
                "informaciją."
            )

        else:

            st.warning(
                "Bent vienos veikliosios medžiagos "
                "patikrinto farmakologinio profilio "
                "prototipo bazėje dar nėra. Todėl "
                "farmakologinė išvada šiai porai "
                "nepateikiama."
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
        # ŠALTINIAI
        # =================================================

        st.markdown(
            "### 📚 Šaltiniai"
        )

        shown_sources = set()

        for ingredient in [
            first_ingredient,
            second_ingredient
        ]:

            info = (
                get_ingredient_knowledge(
                    ingredient
                )
            )

            if info is None:
                continue

            source = clean_value(
                info.get(
                    "source"
                )
            )

            if (
                source != "—"
                and source
                not in shown_sources
            ):

                st.write(
                    f"• {source}"
                )

                shown_sources.add(
                    source
                )

        if not shown_sources:

            st.write(
                "• VVKT registruotų vaistinių "
                "preparatų duomenų rinkinys."
            )

        st.caption(
            "Preparato pavadinimas, veiklioji "
            "medžiaga, stiprumas, farmacinė forma "
            "ir vartojimo būdas gaunami iš projekte "
            "esančio VVKT duomenų rinkinio. "
            "Farmakologiniai aprašymai gaunami iš "
            "`ingredient_knowledge.csv`, o konkrečios "
            "porinės sąveikos – iš "
            "`interactions.csv`."
        )


# =========================================================
# INFORMACIJA APIE MODELĮ
# =========================================================

st.divider()

with st.expander(
    "ℹ️ Apie bazinį AI modelį"
):

    st.write(
        "Bazinis HOG + Logistic Regression "
        "modelis mokytas atpažinti "
        "24 vaistų pakuočių klases."
    )

    st.write(
        "Vaizdas konvertuojamas į pilkumo "
        "skalę ir pakeičiamas į "
        "128×128 pikselių dydį."
    )

    st.write(
        "HOG parametrai: 9 orientacijos, "
        "8×8 pikselių ląstelės ir "
        "2×2 blokai."
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
