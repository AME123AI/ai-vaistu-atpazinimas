from pathlib import Path
import json
import re
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
    return pd.read_csv(ROOT / "interactions.csv")


@st.cache_data
def load_vvkt():
    return pd.read_csv(
        ROOT / "PreparatasPakuote.csv",
        low_memory=False
    )


model = load_model()
classes = list(model.classes_)

drug_data = load_json("drug_profiles.json")
ingredient_profiles = load_json("ingredient_profiles.json")
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
# PAGALBINĖS FUNKCIJOS
# =========================================================

def clean_value(value):
    if pd.isna(value):
        return "—"

    value = str(value).strip()

    if not value:
        return "—"

    return value


def normalize_text(text):
    return (
        str(text)
        .lower()
        .replace("ė", "e")
        .replace("ę", "e")
        .replace("ą", "a")
        .replace("č", "c")
        .replace("š", "s")
        .replace("ų", "u")
        .replace("ū", "u")
        .replace("ž", "z")
        .replace("į", "i")
    )


def get_vvkt_rows(name):
    return vvkt[
        vvkt["preparato_pav"] == name
    ]


def get_vvkt_row(name):
    rows = get_vvkt_rows(name)

    if rows.empty:
        return None

    return rows.iloc[0]


def get_ingredient(name):
    row = get_vvkt_row(name)

    if row is None:
        return None

    value = row.get(
        "veiklioji_medz_lt",
        None
    )

    if pd.isna(value):
        return None

    value = str(value).strip()

    return value if value else None


def show_vvkt_info(name):
    row = get_vvkt_row(name)

    if row is None:
        st.warning(
            "Šio preparato VVKT informacijos rasti nepavyko."
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


# Senų 24 klasių pavadinimų susiejimas su VVKT veikliosiomis
# medžiagomis. Tai leidžia esamas interactions.csv taisykles
# panaudoti VVKT pasirinktiems preparatams, kai veiklioji
# medžiaga sutampa.

LEGACY_INGREDIENT_ALIASES = {
    "Aggrex": [
        "acetilsalicilo"
    ],
    "CELEBREX": [
        "celekoksib"
    ],
    "Candalkan": [
        "kandesartan"
    ],
}


def ingredient_matches_legacy_drug(
    ingredient,
    legacy_drug
):
    if not ingredient:
        return False

    ingredient_norm = normalize_text(
        ingredient
    )

    aliases = LEGACY_INGREDIENT_ALIASES.get(
        legacy_drug,
        []
    )

    for alias in aliases:
        if normalize_text(alias) in ingredient_norm:
            return True

    return False


def find_interaction_rules(
    ingredient_a,
    ingredient_b
):
    results = []

    if not ingredient_a or not ingredient_b:
        return results

    for _, rule in interactions.iterrows():

        drug_a = str(
            rule.get("drug_a", "")
        ).strip()

        drug_b = str(
            rule.get("drug_b", "")
        ).strip()

        direct = (
            ingredient_matches_legacy_drug(
                ingredient_a,
                drug_a
            )
            and
            ingredient_matches_legacy_drug(
                ingredient_b,
                drug_b
            )
        )

        reverse = (
            ingredient_matches_legacy_drug(
                ingredient_a,
                drug_b
            )
            and
            ingredient_matches_legacy_drug(
                ingredient_b,
                drug_a
            )
        )

        if direct or reverse:
            results.append(rule)

    return results


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
        font-size: 1.1rem;
        font-weight: 700;
    }

    .small {
        opacity: .75;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title(
    "💊 AI vaistų atpažinimas ir sąveikų paaiškinimas"
)

st.caption(
    "Edukacinis prototipas: HOG + Logistic Regression "
    "bazinis modelis, OCR ir VVKT duomenys"
)

st.warning(
    "⚠️ Edukacinis prototipas. Informacija nėra "
    "individuali medicininė rekomendacija ir "
    "nepakeičia gydytojo ar vaistininko konsultacijos."
)


# =========================================================
# SESSION STATE
# =========================================================

if "first_drug" not in st.session_state:
    st.session_state.first_drug = None

if "first_vvkt_drug" not in st.session_state:
    st.session_state.first_vvkt_drug = None


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
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# OCR + HOG
# =========================================================

if uploaded:

    image = Image.open(
        uploaded
    ).convert("RGB")

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    ocr_text = pytesseract.image_to_string(
        image
    )

    with st.expander(
        "🔤 OCR nuskaitytas tekstas"
    ):
        st.text(
            ocr_text
            if ocr_text.strip()
            else "Teksto atpažinti nepavyko."
        )

    ocr_text_lower = ocr_text.lower()
    normalized_ocr = normalize_text(
        ocr_text
    )

    ocr_suggestions = []

    detected_ingredient = None
    detected_strength = None
    detected_form = None

    # -----------------------------------------------------
    # A. TIKSLUS PREPARATO PAVADINIMAS
    # -----------------------------------------------------

    exact_matches = []

    for name in vvkt_names:

        if name.lower() in ocr_text_lower:
            exact_matches.append(name)

    exact_matches = sorted(
        exact_matches,
        key=len,
        reverse=True
    )

    if exact_matches:
        ocr_suggestions = exact_matches

    # -----------------------------------------------------
    # B. VEIKLIOJI MEDŽIAGA
    # -----------------------------------------------------

    if not ocr_suggestions:

        ingredient_names = sorted(
            vvkt["veiklioji_medz_lt"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )

        ingredient_lookup = {
            x.lower(): x
            for x in ingredient_names
        }

        ingredient_lower = list(
            ingredient_lookup.keys()
        )

        ocr_lines = [
            line.strip()
            for line in ocr_text.splitlines()
            if len(line.strip()) >= 4
        ]

        for line in ocr_lines:

            search_parts = [line]

            search_parts.extend(
                word
                for word in line.split()
                if len(word) >= 5
            )

            for part in search_parts:

                matches = get_close_matches(
                    part.lower(),
                    ingredient_lower,
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
    # C. STIPRUMAS
    # -----------------------------------------------------

    strength_match = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*mg\b",
        ocr_text_lower
    )

    if strength_match:

        detected_strength = (
            strength_match
            .group(1)
            .replace(",", ".")
            + " mg"
        )

    # -----------------------------------------------------
    # D. FARMACINĖ FORMA
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

    # -----------------------------------------------------
    # E. VVKT FILTRAVIMAS
    # -----------------------------------------------------

    if (
        not ocr_suggestions
        and detected_ingredient
    ):

        candidates = vvkt[
            vvkt["veiklioji_medz_lt"]
            .fillna("")
            .astype(str)
            .str.lower()
            ==
            detected_ingredient.lower()
        ].copy()

        if detected_strength:

            strength_candidates = candidates[
                candidates["stiprumas"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    detected_strength.lower(),
                    regex=False
                )
            ]

            if not strength_candidates.empty:
                candidates = strength_candidates

        if detected_form == "plėvele dengtos tabletės":

            form_candidates = candidates[
                candidates["farmacine_forma_lt"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    "plėvele dengtos tabletės",
                    regex=False
                )
            ]

            if not form_candidates.empty:
                candidates = form_candidates

        elif detected_form == "tabletės":

            form_candidates = candidates[
                candidates["farmacine_forma_lt"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    "tablet",
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
    # F. ATSARGINĖ FUZZY PAIEŠKA
    # -----------------------------------------------------

    if not ocr_suggestions:

        fuzzy_matches = []

        for line in ocr_text.splitlines():

            line = line.strip()

            if len(line) < 5:
                continue

            matches = get_close_matches(
                line,
                vvkt_names,
                n=5,
                cutoff=0.80
            )

            fuzzy_matches.extend(
                matches
            )

        ocr_suggestions = fuzzy_matches

    ocr_suggestions = list(
        dict.fromkeys(
            ocr_suggestions
        )
    )[:20]

    # -----------------------------------------------------
    # OCR APTIKTA INFORMACIJA
    # -----------------------------------------------------

    detected_parts = []

    if detected_ingredient:
        detected_parts.append(
            f"Veiklioji medžiaga: "
            f"{detected_ingredient}"
        )

    if detected_strength:
        detected_parts.append(
            f"Stiprumas: "
            f"{detected_strength}"
        )

    if detected_form:
        detected_parts.append(
            f"Farmacinė forma: "
            f"{detected_form}"
        )

    if detected_parts:

        st.markdown(
            "### 🧾 OCR aptiko"
        )

        for part in detected_parts:
            st.write(
                "• " + part
            )

    # -----------------------------------------------------
    # VVKT KANDIDATAI
    # -----------------------------------------------------

    if ocr_suggestions:

        st.markdown(
            "### 🔎 VVKT atitinkantys preparatai"
        )

        if len(ocr_suggestions) == 1:

            st.success(
                "Pagal OCR informaciją rastas "
                "1 atitinkantis VVKT preparatas."
            )

        else:

            st.info(
                f"Pagal OCR informaciją rasti "
                f"{len(ocr_suggestions)} galimi "
                "VVKT preparatai. "
                "Pasirinkite preparatą pagal pakuotę."
            )

        ocr_selected = st.selectbox(
            "Patvirtinkite preparatą",
            ocr_suggestions,
            key="ocr_vvkt_match"
        )

        if st.button(
            "✅ Patvirtinti šį VVKT preparatą",
            key="confirm_ocr_vvkt"
        ):

            st.session_state.first_vvkt_drug = (
                ocr_selected
            )

            st.success(
                f"Patvirtintas preparatas: "
                f"{ocr_selected}"
            )

        st.markdown(
            "#### 💊 Pasirinkto kandidato VVKT informacija"
        )

        show_vvkt_info(
            ocr_selected
        )

    else:

        st.info(
            "Pagal OCR nuskaitytą informaciją "
            "VVKT kataloge tinkamų preparatų nerasta."
        )

    # -----------------------------------------------------
    # HOG + LOGISTIC REGRESSION BASELINE
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        "### 🤖 2. Bazinis AI modelis"
    )

    st.caption(
        "HOG + Logistic Regression modelis "
        "atpažįsta tik 24 mokymo rinkinio klases. "
        "Jeigu įkeltas preparatas nėra tarp šių "
        "24 klasių, modelio Top-1 rezultatas "
        "neturėtų būti interpretuojamas kaip "
        "patikimas preparato identifikavimas."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            image,
            caption="Įkelta pakuotė",
            use_container_width=True
        )

    with c2:

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

        suggested = (
            model.classes_[
                top[0]
            ]
        )

        if st.button(
            "Patvirtinti bazinio modelio Top-1"
        ):

            st.session_state.first_drug = (
                suggested
            )


# =========================================================
# RANKINĖ VVKT PAIEŠKA
# =========================================================

st.divider()

st.markdown(
    "### 🔎 Rankinė paieška VVKT vaistų kataloge"
)

vvkt_query = st.text_input(
    "Įveskite vaisto pavadinimą",
    placeholder="Pvz. Atacand"
)

if vvkt_query:

    matches = [
        name
        for name in vvkt_names
        if vvkt_query.lower()
        in name.lower()
    ][:20]

    if matches:

        selected_vvkt = st.selectbox(
            "Rasti preparatai",
            matches,
            key="manual_vvkt"
        )

        show_vvkt_info(
            selected_vvkt
        )

        if st.button(
            "✅ Naudoti kaip pirmąjį preparatą",
            key="manual_first"
        ):

            st.session_state.first_vvkt_drug = (
                selected_vvkt
            )

            st.success(
                f"Pasirinktas preparatas: "
                f"{selected_vvkt}"
            )

    else:

        st.info(
            "Pagal įvestą pavadinimą "
            "preparatų nerasta."
        )


# =========================================================
# 3. PIRMO VAISTO PATVIRTINIMAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '✅ 3. Patvirtinkite pirmą preparatą'
    '</div>',
    unsafe_allow_html=True
)

first = None

if st.session_state.first_vvkt_drug:

    first = (
        st.session_state.first_vvkt_drug
    )

    st.success(
        f"Pasirinktas pirmasis preparatas: "
        f"{first}"
    )

    show_vvkt_info(
        first
    )

    if st.button(
        "🔄 Pasirinkti kitą pirmąjį preparatą"
    ):

        st.session_state.first_vvkt_drug = None
        st.rerun()

else:

    st.info(
        "Įkelkite pakuotės nuotrauką ir "
        "patvirtinkite VVKT kandidatą arba "
        "naudokite rankinę VVKT paiešką."
    )


# =========================================================
# 4. ANTRAS VAISTAS
# =========================================================

st.markdown(
    '<div class="step">'
    '💊 4. Pasirinkite antrą vaistą'
    '</div>',
    unsafe_allow_html=True
)

second_query = st.text_input(
    "Ieškokite antro vaisto VVKT kataloge",
    placeholder="Pvz. No-Spa, Celebrex, Atacand..."
)

second = None

if second_query:

    second_matches = [
        name
        for name in vvkt_names
        if second_query.lower()
        in name.lower()
    ][:20]

    if second_matches:

        second = st.selectbox(
            "Pasirinkite antrą preparatą",
            second_matches,
            key="second_vvkt_drug"
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
    '🔬 5. Patikrinkite sąveikos informaciją'
    '</div>',
    unsafe_allow_html=True
)

if st.button(
    "🔬 Patikrinti sąveiką",
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

        st.error(
            "Pasirinkite du skirtingus preparatus."
        )

    else:

        first_ingredient = get_ingredient(
            first
        )

        second_ingredient = get_ingredient(
            second
        )

        first_row = get_vvkt_row(
            first
        )

        second_row = get_vvkt_row(
            second
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

        st.write(
            f"**{first}:** "
            f"{first_ingredient or '—'}"
        )

        st.write(
            f"**{second}:** "
            f"{second_ingredient or '—'}"
        )

        # -------------------------------------------------
        # SĄVEIKŲ TAISYKLĖS
        # -------------------------------------------------

        st.markdown(
            "### 🔬 Kaip šie vaistai veikia kartu"
        )

        found_rules = find_interaction_rules(
            first_ingredient,
            second_ingredient
        )

        if found_rules:

            for rule in found_rules:

                st.warning(
                    str(
                        rule.get(
                            "description",
                            ""
                        )
                    )
                )

                source = str(
                    rule.get(
                        "source",
                        ""
                    )
                ).strip()

                if source:
                    st.caption(
                        "Šaltinis: "
                        + source
                    )

        else:

            st.info(
                "Šiai veikliųjų medžiagų porai "
                "dabartinėje prototipo sąveikų "
                "bazėje nėra įrašytos patikrintos "
                "porinės taisyklės. Tai nėra "
                "teiginys, kad šį derinį saugu "
                "vartoti kartu."
            )

            st.caption(
                "Prototipo sąveikų bazė šiuo metu "
                "yra ribota. Sąveikos išvada "
                "neformuojama vien iš to, kad "
                "taisyklė nerasta."
            )

        # -------------------------------------------------
        # VARTOJIMO BŪDAS
        # -------------------------------------------------

        st.markdown(
            "### 💉 Vartojimo būdas"
        )

        first_route = (
            clean_value(
                first_row.get(
                    "vartojimo_budas"
                )
            )
            if first_row is not None
            else "—"
        )

        second_route = (
            clean_value(
                second_row.get(
                    "vartojimo_budas"
                )
            )
            if second_row is not None
            else "—"
        )

        st.write(
            f"**{first}:** "
            f"{first_route}"
        )

        st.write(
            f"**{second}:** "
            f"{second_route}"
        )

        # -------------------------------------------------
        # DUOMENŲ ŠALTINIO PAAIŠKINIMAS
        # -------------------------------------------------

        st.markdown(
            "### 📚 Duomenų interpretacija"
        )

        st.write(
            "Preparato pavadinimas, veiklioji "
            "medžiaga, stiprumas, farmacinė forma "
            "ir vartojimo būdas gaunami iš "
            "projekte naudojamo VVKT duomenų rinkinio."
        )

        st.write(
            "Konkreti sąveikos informacija rodoma "
            "tik tada, kai prototipo "
            "`interactions.csv` faile yra "
            "atitinkama patikrinta taisyklė."
        )


# =========================================================
# APAČIA
# =========================================================

st.divider()

st.caption(
    "Bazinis modelis: 24 klasės. "
    "HOG apdorojimas: 128×128, "
    "9 orientacijos, 8×8 pikselių ląstelė, "
    "2×2 blokas. OCR ir VVKT paieška naudojami "
    "patobulintame atpažinimo etape."
)
