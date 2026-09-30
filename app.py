from pathlib import Path
import json
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
from skimage.feature import hog


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
    "Edukacinis HOG + Logistic Regression prototipas "
    "su OCR ir VVKT duomenimis"
)

st.warning(
    "⚠️ Edukacinis prototipas. Informacija nėra "
    "individuali medicininė rekomendacija ir "
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
    type=["jpg", "jpeg", "png"]
)


if "first_drug" not in st.session_state:
    st.session_state.first_drug = None
if "first_vvkt_drug" not in st.session_state:
    st.session_state.first_vvkt_drug = None
# =========================================================
# OCR + HOG ATPAŽINIMAS
# =========================================================

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    # -------------------------
    # OCR
    # -------------------------

    ocr_text = pytesseract.image_to_string(image)

    with st.expander("🔤 OCR nuskaitytas tekstas"):
        st.text(
            ocr_text
            if ocr_text.strip()
            else "Teksto atpažinti nepavyko."
        )

    # =====================================================
    # OCR TEKSTO PALYGINIMAS SU VVKT DUOMENIMIS
    # =====================================================

    ocr_text_lower = ocr_text.lower()

    # Normalizuojame OCR tekstą paprastesniam palyginimui
    normalized_ocr = (
        ocr_text_lower
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

    ocr_suggestions = []

    # ---------------------------------
    # 1. TIKSLUS PREPARATO PAVADINIMAS
    # ---------------------------------

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

    # ---------------------------------
    # 2. VEIKLIOJI MEDŽIAGA
    # ---------------------------------

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

        detected_ingredient = None

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
                        ingredient_lookup[matches[0]]
                    )
                    break

            if detected_ingredient:
                break

        # ---------------------------------
        # 3. FILTRUOJAME PAGAL VVKT
        # ---------------------------------

        if detected_ingredient:

            candidates = vvkt[
                vvkt["veiklioji_medz_lt"]
                .fillna("")
                .astype(str)
                .str.lower()
                == detected_ingredient.lower()
            ].copy()

            # ---------------------------------
            # STIPRUMAS
            # ---------------------------------

            import re

            strength_match = re.search(
                r"\b(\d+(?:[.,]\d+)?)\s*mg\b",
                ocr_text_lower
            )

            detected_strength = None

            if strength_match:
                detected_strength = (
                    strength_match.group(1)
                    .replace(",", ".")
                    + " mg"
                )

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

            # ---------------------------------
            # FARMACINĖ FORMA
            # ---------------------------------

            if (
                "plevele dengtos tabletes"
                in normalized_ocr
            ):

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

            elif "tabletes" in normalized_ocr:

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

            # ---------------------------------
            # GALUTINIAI PASIŪLYMAI
            # ---------------------------------

            ocr_suggestions = (
                candidates["preparato_pav"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

    # ---------------------------------
    # 4. ATSARGINĖ PAVADINIMO PAIEŠKA
    # ---------------------------------

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

            fuzzy_matches.extend(matches)

        ocr_suggestions = fuzzy_matches
       # =====================================================
    # PARODOME, KĄ OCR PAVYKO ATPAŽINTI
    # =====================================================

    detected_parts = []

    if "detected_ingredient" in locals() and detected_ingredient:
        detected_parts.append(
            f"Veiklioji medžiaga: {detected_ingredient}"
        )

    if "detected_strength" in locals() and detected_strength:
        detected_parts.append(
            f"Stiprumas: {detected_strength}"
        )

    if "plevele dengtos tabletes" in normalized_ocr:
        detected_parts.append(
            "Farmacinė forma: plėvele dengtos tabletės"
        )
    elif "tabletes" in normalized_ocr:
        detected_parts.append(
            "Farmacinė forma: tabletės"
        )

    if detected_parts:
        st.markdown("### 🧾 OCR aptiko")

        for part in detected_parts:
            st.write("• " + part)

    # Pašaliname pasikartojančius kandidatus
    ocr_suggestions = list(
        dict.fromkeys(ocr_suggestions)
    )

    # Rodome daugiausia 20 kandidatų
    ocr_suggestions = ocr_suggestions[:20]

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
                f"{len(ocr_suggestions)} galimi VVKT preparatai. "
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
            st.session_state.first_vvkt_drug = ocr_selected
            st.success(
                f"Patvirtintas preparatas: {ocr_selected}"
            )
        selected_rows = vvkt[
            vvkt["preparato_pav"] == ocr_selected
        ]

        if not selected_rows.empty:
            row = selected_rows.iloc[0]

            st.markdown(
                "#### 💊 Patvirtinto preparato VVKT informacija"
            )

            st.write(
                "**Pavadinimas:**",
                row.get("preparato_pav", "—")
            )

            st.write(
                "**Veiklioji medžiaga:**",
                row.get("veiklioji_medz_lt", "—")
            )

            st.write(
                "**Stiprumas:**",
                row.get("stiprumas", "—")
            )

            st.write(
                "**Farmacinė forma:**",
                row.get("farmacine_forma_lt", "—")
            )

            st.write(
                "**Vartojimo būdas:**",
                row.get("vartojimo_budas", "—")
            )

            st.write(
                "**Recepto poreikis:**",
                row.get("recepto_poreikis", "—")
            )

    else:
        st.info(
            "Pagal OCR nuskaitytą informaciją "
            "VVKT kataloge tinkamų preparatų nerasta."
        )
        # -------------------------
    # HOG + LOGISTIC REGRESSION
    # -------------------------

    c1, c2 = st.columns(2)

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            image,
            caption="Įkelta pakuotė",
            use_container_width=True
        )


    with c2:

        arr = np.asarray(
            image.convert("L").resize((128, 128)),
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

        probs = model.predict_proba(features)[0]

        top = np.argsort(
            probs
        )[::-1][:3]

        st.markdown(
            "### 🤖 2. Bazinis AI modelis"
        )

        st.caption(
            "HOG + Logistic Regression modelis "
            "atpažįsta tik 24 mokymo rinkinio klases."
        )

        for i, idx in enumerate(top, 1):

            st.write(
                f"**{i}. {model.classes_[idx]}** — "
                f"{probs[idx] * 100:.1f}% "
                f"modelio tikimybės įvertis"
            )

        suggested = model.classes_[top[0]]

        if st.button("Patvirtinti Top-1"):

            st.session_state.first_drug = suggested


# =========================================================
# VVKT RANKINĖ PAIEŠKA
# =========================================================

st.divider()

st.markdown(
    "### 🔎 Patobulinta paieška VVKT vaistų kataloge"
)

vvkt_query = st.text_input(
    "Įveskite vaisto pavadinimą",
    placeholder="Pvz. Atacand"
)


if vvkt_query:

    matches = [
        name
        for name in vvkt_names
        if vvkt_query.lower() in name.lower()
    ][:20]


    if matches:

        selected_vvkt = st.selectbox(
            "Rasti preparatai",
            matches
        )

        selected_rows = vvkt[
            vvkt["preparato_pav"] == selected_vvkt
        ]


        if not selected_rows.empty:

            row = selected_rows.iloc[0]

            st.markdown(
                "#### 💊 Preparato informacija"
            )

            st.write(
                "**Pavadinimas:**",
                row.get("preparato_pav", "—")
            )

            st.write(
                "**Veiklioji medžiaga:**",
                row.get("veiklioji_medz_lt", "—")
            )

            st.write(
                "**Stiprumas:**",
                row.get("stiprumas", "—")
            )

            st.write(
                "**Farmacinė forma:**",
                row.get("farmacine_forma_lt", "—")
            )

            st.write(
                "**Vartojimo būdas:**",
                row.get("vartojimo_budas", "—")
            )

            st.write(
                "**Recepto poreikis:**",
                row.get("recepto_poreikis", "—")
            )

    else:

        st.info(
            "Pagal įvestą pavadinimą preparatų nerasta."
        )


# =========================================================
# 3. PIRMO VAISTO PATVIRTINIMAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '✅ 3. Patvirtinkite arba pataisykite rezultatą'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.first_vvkt_drug:

    first = st.session_state.first_vvkt_drug

    st.success(
        f"Pasirinktas pirmasis preparatas: {first}"
    )

    first_rows = vvkt[
        vvkt["preparato_pav"] == first
    ]

    if not first_rows.empty:
        first_row = first_rows.iloc[0]

        st.write(
            "**Veiklioji medžiaga:**",
            first_row.get("veiklioji_medz_lt", "—")
        )

        st.write(
            "**Stiprumas:**",
            first_row.get("stiprumas", "—")
        )

        st.write(
            "**Farmacinė forma:**",
            first_row.get("farmacine_forma_lt", "—")
        )

    if st.button(
        "🔄 Pasirinkti kitą pirmąjį preparatą"
    ):
        st.session_state.first_vvkt_drug = None
        st.rerun()

else:

    st.info(
        "Pirmiausia įkelkite pakuotės nuotrauką "
        "ir patvirtinkite vieną iš VVKT pasiūlytų preparatų."
    )

    default_index = (
        classes.index(st.session_state.first_drug)
        if st.session_state.first_drug in classes
        else 0
    )

    baseline_first = st.selectbox(
        "Arba pasirinkite bazinio modelio preparatą",
        classes,
        index=default_index
    )

    first = baseline_first
    st.session_state.first_drug = baseline_first

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
    placeholder="Pvz. Celebrex, Atacand, Diflucan..."
)

second = None

if second_query:

    second_matches = [
        name
        for name in vvkt_names
        if second_query.lower() in name.lower()
    ][:20]

    if second_matches:

        second = st.selectbox(
            "Pasirinkite antrą preparatą",
            second_matches,
            key="second_vvkt_drug"
        )

        second_rows = vvkt[
            vvkt["preparato_pav"] == second
        ]

        if not second_rows.empty:
            second_row = second_rows.iloc[0]

            st.write(
                "**Veiklioji medžiaga:**",
                second_row.get("veiklioji_medz_lt", "—")
            )

            st.write(
                "**Stiprumas:**",
                second_row.get("stiprumas", "—")
            )

            st.write(
                "**Farmacinė forma:**",
                second_row.get("farmacine_forma_lt", "—")
            )

    else:
        st.info(
            "Pagal įvestą pavadinimą VVKT preparatų nerasta."
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


def route_text(routes):
    return (
        ", ".join(routes)
        if routes
        else "vartojimo būdas neįvestas"
    )


def show_drug(name):

    d = drug_data.get(name, {})

    st.markdown(f"**{name}**")

    st.write(
        "Vartojimo būdas:",
        route_text(d.get("routes", []))
    )

    ings = d.get("ingredients", [])

    if not ings:

        st.info(
            "Šio preparato veikliųjų medžiagų profilį "
            "dar reikia patikrinti pagal konkretaus "
            "preparato informacinį lapelį."
        )

        return


    for ing in ings:

        p = ingredient_profiles.get(ing)

        if p:

            st.write(
                f"• **{ing}** — "
                f"{p['group']}. {p['effect']}"
            )

        else:

            st.write(
                f"• **{ing}**"
            )


if st.button(
    "🔬 Patikrinti sąveiką",
    type="primary"
):

    if not second:
        st.warning(
            "Pirmiausia pasirinkite antrą preparatą."
        )

    elif first == second:

        st.error(
            "Pasirinkite du skirtingus preparatus."
        )

    else:

        st.subheader(
            f"💊 {first} + {second}"
        )

        st.markdown(
            "### 🧪 Veikliosios medžiagos"
        )

        a = drug_data.get(first, {})
        b = drug_data.get(second, {})

        show_drug(first)
        show_drug(second)


        st.markdown(
            "### 🔬 Kaip šie vaistai veikia kartu"
        )

        mask = (
            (
                (interactions.drug_a == first)
                & (interactions.drug_b == second)
            )
            |
            (
                (interactions.drug_a == second)
                & (interactions.drug_b == first)
            )
        )

        found = interactions[mask]


        if not found.empty:

            for _, row in found.iterrows():

                st.warning(
                    row["description"]
                )

                st.caption(
                    "Šaltinis: " + row["source"]
                )

        else:

            ings1 = a.get("ingredients", [])
            ings2 = b.get("ingredients", [])

            known1 = [
                ingredient_profiles[x]["effect"]
                for x in ings1
                if x in ingredient_profiles
            ]

            known2 = [
                ingredient_profiles[x]["effect"]
                for x in ings2
                if x in ingredient_profiles
            ]

            st.info(
                "Šiai porai nėra įrašytos konkrečios "
                "porinės taisyklės mūsų prototipo bazėje. "
                "Toliau pateikiamas veikimo mechanizmų "
                "palyginimas; tai nėra teiginys, kad "
                "derinys yra saugus."
            )


            if known1:

                st.write(
                    f"**{first}:** "
                    + " ".join(known1)
                )


            if known2:

                st.write(
                    f"**{second}:** "
                    + " ".join(known2)
                )


            if not known1 or not known2:

                st.warning(
                    "Bent vieno preparato sudėtis šiame "
                    "prototipe dar nėra pakankamai "
                    "suprofiliuota, todėl automatinės "
                    "sąveikos išvados neteikiamos."
                )


        st.markdown(
            "### 💉 Vartojimo būdas"
        )

        st.write(
            f"{first}: "
            f"{route_text(a.get('routes', []))}"
        )

        st.write(
            f"{second}: "
            f"{route_text(b.get('routes', []))}"
        )


# =========================================================
# APAČIA
# =========================================================

st.divider()

st.caption(
    "Bazinis modelis: 24 klasės. "
    "HOG apdorojimas: 128×128, 9 orientacijos, "
    "8×8 pikselių ląstelė, 2×2 blokas."
)
