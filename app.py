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
    # OCR TEKSTO PALYGINIMAS SU VVKT PREPARATŲ PAVADINIMAIS
    # =====================================================

    ocr_text_lower = ocr_text.lower()

    # 1. Pirmiausia ieškome tikslaus VVKT preparato
    # pavadinimo visame OCR tekste.
    exact_matches = []

    for name in vvkt_names:
        if name.lower() in ocr_text_lower:
            exact_matches.append(name)

    # Ilgesni pavadinimai laikomi specifiškesniais.
    exact_matches = sorted(
        exact_matches,
        key=len,
        reverse=True
    )

    ocr_suggestions = exact_matches.copy()

    # 2. Jei tikslaus pavadinimo OCR tekste neradome,
    # naudojame fuzzy paiešką.
    if not ocr_suggestions:

        ocr_lines = [
            line.strip()
            for line in ocr_text.splitlines()
            if len(line.strip()) >= 4
        ]

        fuzzy_matches = []

        for line in ocr_lines:

            # Tikriname ne tik visą OCR eilutę,
            # bet ir atskirus jos žodžius.
            search_parts = [line]

            search_parts.extend(
                word
                for word in line.split()
                if len(word) >= 4
            )

            for part in search_parts:

                matches = get_close_matches(
                    part,
                    vvkt_names,
                    n=5,
                    cutoff=0.70
                )

                fuzzy_matches.extend(matches)

        ocr_suggestions = list(
            dict.fromkeys(fuzzy_matches)
        )

    # Rodome ne daugiau kaip 10 pasiūlymų.
    ocr_suggestions = ocr_suggestions[:10]


    if ocr_suggestions:

        st.markdown(
            "### 🔎 OCR pasiūlyti VVKT preparatai"
        )

        ocr_selected = st.selectbox(
            "Pasirinkite labiausiai atitinkantį preparatą",
            ocr_suggestions,
            key="ocr_vvkt_match"
        )

        selected_rows = vvkt[
            vvkt["preparato_pav"] == ocr_selected
        ]

        if not selected_rows.empty:

            row = selected_rows.iloc[0]

            st.markdown(
                "#### 💊 OCR rasto preparato informacija"
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
            "OCR tekste nepavyko rasti pakankamai "
            "panašaus VVKT preparato pavadinimo."
        )


    # -------------------------
    # HOG + LOGISTIC REGRESSION
    # -------------------------

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


default_index = (
    classes.index(st.session_state.first_drug)
    if st.session_state.first_drug in classes
    else 0
)

first = st.selectbox(
    "Pirmasis vaistas",
    classes,
    index=default_index
)

st.session_state.first_drug = first


# =========================================================
# 4. ANTRAS VAISTAS
# =========================================================

st.markdown(
    '<div class="step">'
    '💊 4. Pasirinkite antrą vaistą'
    '</div>',
    unsafe_allow_html=True
)

second = st.selectbox(
    "Antrasis vaistas",
    classes,
    index=1 if len(classes) > 1 else 0
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

    if first == second:

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
