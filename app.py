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
# GRUPIŲ SĄVEIKOS
# =========================================================

def find_group_interaction(
    ingredient_1,
    ingredient_2
):

    if interaction_rules.empty:
        return None

    info_1 = get_ingredient_knowledge(
        ingredient_1
    )

    info_2 = get_ingredient_knowledge(
        ingredient_2
    )

    if info_1 is None or info_2 is None:
        return None

    group_1 = normalize_text(
        info_1.get("group")
    )

    group_2 = normalize_text(
        info_2.get("group")
    )

    if not group_1 or not group_2:
        return None

    for _, rule in interaction_rules.iterrows():

        rule_a = normalize_text(
            rule.get("group_a")
        )

        rule_b = normalize_text(
            rule.get("group_b")
        )

        direct = (
            group_1 == rule_a
            and group_2 == rule_b
        )

        reverse = (
            group_1 == rule_b
            and group_2 == rule_a
        )

        if direct or reverse:
            return rule

    return None


# =========================================================
# OCR – TELEFONO NUOTRAUKOMS
# =========================================================

def prepare_ocr_images(image):
    """
    Paruošia kelias tos pačios nuotraukos versijas OCR.

    Tikrinama:
    - telefono EXIF orientacija;
    - 0°, 90°, 180° ir 270°;
    - originalus vaizdas;
    - pilkumo vaizdas;
    - automatinis kontrastas;
    - sustiprintas kontrastas;
    - paryškintos raidės.
    """

    # Sutvarkome telefono nuotraukos EXIF orientaciją
    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    prepared_images = []

    # Tikriname visas keturias galimas teksto orientacijas
    for angle in (0, 90, 180, 270):

        rotated = image.rotate(
            angle,
            expand=True
        )

        # ---------------------------------------------
        # TELEFONO NUOTRAUKOS DYDŽIO OPTIMIZAVIMAS
        # ---------------------------------------------

        width, height = rotated.size

        max_side = 1800

        if max(width, height) > max_side:

            scale = (
                max_side
                / max(width, height)
            )

            new_width = max(
                1,
                int(width * scale)
            )

            new_height = max(
                1,
                int(height * scale)
            )

            rotated = rotated.resize(
                (
                    new_width,
                    new_height
                ),
                Image.Resampling.LANCZOS
            )

        # ---------------------------------------------
        # 1. ORIGINALI VERSIJA
        # ---------------------------------------------

        prepared_images.append(
            rotated
        )

        # ---------------------------------------------
        # 2. PILKUMO VERSIJA
        # ---------------------------------------------

        gray = ImageOps.grayscale(
            rotated
        )

        prepared_images.append(
            gray
        )

        # ---------------------------------------------
        # 3. AUTOMATINIS KONTRASTAS
        # ---------------------------------------------

        autocontrast = ImageOps.autocontrast(
            gray
        )

        prepared_images.append(
            autocontrast
        )

        # ---------------------------------------------
        # 4. STIPRESNIS KONTRASTAS
        # ---------------------------------------------

        contrast = ImageEnhance.Contrast(
            autocontrast
        ).enhance(1.8)

        prepared_images.append(
            contrast
        )

        # ---------------------------------------------
        # 5. PARYŠKINTOS RAIDĖS
        # ---------------------------------------------

        sharp = ImageEnhance.Sharpness(
            contrast
        ).enhance(2.0)

        prepared_images.append(
            sharp
        )

    return prepared_images


def run_ocr(image):
    """
    Paleidžia Tesseract OCR kelioms nuotraukos
    orientacijoms ir keliems teksto analizės režimams.
    """

    texts = []

    prepared_images = prepare_ocr_images(
        image
    )

    # Skirtingi Tesseract puslapio analizės režimai:
    #
    # PSM 6  – vientisas teksto blokas
    # PSM 11 – išsklaidytas tekstas
    # PSM 12 – išsklaidytas tekstas su orientacijos analize

    configs = [
        "--psm 6",
        "--psm 11",
        "--psm 12"
    ]

    for prepared in prepared_images:

        for config in configs:

            try:

                text = (
                    pytesseract.image_to_string(
                        prepared,
                        config=config
                    )
                )

                text = text.strip()

                if text:
                    texts.append(text)

            except Exception:
                # Vieno OCR bandymo klaida neturi
                # sustabdyti visos programos
                continue

    # Pašaliname identiškus OCR rezultatus
    unique_texts = list(
        dict.fromkeys(texts)
    )

    return "\n".join(
        unique_texts
    )


# =========================================================
# OCR – STIPRUMO ATPAŽINIMAS
# =========================================================

def extract_strengths(text):

    normalized = normalize_text(
        text
    )

    matches = re.findall(
        r"\b\d+(?:[.,]\d+)?\s*"
        r"(?:mg|mcg|ug|g|ml)\b",
        normalized
    )

    results = []

    for item in matches:

        item = item.replace(
            ",",
            "."
        )

        item = re.sub(
            r"\s+",
            " ",
            item
        )

        if item not in results:
            results.append(item)

    return results


# =========================================================
# OCR – FARMACINĖS FORMOS ATPAŽINIMAS
# =========================================================

def detect_form(text):

    t = normalize_text(
        text
    )

    rules = [

        (
            [
                "plevele",
                "dengtos",
                "tabletes"
            ],
            "plėvele dengtos tabletės"
        ),

        (
            [
                "minkstosios",
                "kapsules"
            ],
            "minkštosios kapsulės"
        ),

        (
            [
                "kietosios",
                "kapsules"
            ],
            "kietosios kapsulės"
        ),

        (
            [
                "injekcinis",
                "tirpalas"
            ],
            "injekcinis tirpalas"
        ),

        (
            [
                "geriamasis",
                "tirpalas"
            ],
            "geriamasis tirpalas"
        ),

        (
            [
                "geriamieji",
                "lasai"
            ],
            "geriamieji lašai"
        ),

        (
            [
                "tabletes"
            ],
            "tabletės"
        ),

        (
            [
                "kapsules"
            ],
            "kapsulės"
        ),

        (
            [
                "sirupas"
            ],
            "sirupas"
        ),

        (
            [
                "gelis"
            ],
            "gelis"
        ),

        (
            [
                "kremas"
            ],
            "kremas"
        ),

        (
            [
                "tepalas"
            ],
            "tepalas"
        ),

        (
            [
                "milteliai"
            ],
            "milteliai"
        ),

        (
            [
                "granules"
            ],
            "granulės"
        )
    ]

    for words, form in rules:

        if all(
            word in t
            for word in words
        ):

            return form

    return None


# =========================================================
# OCR – TEKSTO FRAGMENTAI
# =========================================================

def build_ocr_fragments(text):
    """
    Iš OCR teksto sukuria trumpesnius fragmentus.

    Tai leidžia rasti vaisto pavadinimą net tada,
    kai Tesseract toje pačioje eilutėje perskaito
    ir kitą tekstą.
    """

    fragments = []

    lines = [
        normalize_text(line)
        for line in text.splitlines()
    ]

    lines = [
        line
        for line in lines
        if len(line) >= 3
    ]

    for line in lines:

        fragments.append(
            line
        )

        words = line.split()

        # Tikriname 1–4 žodžių kombinacijas
        for size in range(
            1,
            5
        ):

            if len(words) < size:
                continue

            for i in range(
                len(words) - size + 1
            ):

                fragment = " ".join(
                    words[
                        i:i + size
                    ]
                )

                if len(fragment) >= 3:

                    fragments.append(
                        fragment
                    )

    # Pašaliname pasikartojimus
    return list(
        dict.fromkeys(
            fragments
        )
    )


# =========================================================
# OCR – VEIKLIOSIOS MEDŽIAGOS ATPAŽINIMAS
# =========================================================

def detect_ingredient(text):

    if (
        "veiklioji_medz_lt"
        not in vvkt.columns
    ):
        return None

    ingredients = (
        vvkt["veiklioji_medz_lt"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    ocr_norm = normalize_text(
        text
    )

    # ---------------------------------------------
    # 1. TIKSLUS VEIKLIOSIOS MEDŽIAGOS RADIMAS
    # ---------------------------------------------

    exact = []

    for ingredient in ingredients:

        ing_norm = normalize_text(
            ingredient
        )

        if (
            len(ing_norm) >= 5
            and ing_norm in ocr_norm
        ):

            exact.append(
                ingredient
            )

    if exact:

        # Jei rasta daugiau nei viena,
        # pasirenkame ilgiausią tikslų atitikmenį
        return max(
            exact,
            key=lambda x: len(
                normalize_text(x)
            )
        )

    # ---------------------------------------------
    # 2. FUZZY VEIKLIOSIOS MEDŽIAGOS PAIEŠKA
    # ---------------------------------------------

    normalized_map = {
        normalize_text(x): x
        for x in ingredients
    }

    choices = list(
        normalized_map.keys()
    )

    best_name = None
    best_score = 0.0

    fragments = build_ocr_fragments(
        text
    )

    for fragment in fragments:

        # Labai trumpi fragmentai sukelia
        # per daug klaidingų atitikimų
        if len(fragment) < 5:
            continue

        matches = get_close_matches(
            fragment,
            choices,
            n=1,
            cutoff=0.70
        )

        if not matches:
            continue

        match = matches[0]

        score = similarity(
            fragment,
            match
        )

        if score > best_score:

            best_score = score

            best_name = (
                normalized_map[
                    match
                ]
            )

    return best_name

# =========================================================
# OCR PAVADINIMO KANDIDATAI
# =========================================================

def find_name_candidates(
    ocr_text,
    max_names=50
):

    ocr_norm = normalize_text(
        ocr_text
    )

    normalized_map = {
        normalize_text(name): name
        for name in vvkt_names
    }

    normalized_names = list(
        normalized_map.keys()
    )

    scores = {}

    for normalized_name, original_name in (
        normalized_map.items()
    ):

        if len(normalized_name) < 3:
            continue

        if normalized_name in ocr_norm:

            scores[original_name] = 1.0

    fragments = sorted(
        build_ocr_fragments(
            ocr_text
        ),
        key=len,
        reverse=True
    )[:100]

    for fragment in fragments:

        if len(fragment) < 4:
            continue

        matches = get_close_matches(
            fragment,
            normalized_names,
            n=5,
            cutoff=0.55
        )

        for match in matches:

            original = (
                normalized_map[match]
            )

            score = similarity(
                fragment,
                match
            )

            if score > scores.get(
                original,
                0
            ):

                scores[original] = score

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        name
        for name, _
        in ranked[:max_names]
    ]


# =========================================================
# OCR BALAI
# =========================================================

def name_score(
    name,
    ocr_text
):

    drug = normalize_text(name)
    text = normalize_text(ocr_text)

    if drug in text:
        return 1.0

    best = 0.0

    for fragment in build_ocr_fragments(
        ocr_text
    ):

        score = similarity(
            drug,
            fragment
        )

        best = max(
            best,
            score
        )

    return best


def strength_score(
    strength,
    ocr_text
):

    strength = clean_value(
        strength
    )

    if strength == "—":
        return 0.0

    detected = extract_strengths(
        ocr_text
    )

    if not detected:
        return 0.0

    strength_norm = normalize_text(
        strength
    ).replace(" ", "")

    for item in detected:

        item_norm = normalize_text(
            item
        ).replace(" ", "")

        if item_norm in strength_norm:
            return 1.0

    return 0.0


def ingredient_score(
    ingredient,
    ocr_text
):

    ingredient = clean_value(
        ingredient
    )

    if ingredient == "—":
        return 0.0

    ing = normalize_text(
        ingredient
    )

    text = normalize_text(
        ocr_text
    )

    if ing in text:
        return 1.0

    best = 0.0

    for fragment in build_ocr_fragments(
        ocr_text
    ):

        if len(fragment) < 5:
            continue

        best = max(
            best,
            similarity(
                ing,
                fragment
            )
        )

    if best < 0.58:
        return 0.0

    return best


def form_score(
    form,
    ocr_text
):

    form = clean_value(
        form
    )

    if form == "—":
        return 0.0

    detected = detect_form(
        ocr_text
    )

    if not detected:
        return 0.0

    a = normalize_text(form)
    b = normalize_text(detected)

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.85

    return similarity(a, b)


# =========================================================
# VVKT KANDIDATŲ REITINGAVIMAS
# =========================================================

def rank_vvkt_candidates(
    ocr_text,
    top_n=5
):

    if not ocr_text.strip():
        return []

    candidate_names = find_name_candidates(
        ocr_text,
        max_names=50
    )

    detected_ingredient = detect_ingredient(
        ocr_text
    )

    if detected_ingredient:

        ingredient_rows = vvkt[
            vvkt["veiklioji_medz_lt"]
            .fillna("")
            .astype(str)
            .apply(normalize_text)
            ==
            normalize_text(
                detected_ingredient
            )
        ]

        for name in (
            ingredient_rows[
                "preparato_pav"
            ]
            .dropna()
            .astype(str)
            .tolist()
        ):

            if name not in candidate_names:
                candidate_names.append(name)

    results = []

    for name in candidate_names:

        rows = get_vvkt_rows(name)

        best_result = None

        for _, row in rows.iterrows():

            ns = name_score(
                name,
                ocr_text
            )

            ins = ingredient_score(
                row.get(
                    "veiklioji_medz_lt"
                ),
                ocr_text
            )

            ss = strength_score(
                row.get("stiprumas"),
                ocr_text
            )

            fs = form_score(
                row.get(
                    "farmacine_forma_lt"
                ),
                ocr_text
            )

            total = (
                ns * 0.55
                + ins * 0.20
                + ss * 0.15
                + fs * 0.10
            )

            result = {
                "name": name,
                "ingredient": clean_value(
                    row.get(
                        "veiklioji_medz_lt"
                    )
                ),
                "strength": clean_value(
                    row.get("stiprumas")
                ),
                "form": clean_value(
                    row.get(
                        "farmacine_forma_lt"
                    )
                ),
                "route": clean_value(
                    row.get(
                        "vartojimo_budas"
                    )
                ),
                "score": total,
                "name_score": ns,
                "ingredient_score": ins,
                "strength_score": ss,
                "form_score": fs
            }

            if (
                best_result is None
                or total > best_result["score"]
            ):
                best_result = result

        if best_result is not None:
            results.append(best_result)

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        x
        for x in results
        if x["score"] >= 0.28
    ][:top_n]


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
    "OCR + VVKT vaistų duomenys + "
    "farmakologinių grupių analizė"
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
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# 2. ATPAŽINIMAS
# =========================================================

if uploaded:

   image = ImageOps.exif_transpose(
    Image.open(uploaded)
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

    with st.spinner(
        "Analizuojama pakuotė..."
    ):

        ocr_text = run_ocr(image)

        ranked_candidates = (
            rank_vvkt_candidates(
                ocr_text,
                top_n=5
            )
        )

    strengths = extract_strengths(
        ocr_text
    )

    detected_form = detect_form(
        ocr_text
    )

    detected_ingredient = detect_ingredient(
        ocr_text
    )

    st.markdown(
        "#### 🧾 Iš pakuotės aptikta informacija"
    )

    st.write(
        "**Veiklioji medžiaga:**",
        detected_ingredient
        or "automatiškai nenustatyta"
    )

    st.write(
        "**Stiprumas:**",
        ", ".join(strengths[:3])
        if strengths
        else "automatiškai nenustatytas"
    )

    st.write(
        "**Farmacinė forma:**",
        detected_form
        or "automatiškai nenustatyta"
    )

    st.markdown(
        "#### 🔎 Galimi VVKT preparatai"
    )

    if ranked_candidates:

        labels = {}

        for index, candidate in enumerate(
            ranked_candidates,
            start=1
        ):

            label = (
                f"{index}. "
                f"{candidate['name']} | "
                f"{candidate['strength']} | "
                f"{candidate['score'] * 100:.0f}% atitikimas"
            )

            labels[label] = candidate

        selected_label = st.selectbox(
            "Pasirinkite preparatą",
            list(labels.keys()),
            key="ocr_candidate"
        )

        candidate = labels[
            selected_label
        ]

        st.success(
            f"💊 Galimas preparatas: "
            f"**{candidate['name']}**"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Bendras atitikimas",
            f"{candidate['score'] * 100:.0f}%"
        )

        c2.metric(
            "Pavadinimas",
            f"{candidate['name_score'] * 100:.0f}%"
        )

        c3.metric(
            "Veiklioji medžiaga",
            f"{candidate['ingredient_score'] * 100:.0f}%"
        )

        st.write(
            "**Veiklioji medžiaga:**",
            candidate["ingredient"]
        )

        st.write(
            "**Stiprumas:**",
            candidate["strength"]
        )

        st.write(
            "**Farmacinė forma:**",
            candidate["form"]
        )

        st.caption(
            "Atitikimo procentas yra "
            "paieškos algoritmo balas, "
            "o ne statistinė AI tikimybė."
        )

        if st.button(
            "✅ Patvirtinti preparatą",
            type="primary"
        ):

            st.session_state[
                "first_vvkt_drug"
            ] = candidate["name"]

            st.rerun()

    else:

        st.warning(
            "Patikimo VVKT kandidato "
            "automatiškai parinkti nepavyko."
        )

    # =====================================================
    # BASELINE
    # =====================================================

    with st.expander(
        "📊 Bazinio AI modelio rezultatas"
    ):

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
            "Baseline modelis mokytas tik "
            "su 24 preparatų klasėmis."
        )


# =========================================================
# RANKINĖ PAIEŠKA
# =========================================================

st.divider()

st.markdown(
    "### 🔎 Rankinė pirmo preparato paieška"
)

first_query = st.text_input(
    "Ieškoti pirmo preparato",
    placeholder="Pvz. Nalgesin, IBUPROM..."
)

if first_query:

    matches = find_names(
        first_query
    )

    if matches:

        manual_first = st.selectbox(
            "Rasti preparatai",
            matches,
            key="manual_first"
        )

        show_vvkt_info(
            manual_first
        )

        if st.button(
            "✅ Naudoti šį preparatą"
        ):

            st.session_state[
                "first_vvkt_drug"
            ] = manual_first

            st.rerun()


# =========================================================
# 3. PATVIRTINTAS PREPARATAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '✅ 3. Patvirtinkite rezultatą'
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

    show_vvkt_info(first)

else:

    st.info(
        "Dar nepatvirtintas "
        "pirmasis preparatas."
    )


# =========================================================
# 4. ANTRAS VAISTAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '💊 4. Pasirinkite antrą vaistą'
    '</div>',
    unsafe_allow_html=True
)

second_query = st.text_input(
    "Ieškoti antro preparato",
    placeholder="Pvz. Dolmen, NO-SPA..."
)

second = None

if second_query:

    matches = find_names(
        second_query
    )

    if matches:

        second = st.selectbox(
            "Pasirinkite antrą preparatą",
            matches,
            key="second_drug"
        )

        show_vvkt_info(second)


# =========================================================
# 5. FARMAKOLOGINĖ ANALIZĖ
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '🔬 5. Kaip šie vaistai veikia kartu'
    '</div>',
    unsafe_allow_html=True
)

if st.button(
    "🔬 Analizuoti vaistų derinį",
    type="primary"
):

    if not first:

        st.warning(
            "Pirmiausia patvirtinkite "
            "pirmą preparatą."
        )

    elif not second:

        st.warning(
            "Pasirinkite antrą preparatą."
        )

    elif first == second:

        st.warning(
            "Pasirinkite du skirtingus preparatus."
        )

    else:

        first_row = get_vvkt_row(first)
        second_row = get_vvkt_row(second)

        ingredient_1 = get_vvkt_ingredient(
            first
        )

        ingredient_2 = get_vvkt_ingredient(
            second
        )

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
                ingredient_1 or "—"
            )

            if first_row is not None:

                st.write(
                    "**Stiprumas:**",
                    clean_value(
                        first_row.get("stiprumas")
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
                ingredient_2 or "—"
            )

            if second_row is not None:

                st.write(
                    "**Stiprumas:**",
                    clean_value(
                        second_row.get("stiprumas")
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
            "## 🔬 KAIP ŠIE VAISTAI VEIKIA KARTU"
        )

        known_1 = show_ingredient_explanation(
            first,
            ingredient_1
        )

        known_2 = show_ingredient_explanation(
            second,
            ingredient_2
        )

        info_1 = get_ingredient_knowledge(
            ingredient_1
        )

        info_2 = get_ingredient_knowledge(
            ingredient_2
        )

        # =================================================
        # GRUPIŲ SĄVEIKOS TAISYKLĖ
        # =================================================

        group_rule = find_group_interaction(
            ingredient_1,
            ingredient_2
        )

        if group_rule is not None:

            st.markdown(
                "### ⚠️ Galima farmakologinė sąveika"
            )

            st.warning(
                clean_value(
                    group_rule.get(
                        "interaction"
                    )
                )
            )

            st.markdown(
                "#### Galimas poveikis / rizika"
            )

            st.write(
                clean_value(
                    group_rule.get(
                        "effect"
                    )
                )
            )

            st.markdown(
                "#### 📋 Išvada"
            )

            st.error(
                clean_value(
                    group_rule.get(
                        "conclusion"
                    )
                )
            )

            source = clean_value(
                group_rule.get(
                    "source"
                )
            )

            if source != "—":

                st.markdown(
                    "#### 📚 Sąveikos šaltinis"
                )

                st.write(
                    source
                )

        # =================================================
        # JEI GRUPINĖS TAISYKLĖS NĖRA
        # =================================================

        else:

            st.markdown(
                "### 📋 Išvada"
            )

            if known_1 and known_2:

                st.info(
                    "Abiejų veikliųjų medžiagų "
                    "farmakologiniai profiliai "
                    "prototipo bazėje yra aprašyti, "
                    "tačiau šiai konkrečiai "
                    "farmakologinių grupių porai "
                    "patikrinta sąveikos taisyklė "
                    "dar neįtraukta. "
                    "Tai nėra teiginys, kad "
                    "vaistus saugu vartoti kartu."
                )

            else:

                st.warning(
                    "Bent vienos veikliosios "
                    "medžiagos farmakologinis "
                    "profilis dar neįtrauktas "
                    "į patikrintą prototipo bazę. "
                    "Todėl automatinė porinė "
                    "farmakologinė išvada "
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
                info.get("source")
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
