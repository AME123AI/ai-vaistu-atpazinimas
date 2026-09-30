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
    return joblib.load(
        ROOT / "hog_logreg.joblib"
    )


@st.cache_data
def load_json(name):
    path = ROOT / name

    if not path.exists():
        return {}

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


@st.cache_data
def load_interactions():
    path = ROOT / "interactions.csv"

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

    return pd.read_csv(
        path,
        encoding="utf-8-sig"
    )


model = load_model()

drug_data = load_json(
    "drug_profiles.json"
)

ingredient_profiles = load_json(
    "ingredient_profiles.json"
)

interactions = load_interactions()

ingredient_knowledge = (
    load_ingredient_knowledge()
)

vvkt = load_vvkt()


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

    # OCR dažnai skirtingai perskaito brūkšnelius
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Paliekame raides, skaičius, +, /, -
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

    if not value:
        return "—"

    return value


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

    if not value:
        return None

    return value


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
# FARMAKOLOGINĖ INFORMACIJA
# =========================================================

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
            f"{ingredient or 'veiklioji medžiaga nenustatyta'}**"
        )

        st.caption(
            "Šios veikliosios medžiagos "
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
        f"**{drug_name} – {ingredient}**"
    )

    st.write(
        f"**Grupė:** {group}"
    )

    st.write(
        f"**Veikimo mechanizmas:** "
        f"{mechanism}"
    )

    st.write(
        f"**Poveikis:** {effect}"
    )

    return True


# =========================================================
# OCR – VAIZDO PARUOŠIMAS
# =========================================================

def prepare_ocr_images(image):

    images = []

    # Originalus vaizdas
    images.append(image)

    # Padidintas vaizdas
    width, height = image.size

    scale = 2

    enlarged = image.resize(
        (
            width * scale,
            height * scale
        )
    )

    images.append(enlarged)

    # Pilkumo skalė
    gray = ImageOps.grayscale(
        enlarged
    )

    gray = ImageOps.autocontrast(
        gray
    )

    images.append(gray)

    # Didesnis kontrastas
    contrast = ImageEnhance.Contrast(
        gray
    ).enhance(2.0)

    images.append(contrast)

    return images


def run_ocr(image):

    images = prepare_ocr_images(
        image
    )

    texts = []

    configs = [
        "--psm 6",
        "--psm 11"
    ]

    for prepared in images:

        for config in configs:

            try:

                text = (
                    pytesseract.image_to_string(
                        prepared,
                        config=config
                    )
                )

                if text.strip():
                    texts.append(text)

            except Exception:
                pass

    # Pasikartojančių OCR rezultatų nereikia
    unique = []

    for text in texts:

        if text not in unique:
            unique.append(text)

    return "\n".join(unique)


# =========================================================
# OCR – STIPRUMAS
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

        item = item.replace(",", ".")

        item = re.sub(
            r"\s+",
            " ",
            item
        )

        if item not in results:
            results.append(item)

    return results


# =========================================================
# OCR – FARMACINĖ FORMA
# =========================================================

def detect_form(text):

    t = normalize_text(text)

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
            ["tabletes"],
            "tabletės"
        ),

        (
            ["kapsules"],
            "kapsulės"
        ),

        (
            ["sirupas"],
            "sirupas"
        ),

        (
            ["gelis"],
            "gelis"
        ),

        (
            ["kremas"],
            "kremas"
        )
    ]

    for words, form in rules:

        found = sum(
            1
            for word in words
            if word in t
        )

        if found == len(words):
            return form

    return None


# =========================================================
# OCR – VEIKLIOJI MEDŽIAGA
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

    # 1. Tikslus radimas
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

        return max(
            exact,
            key=lambda x: len(
                normalize_text(x)
            )
        )

    # 2. Fuzzy paieška
    normalized_map = {
        normalize_text(x): x
        for x in ingredients
    }

    choices = list(
        normalized_map.keys()
    )

    fragments = build_ocr_fragments(
        text
    )

    best_name = None
    best_score = 0.0

    for fragment in fragments:

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
# OCR – TEKSTO FRAGMENTAI
# =========================================================

def build_ocr_fragments(text):

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

        fragments.append(line)

        words = line.split()

        # 1–4 žodžių kombinacijos
        for size in range(1, 5):

            if len(words) < size:
                continue

            for i in range(
                len(words) - size + 1
            ):

                fragment = " ".join(
                    words[i:i + size]
                )

                if len(fragment) >= 3:
                    fragments.append(
                        fragment
                    )

    # Taip pat viso OCR teksto žodžių kombinacijos
    all_words = normalize_text(
        text
    ).split()

    for size in range(1, 4):

        for i in range(
            max(
                0,
                len(all_words) - size + 1
            )
        ):

            fragment = " ".join(
                all_words[i:i + size]
            )

            if len(fragment) >= 3:
                fragments.append(
                    fragment
                )

    return list(
        dict.fromkeys(
            fragments
        )
    )


# =========================================================
# OCR – PAVADINIMO KANDIDATAI
# =========================================================

def find_name_candidates(
    ocr_text,
    max_names=40
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

    # -----------------------------------------------------
    # 1. Tikslus / dalinis pavadinimas
    # -----------------------------------------------------

    for normalized_name, original_name in (
        normalized_map.items()
    ):

        if len(normalized_name) < 3:
            continue

        if normalized_name in ocr_norm:

            scores[original_name] = max(
                scores.get(
                    original_name,
                    0
                ),
                1.0
            )

    # -----------------------------------------------------
    # 2. Fuzzy paieška pagal OCR fragmentus
    # -----------------------------------------------------

    fragments = build_ocr_fragments(
        ocr_text
    )

    # Ilgesni fragmentai dažniausiai informatyvesni
    fragments = sorted(
        fragments,
        key=len,
        reverse=True
    )

    # Apsauga nuo pernelyg didelio skaičiavimo
    fragments = fragments[:100]

    for fragment in fragments:

        # Labai bendri fragmentai ignoruojami
        if len(fragment) < 4:
            continue

        matches = get_close_matches(
            fragment,
            normalized_names,
            n=5,
            cutoff=0.55
        )

        for match in matches:

            original_name = (
                normalized_map[
                    match
                ]
            )

            score = similarity(
                fragment,
                match
            )

            if score > scores.get(
                original_name,
                0
            ):

                scores[
                    original_name
                ] = score

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        name
        for name, score in ranked[
            :max_names
        ]
    ]


# =========================================================
# ATITIKIMO BALAI
# =========================================================

def name_score(
    name,
    ocr_text
):

    drug = normalize_text(
        name
    )

    text = normalize_text(
        ocr_text
    )

    if not drug:
        return 0.0

    if drug in text:
        return 1.0

    fragments = build_ocr_fragments(
        ocr_text
    )

    best = 0.0

    for fragment in fragments:

        score = similarity(
            drug,
            fragment
        )

        if score > best:
            best = score

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

    # Vien tik skaičiaus palyginimas
    row_numbers = re.findall(
        r"\d+(?:[.,]\d+)?",
        strength_norm
    )

    ocr_numbers = []

    for item in detected:

        ocr_numbers.extend(
            re.findall(
                r"\d+(?:[.,]\d+)?",
                item
            )
        )

    for number in row_numbers:

        if number in ocr_numbers:
            return 0.75

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

    fragments = build_ocr_fragments(
        ocr_text
    )

    best = 0.0

    for fragment in fragments:

        if len(fragment) < 5:
            continue

        score = similarity(
            ing,
            fragment
        )

        if score > best:
            best = score

    # Silpnas panašumas nelaikomas įrodymu
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

    a = normalize_text(
        form
    )

    b = normalize_text(
        detected
    )

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.85

    return similarity(
        a,
        b
    )


# =========================================================
# TOP VVKT KANDIDATŲ REITINGAVIMAS
# =========================================================

def rank_vvkt_candidates(
    ocr_text,
    top_n=5
):

    if not ocr_text.strip():
        return []

    candidate_names = (
        find_name_candidates(
            ocr_text,
            max_names=50
        )
    )

    detected_ingredient = (
        detect_ingredient(
            ocr_text
        )
    )

    detected_strengths = (
        extract_strengths(
            ocr_text
        )
    )

    detected_form = (
        detect_form(
            ocr_text
        )
    )

    # -----------------------------------------------------
    # Jeigu OCR rado veikliąją medžiagą,
    # pridedame visus jos VVKT preparatus
    # -----------------------------------------------------

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
                candidate_names.append(
                    name
                )

    # -----------------------------------------------------
    # Jeigu pavadinimų visai nepavyko rasti,
    # panaudojame stiprumą + formą
    # -----------------------------------------------------

    if not candidate_names:

        fallback = vvkt.copy()

        if detected_strengths:

            number = re.findall(
                r"\d+(?:[.,]\d+)?",
                detected_strengths[0]
            )

            if number:

                n = number[0]

                temp = fallback[
                    fallback["stiprumas"]
                    .fillna("")
                    .astype(str)
                    .str.contains(
                        n,
                        regex=False
                    )
                ]

                if not temp.empty:
                    fallback = temp

        if detected_form:

            form_norm = normalize_text(
                detected_form
            )

            temp = fallback[
                fallback[
                    "farmacine_forma_lt"
                ]
                .fillna("")
                .astype(str)
                .apply(normalize_text)
                .apply(
                    lambda x:
                    similarity(
                        x,
                        form_norm
                    ) > 0.55
                )
            ]

            if not temp.empty:
                fallback = temp

        candidate_names = (
            fallback[
                "preparato_pav"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()[:100]
        )

    # -----------------------------------------------------
    # Kiekvienam preparatui parenkame geriausiai
    # atitinkančią VVKT pakuotę
    # -----------------------------------------------------

    results = []

    for name in candidate_names:

        rows = get_vvkt_rows(
            name
        )

        if rows.empty:
            continue

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
                row.get(
                    "stiprumas"
                ),
                ocr_text
            )

            fs = form_score(
                row.get(
                    "farmacine_forma_lt"
                ),
                ocr_text
            )

            # ---------------------------------------------
            # SVORIAI
            # ---------------------------------------------
            #
            # Pavadinimas svarbiausias.
            # Veiklioji medžiaga – antras signalas.
            # Stiprumas ir forma padeda atskirti pakuotes.
            # ---------------------------------------------

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
                    row.get(
                        "stiprumas"
                    )
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
                "prescription": clean_value(
                    row.get(
                        "recepto_poreikis"
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
                or total
                > best_result["score"]
            ):
                best_result = result

        if best_result is not None:
            results.append(
                best_result
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Pašaliname silpnus atsitiktinius rezultatus
    filtered = [
        result
        for result in results
        if result["score"] >= 0.28
    ]

    return filtered[:top_n]


# =========================================================
# PORINĖS SĄVEIKOS
# =========================================================

def find_trade_interactions(
    first_name,
    second_name
):

    found = []

    if interactions.empty:
        return found

    required = {
        "drug_a",
        "drug_b"
    }

    if not required.issubset(
        interactions.columns
    ):
        return found

    first_norm = normalize_text(
        first_name
    )

    second_norm = normalize_text(
        second_name
    )

    for _, rule in interactions.iterrows():

        a = normalize_text(
            rule.get(
                "drug_a",
                ""
            )
        )

        b = normalize_text(
            rule.get(
                "drug_b",
                ""
            )
        )

        direct = (
            first_norm == a
            and second_norm == b
        )

        reverse = (
            first_norm == b
            and second_norm == a
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

    .candidate {
        padding: 12px;
        border: 1px solid rgba(150,150,150,.25);
        border-radius: 10px;
        margin-bottom: 8px;
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
    "Patobulintas prototipas: "
    "OCR + VVKT duomenys + "
    "HOG / Logistic Regression baseline"
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

    with st.spinner(
        "Analizuojama pakuotė..."
    ):

        ocr_text = run_ocr(
            image
        )

        ranked_candidates = (
            rank_vvkt_candidates(
                ocr_text,
                top_n=5
            )
        )

    # -----------------------------------------------------
    # APTIKTA INFORMACIJA
    # -----------------------------------------------------

    strengths = extract_strengths(
        ocr_text
    )

    detected_form = detect_form(
        ocr_text
    )

    detected_ingredient = (
        detect_ingredient(
            ocr_text
        )
    )

    if (
        detected_ingredient
        or strengths
        or detected_form
    ):

        st.markdown(
            "#### 🧾 Iš pakuotės aptikta informacija"
        )

        if detected_ingredient:

            st.write(
                "**Veiklioji medžiaga:**",
                detected_ingredient
            )

        if strengths:

            st.write(
                "**Stiprumas:**",
                ", ".join(
                    strengths[:3]
                )
            )

        if detected_form:

            st.write(
                "**Farmacinė forma:**",
                detected_form
            )

    # -----------------------------------------------------
    # TOP-5
    # -----------------------------------------------------

    st.markdown(
        "#### 🔎 Galimi VVKT preparatai"
    )

    if ranked_candidates:

        candidate_labels = {}

        for index, candidate in enumerate(
            ranked_candidates,
            start=1
        ):

            percent = (
                candidate["score"]
                * 100
            )

            label = (
                f"{index}. "
                f"{candidate['name']} | "
                f"{candidate['strength']} | "
                f"{percent:.0f}% atitikimas"
            )

            candidate_labels[
                label
            ] = candidate

        selected_label = st.selectbox(
            "Pasirinkite preparatą",
            list(
                candidate_labels.keys()
            ),
            key="ocr_ranked_candidate"
        )

        selected_candidate = (
            candidate_labels[
                selected_label
            ]
        )

        st.success(
            f"💊 Galimas preparatas: "
            f"**{selected_candidate['name']}**"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Bendras atitikimas",
                f"{selected_candidate['score'] * 100:.0f}%"
            )

        with c2:

            st.metric(
                "Pavadinimas",
                f"{selected_candidate['name_score'] * 100:.0f}%"
            )

        with c3:

            st.metric(
                "Veiklioji medžiaga",
                f"{selected_candidate['ingredient_score'] * 100:.0f}%"
            )

        st.write(
            "**Veiklioji medžiaga:**",
            selected_candidate[
                "ingredient"
            ]
        )

        st.write(
            "**Stiprumas:**",
            selected_candidate[
                "strength"
            ]
        )

        st.write(
            "**Farmacinė forma:**",
            selected_candidate[
                "form"
            ]
        )

        st.write(
            "**Vartojimo būdas:**",
            selected_candidate[
                "route"
            ]
        )

        st.caption(
            "Atitikimo procentas yra "
            "prototipo paieškos balas, "
            "o ne statistinė AI tikimybė."
        )

        if st.button(
            "✅ Patvirtinti preparatą",
            type="primary",
            key="confirm_first"
        ):

            st.session_state[
                "first_vvkt_drug"
            ] = selected_candidate[
                "name"
            ]

            st.success(
                "Preparatas patvirtintas."
            )

    else:

        st.warning(
            "Automatiškai patikimo VVKT "
            "preparato parinkti nepavyko. "
            "Naudokite rankinę paiešką žemiau."
        )

    # -----------------------------------------------------
    # BASELINE MODELIS
    # -----------------------------------------------------

    with st.expander(
        "📊 Bazinio modelio rezultatas (tyrimui)"
    ):

        st.caption(
            "Šis HOG + Logistic Regression "
            "modelis mokytas tik su 24 "
            "preparatų klasėmis. "
            "Jis nėra naudojamas galutiniam "
            "VVKT preparato identifikavimui."
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
            "Šie įverčiai nėra "
            "kalibruotas pasitikėjimo matas."
        )


# =========================================================
# RANKINĖ PIRMO PREPARATO PAIEŠKA
# =========================================================

st.divider()

st.markdown(
    "### 🔎 Jei reikia – raskite pirmą "
    "preparatą rankiniu būdu"
)

first_query = st.text_input(
    "Ieškoti pirmo preparato VVKT kataloge",
    placeholder=(
        "Pvz. Nalgesin, IBUPROM, Atacand..."
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
# 5. SĄVEIKOS / VEIKIMO PALYGINIMAS
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

        # -------------------------------------------------
        # KAIP VEIKIA KARTU
        # -------------------------------------------------

        st.markdown(
            "### 🔬 KAIP ŠIE VAISTAI VEIKIA KARTU"
        )

        known_first = (
            show_ingredient_explanation(
                first,
                first_ingredient
            )
        )

        st.write("")

        known_second = (
            show_ingredient_explanation(
                second,
                second_ingredient
            )
        )

        # -------------------------------------------------
        # BENDRAS POVEIKIS
        # -------------------------------------------------

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

        if (
            first_info is not None
            and second_info is not None
        ):

            st.markdown(
                "#### Bendras poveikis"
            )

            st.write(
                f"**{first_ingredient}:** "
                f"{clean_value(first_info.get('effect'))}"
            )

            st.write(
                f"**{second_ingredient}:** "
                f"{clean_value(second_info.get('effect'))}"
            )

            st.caption(
                "Šis palyginimas aprašo "
                "veikimo mechanizmus. "
                "Jis pats savaime neparodo, "
                "ar konkretų derinį saugu "
                "vartoti kartu."
            )

        # -------------------------------------------------
        # PATIKRINTA PORINĖ TAISYKLĖ
        # -------------------------------------------------

        rules = find_trade_interactions(
            first,
            second
        )

        if rules:

            st.markdown(
                "#### ⚠️ Patikrinta porinės "
                "sąveikos informacija"
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
                "Šiai konkrečiai preparatų "
                "porai prototipo patikrintų "
                "porinių sąveikų faile atskira "
                "taisyklė neįrašyta. "
                "Tai nėra teiginys, kad "
                "derinį saugu vartoti kartu."
            )

        # -------------------------------------------------
        # IŠVADA
        # -------------------------------------------------

        st.markdown(
            "#### 📋 Išvada"
        )

        if known_first and known_second:

            st.info(
                "Prototipas gali palyginti "
                "šių veikliųjų medžiagų "
                "farmakologinius profilius. "
                "Tačiau vien veikimo mechanizmų "
                "palyginimas nepatvirtina "
                "konkretaus vaistų derinio "
                "saugumo. Reikia vertinti "
                "konkrečias dozes, vartojimo "
                "būdą, kontraindikacijas, "
                "kitus vartojamus vaistus ir "
                "oficialią preparatų informaciją."
            )

        else:

            st.warning(
                "Bent vienos veikliosios "
                "medžiagos patikrinto "
                "farmakologinio profilio "
                "prototipo bazėje dar nėra, "
                "todėl išsamesnė farmakologinė "
                "išvada nepateikiama."
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
        # ŠALTINIAI
        # -------------------------------------------------

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

        st.write(
            "• Preparato duomenys: "
            "projekte esantis VVKT "
            "registruotų vaistinių preparatų "
            "duomenų rinkinys."
        )


# =========================================================
# INFORMACIJA APIE SISTEMĄ
# =========================================================

st.divider()

with st.expander(
    "ℹ️ Apie atpažinimo sistemą"
):

    st.markdown(
        "**Patobulintas atpažinimas**"
    )

    st.write(
        "Pakuotės tekstas keliais būdais "
        "nuskaitomas OCR. Tada rezultatas "
        "lyginamas su VVKT katalogo "
        "preparatų pavadinimais, "
        "veikliosiomis medžiagomis, "
        "stiprumu ir farmacine forma."
    )

    st.write(
        "Sistema pateikia iki 5 geriausiai "
        "atitinkančių VVKT kandidatų. "
        "Galutinį preparatą patvirtina "
        "vartotojas."
    )

    st.markdown(
        "**Bazinis modelis**"
    )

    st.write(
        "HOG + Logistic Regression "
        "modelis mokytas atpažinti "
        "24 vaistų pakuočių klases."
    )

    st.write(
        "Vaizdas konvertuojamas į "
        "pilkumo skalę ir pakeičiamas "
        "į 128×128 pikselių dydį."
    )

    st.write(
        "HOG parametrai: "
        "9 orientacijos, "
        "8×8 pikselių ląstelės ir "
        "2×2 blokai."
    )

    st.write(
        "Bazinis modelis rodomas tik "
        "tyrimui ir palyginimui, nes "
        "jis negali atpažinti preparatų "
        "už savo 24 mokymo klasių ribų."
    )


st.caption(
    "Edukacinis AI prototipas – "
    "vaistų pakuočių atpažinimas "
    "ir sąveikų informacijos demonstravimas."
)
