import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from paddleocr import PaddleOCR


st.set_page_config(
    page_title="PaddleOCR + VVKT testas",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 PaddleOCR + VVKT vaistų pakuotės testas")

st.caption(
    "Atskiras eksperimentas. "
    "Pagrindinis app.py nekeičiamas."
)


# ---------------------------------------------------------
# Teksto normalizavimas
# ---------------------------------------------------------

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).strip().lower()

    # Lietuviškas raides paverčiame į paprastesnę formą,
    # kad OCR klaidos mažiau trukdytų palyginimui.
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ---------------------------------------------------------
# VVKT duomenys
# ---------------------------------------------------------

@st.cache_data
def load_vvkt_data():
    possible_paths = [
        Path("PreparatasPakuote.csv"),
        Path("../PreparatasPakuote.csv"),
        Path(__file__).resolve().parent.parent / "PreparatasPakuote.csv",
    ]

    csv_path = None

    for path in possible_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        raise FileNotFoundError(
            "Nepavyko rasti PreparatasPakuote.csv."
        )

    df = pd.read_csv(
        csv_path,
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = [
        "preparato_pav",
        "veiklioji_medz_lt",
        "stiprumas",
        "farmacine_forma_lt",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "VVKT CSV trūksta stulpelių: "
            + ", ".join(missing)
        )

    df["_name_norm"] = (
        df["preparato_pav"]
        .astype(str)
        .map(normalize_text)
    )

    df["_ingredient_norm"] = (
        df["veiklioji_medz_lt"]
        .astype(str)
        .map(normalize_text)
    )

    df["_strength_norm"] = (
        df["stiprumas"]
        .astype(str)
        .map(normalize_text)
    )

    df["_form_norm"] = (
        df["farmacine_forma_lt"]
        .astype(str)
        .map(normalize_text)
    )

    return df


# ---------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------

@st.cache_resource
def load_paddle_ocr():
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def extract_paddle_text(result):
    found_text = []

    for item in result:
        data = getattr(
            item,
            "json",
            None,
        )

        if callable(data):
            data = data()

        if not isinstance(data, dict):
            continue

        res = data.get(
            "res",
            data,
        )

        texts = res.get(
            "rec_texts",
            [],
        )

        for text in texts:
            text = str(text).strip()

            if text:
                found_text.append(text)

    return found_text


# ---------------------------------------------------------
# Stiprumo aptikimas
# ---------------------------------------------------------

def detect_strengths(text):
    text = normalize_text(text)

    patterns = [
        r"\b\d+(?:[.,]\d+)?\s*mg\b",
        r"\b\d+(?:[.,]\d+)?\s*g\b",
        r"\b\d+(?:[.,]\d+)?\s*mcg\b",
        r"\b\d+(?:[.,]\d+)?\s*µg\b",
    ]

    results = []

    for pattern in patterns:
        for match in re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            match = normalize_text(match)

            if match not in results:
                results.append(match)

    return results


# ---------------------------------------------------------
# Farmacinės formos signalai
# ---------------------------------------------------------

def detect_form(text):
    text = normalize_text(text)

    form_keywords = {
        "tablet": "tabletės",
        "tabletes": "tabletės",
        "tableciu": "tabletės",
        "kapsul": "kapsulės",
        "kapsules": "kapsulės",
        "kapsuliu": "kapsulės",
        "sirup": "sirupas",
        "suspens": "suspensija",
        "tirpal": "tirpalas",
        "krem": "kremas",
        "tepal": "tepalas",
        "gel": "gelis",
        "purskal": "purškalas",
        "miltel": "milteliai",
        "zvakut": "žvakutės",
    }

    for keyword, form in form_keywords.items():
        if keyword in text:
            return form

    return ""


# ---------------------------------------------------------
# OCR fragmentai
# ---------------------------------------------------------

# ---------------------------------------------------------
# OCR fragmentai
# ---------------------------------------------------------

GENERIC_OCR_WORDS = {
    # Farmacinės formos / pakuotės
    "tablete",
    "tabletes",
    "tableciu",
    "tablet",
    "kapsule",
    "kapsules",
    "kapsuliu",
    "kapsul",
    "sirupas",
    "sirup",
    "tirpalas",
    "tirpal",
    "kremas",
    "krem",
    "tepalas",
    "tepal",
    "gelis",
    "gel",
    "purskalas",
    "purskal",
    "milteliai",
    "miltel",
    "zvakutes",
    "zvakut",

    # Dažni farmaciniai / cheminiai žodžiai
    "hydrochloridum",
    "hydrochloride",
    "hidrochloridas",
    "hidrochlorido",
    "chloridum",
    "chloride",
    "chloridas",
    "chlorido",
    "natrii",
    "sodium",
    "kalio",
    "potassium",

    # Bendriniai užrašai
    "mg",
    "ml",
    "geriamasis",
    "geriamoji",
    "dengtos",
    "dengta",
    "plėvele",
    "plevele",
}


def is_useful_name_fragment(fragment):
    fragment = normalize_text(fragment)

    if not fragment:
        return False

    # Vien skaičiai nėra prekės ženklo signalas.
    if fragment.replace(" ", "").isdigit():
        return False

    words = fragment.split()

    useful_words = []

    for word in words:
        if len(word) < 4:
            continue

        if word in GENERIC_OCR_WORDS:
            continue

        useful_words.append(word)

    return len(useful_words) > 0


def build_fragments(lines):
    fragments = []

    for line in lines:
        line_norm = normalize_text(line)

        if not line_norm:
            continue

        # Visa OCR eilutė gali būti naudinga,
        # tačiau tik jei joje yra bent vienas
        # informatyvus žodis.
        if is_useful_name_fragment(line_norm):
            fragments.append(line_norm)

        for word in line_norm.split():
            if (
                len(word) >= 4
                and word not in GENERIC_OCR_WORDS
            ):
                fragments.append(word)

    return list(dict.fromkeys(fragments))


# ---------------------------------------------------------
# VVKT kandidatų paieška
# ---------------------------------------------------------

def find_vvkt_candidates(
    df,
    ocr_lines,
    max_results=10,
):
    full_text = normalize_text(
        " ".join(ocr_lines)
    )

    fragments = build_fragments(
        ocr_lines
    )

    strengths = detect_strengths(
        full_text
    )

    detected_form = detect_form(
        full_text
    )

    candidates = []

    # Prekės ženklo palyginimui pirmiausia naudojame
    # trumpesnį VVKT pavadinimą iki stiprumo/skaičių.
    for idx, row in df.iterrows():
        name = row["_name_norm"]

        if not name:
            continue

        name_without_numbers = re.split(
            r"\b\d",
            name,
            maxsplit=1,
        )[0].strip()

        if not name_without_numbers:
            name_without_numbers = name

        name_words = [
            word
            for word in name_without_numbers.split()
            if len(word) >= 4
        ]

        name_score = 0.0
        best_fragment = ""

        # Tikslus / dalinis pavadinimo signalas
        for fragment in fragments:
            if len(fragment) < 4:
                continue

            if (
                fragment == name_without_numbers
                or fragment in name_without_numbers
                or name_without_numbers in fragment
            ):
                score = 1.0
            else:
                score = SequenceMatcher(
                    None,
                    fragment,
                    name_without_numbers,
                ).ratio()

                # Palyginame ir su atskirais
                # preparato pavadinimo žodžiais.
                for word in name_words:
                    word_score = SequenceMatcher(
                        None,
                        fragment,
                        word,
                    ).ratio()

                    if word_score > score:
                        score = word_score

            if score > name_score:
                name_score = score
                best_fragment = fragment

        # Per silpnų pavadinimų net nevertiname toliau.
        # Tai apsaugo nuo atsitiktinių VVKT kandidatų.
        if name_score < 0.68:
            continue
        strength_score = 0.0

        if strengths:
            vvkt_strength = row["_strength_norm"]

            for strength in strengths:
                if (
                    strength
                    and strength in vvkt_strength
                ):
                    strength_score = 1.0
                    break

        form_score = 0.0

        if detected_form:
            vvkt_form = row["_form_norm"]

            form_root = normalize_text(
                detected_form
            )[:5]

            if (
                form_root
                and form_root in vvkt_form
            ):
                form_score = 1.0

        # Pavadinimas svarbiausias.
        # Stiprumas ir forma tik sustiprina kandidatą.
        total_score = (
            name_score * 0.80
            + strength_score * 0.15
            + form_score * 0.05
        )

        candidates.append(
            {
                "Atitikimo balas": round(
                    total_score * 100,
                    1,
                ),
                "Pavadinimo balas": round(
                    name_score * 100,
                    1,
                ),
                "OCR fragmentas": best_fragment,
                "Preparatas": row[
                    "preparato_pav"
                ],
                "Veiklioji medžiaga": row[
                    "veiklioji_medz_lt"
                ],
                "Stiprumas": row[
                    "stiprumas"
                ],
                "Farmacinė forma": row[
                    "farmacine_forma_lt"
                ],
            }
        )

    candidates.sort(
        key=lambda x: (
            x["Atitikimo balas"],
            x["Pavadinimo balas"],
        ),
        reverse=True,
    )

    # Tas pats preparatas VVKT gali turėti kelias
    # pakuotes. Testui paliekame unikalesnius variantus.
    unique = []
    seen = set()

    for candidate in candidates:
        key = (
            candidate["Preparatas"],
            candidate["Veiklioji medžiaga"],
            candidate["Stiprumas"],
            candidate["Farmacinė forma"],
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(candidate)

        if len(unique) >= max_results:
            break

    return (
        unique,
        strengths,
        detected_form,
    )


# ---------------------------------------------------------
# Programos sąsaja
# ---------------------------------------------------------

try:
    vvkt_df = load_vvkt_data()

    st.caption(
        f"VVKT įrašų įkelta: "
        f"{len(vvkt_df):,}"
    )

except Exception as exc:
    st.error(
        f"Nepavyko įkelti VVKT duomenų: {exc}"
    )
    st.stop()


uploaded = st.file_uploader(
    "Įkelkite vaisto pakuotės nuotrauką",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


if uploaded:
    image = ImageOps.exif_transpose(
        Image.open(uploaded)
    ).convert("RGB")

    st.image(
        image,
        caption="Testuojama nuotrauka",
        width=430,
    )

    if st.button(
        "🔬 Paleisti PaddleOCR + VVKT",
        type="primary",
    ):
        with st.spinner(
            "PaddleOCR analizuoja nuotrauką..."
        ):
            # OCR objektą įkeliame prieš matuojant
            # pačią nuotraukos analizę.
            ocr = load_paddle_ocr()

            start = time.perf_counter()

            result = ocr.predict(
                np.asarray(image)
            )

            elapsed = (
                time.perf_counter()
                - start
            )

        found_text = extract_paddle_text(
            result
        )

        st.success(
            f"PaddleOCR analizė baigta "
            f"per {elapsed:.2f} s"
        )

        st.subheader(
            "1. PaddleOCR perskaitytas tekstas"
        )

        if found_text:
            st.text(
                "\n".join(found_text)
            )

            (
                candidates,
                strengths,
                detected_form,
            ) = find_vvkt_candidates(
                vvkt_df,
                found_text,
            )

            st.subheader(
                "2. Iš OCR aptikti požymiai"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    "**Stiprumas:**",
                    ", ".join(strengths)
                    if strengths
                    else "neaptiktas",
                )

            with col2:
                st.write(
                    "**Farmacinė forma:**",
                    detected_form
                    if detected_form
                    else "neaptikta",
                )

            st.subheader(
                "3. Geriausi VVKT kandidatai"
            )

            if candidates:
                candidate_df = pd.DataFrame(
                    candidates
                )

                st.dataframe(
                    candidate_df,
                    use_container_width=True,
                    hide_index=True,
                )

                best = candidates[0]

                if (
                    best["Atitikimo balas"] >= 85
                    and
                    best["Pavadinimo balas"] >= 80
                ):
                    st.success(
                        "Stiprus testinis VVKT "
                        "atitikimas: "
                        f"{best['Preparatas']}"
                    )

                elif (
                    best["Atitikimo balas"] >= 70
                ):
                    st.warning(
                        "Yra tikėtinas VVKT "
                        "kandidatas, bet rezultatą "
                        "reikia patvirtinti: "
                        f"{best['Preparatas']}"
                    )

                else:
                    st.warning(
                        "Patikimo VVKT atitikimo "
                        "nenustatyta. Rodomi tik "
                        "galimi kandidatai."
                    )

            else:
                st.warning(
                    "Pagal PaddleOCR tekstą "
                    "VVKT kandidatų nerasta."
                )

        else:
            st.warning(
                "PaddleOCR teksto negrąžino."
            )

            with st.expander(
                "Žalias PaddleOCR rezultatas"
            ):
                st.write(result)
