from demo_risk_rules import analyze as analyze_demo_risk
from demo_saveikos import find_demo_pair
from porines_saveikos_saltiniai import show_pair_result
from pathlib import Path
import json
import re
import unicodedata
import hashlib
from difflib import get_close_matches, SequenceMatcher

import joblib
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PaddleOCR = None
    PADDLE_AVAILABLE = False

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

# Bandomieji VAPRIS dokumentų 4.5 skyriaus tekstai (nepatvirtinti).
pcs_interactions = load_csv("saveikos_rezultatai_v2_NAUJAS_TEST.csv")

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


def show_vvkt_info(name, selected_row=None):

    row = selected_row if selected_row is not None else get_vvkt_row(name)

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


def show_selected_vvkt_info(chosen):
    """Rodo pasirinktą VVKT kandidatą, o ne pirmą to paties pavadinimo eilutę."""
    for label, key in (("Pavadinimas", "name"), ("Veiklioji medžiaga", "ingredient"),
                       ("Stiprumas", "strength"), ("Farmacinė forma", "form"),
                       ("Vartojimo būdas", "route")):
        st.write(f"**{label}:**", chosen.get(key) or "nenurodyta")
    st.caption("Patvirtinti pasirinkto VVKT kandidato duomenys. Jei OCR ir VVKT nesutampa, būtina papildoma patikra.")


def manual_vvkt_options(name):
    """Atskiria to paties pavadinimo VVKT pakuočių variantus."""
    rows = get_vvkt_rows(name)
    options = {}
    for _, row in rows.iterrows():
        chosen = {
            "name": clean_value(row.get("preparato_pav")),
            "ingredient": clean_value(row.get("veiklioji_medz_lt")),
            "strength": clean_value(row.get("stiprumas")),
            "form": clean_value(row.get("farmacine_forma_lt")),
            "route": clean_value(row.get("vartojimo_budas")),
        }
        label = " | ".join((chosen["strength"], chosen["form"], chosen["ingredient"], chosen["route"]))
        options[label] = chosen
    return options


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
    ingredient,
    suppress_missing=False
):

    info = get_ingredient_knowledge(
        ingredient
    )

    st.markdown(
        f"#### {drug_name} – "
        f"{ingredient or 'veiklioji medžiaga nenustatyta'}"
    )

    if info is None:

        if suppress_missing:
            return False

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
# BANDOMOJO VAPRIS SĄVEIKŲ FAILO PERŽIŪRA
# =========================================================

def show_test_document_interactions(drug_name):
    """Rodo tik vienareikšmiškai susietą bandomąjį tekstą; nevertina vaistų poros."""
    if pcs_interactions.empty:
        st.caption("Bandomasis sąveikų CSV failas neįkeltas arba tuščias.")
        return

    required = {"Pavadinimas", "4.5_Saveika"}
    if not required.issubset(pcs_interactions.columns):
        st.warning("Bandomojo CSV struktūra neatitinka laukiamų stulpelių.")
        return

    target = normalize_text(drug_name)
    matches = pcs_interactions[
        pcs_interactions["Pavadinimas"].fillna("").astype(str)
        .apply(normalize_text) == target
    ]
    if matches.empty:
        st.caption(f"{drug_name}: bandomajame faile informacijos nerasta.")
        return

    # Skirtingi dokumentų tekstai tam pačiam pavadinimui gali reikšti
    # skirtingus stiprumus ar formas. Neparenkame atsitiktinės eilutės.
    texts = [str(v).strip() for v in matches["4.5_Saveika"].dropna()
             if str(v).strip()]
    unique_texts = list(dict.fromkeys(texts))
    if len(unique_texts) != 1:
        st.warning(f"{drug_name}: rasti keli skirtingi arba nepakankami "
                   "dokumentų įrašai. Reikalinga rankinė patikra.")
        return

    # Saugiklis prieš ankstesnę ACC / Accord klaidą.
    if target in {"acc", "acc long"} and any(
        phrase in normalize_text(unique_texts[0])
        for phrase in ("abacavir/lamivudine", "abakaviro ir lamivudino")
    ):
        st.error("Dokumento tapatybės neatitikimas – tekstas nerodomas.")
        return

    st.markdown(f"**{drug_name} – dokumente pateiktas 4.5 skyriaus tekstas**")
    st.text(unique_texts[0])
    st.caption("Bandomasis iš VAPRIS surinktas tekstas. Dokumento tipas, "
               "veiklioji medžiaga, stiprumas ir forma dar nėra "
               "nepriklausomai patvirtinti. Tai nėra šių dviejų "
               "preparatų sąveikos įvertinimas.")


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


def prepare_fast_ocr_image(image):
    """
    Paruošia telefono nuotrauką greitam OCR.
    """
    base = ImageOps.exif_transpose(image).convert("RGB")

    width, height = base.size
    max_side = 1600

    if max(width, height) > max_side:
        scale = max_side / max(width, height)

        base = base.resize(
            (
                max(1, int(width * scale)),
                max(1, int(height * scale))
            ),
            Image.Resampling.LANCZOS
        )

    return base


def run_fast_ocr(image):
    """
    Pirmas, greitas OCR etapas.
    """
    base = prepare_fast_ocr_image(image)

    gray = ImageOps.grayscale(base)
    gray = ImageOps.autocontrast(gray)

    try:
        text = pytesseract.image_to_string(
            gray,
            config="--psm 11"
        )

        return text.strip()

    except Exception:
        return ""

@st.cache_resource
def load_paddle_ocr():
    """
    Lengvesnis PaddleOCR variantas Streamlit Cloud aplinkai.
    Modelis įkeliamas tik tada, kai Tesseract rezultato nepakanka.
    """
    if not PADDLE_AVAILABLE:
        return None

    try:
        return PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu"
        )
    except Exception:
        return None


def run_paddle_ocr(image):
    """
    PaddleOCR etapas sudėtingesnėms telefono nuotraukoms.

    Naudojami keli tikslingi vaizdo variantai:
    1. originaliai paruošta nuotrauka;
    2. padidinto kontrasto nuotrauka;
    3. 90 laipsnių pasukta nuotrauka.

    Paddle modelis įkeliamas tik vieną kartą.
    """

    ocr = load_paddle_ocr()

    if ocr is None:
        return ""

    base = prepare_fast_ocr_image(image)

    # -------------------------------------------------
    # 1. ORIGINALUS VAIZDAS
    # -------------------------------------------------

    variants = [
        base
    ]

    # -------------------------------------------------
    # 2. KONTRASTINGESNIS VAIZDAS
    # -------------------------------------------------

    gray = ImageOps.grayscale(
        base
    )

    gray = ImageOps.autocontrast(
        gray
    )

    contrast = ImageEnhance.Contrast(
        gray
    ).enhance(1.6)

    # PaddleOCR gauna RGB vaizdą.
    contrast = contrast.convert("RGB")

    variants.append(
        contrast
    )

    # -------------------------------------------------
    # 3. PASUKTAS VAIZDAS
    # -------------------------------------------------

    rotated = base.rotate(
        90,
        expand=True
    )

    variants.append(
        rotated
    )

    # -------------------------------------------------
    # OCR
    # -------------------------------------------------

    texts = []

    for variant in variants:

        try:

            results = ocr.predict(
                np.asarray(variant)
            )

        except Exception:
            continue

        for item in results:

            try:

                data = item.json

                if callable(data):
                    data = data()

                if isinstance(data, str):
                    data = json.loads(data)

                if not isinstance(data, dict):
                    continue

                result_data = data.get(
                    "res",
                    data
                )

                rec_texts = result_data.get(
                    "rec_texts",
                    []
                )

                for text in rec_texts:

                    text = str(
                        text
                    ).strip()

                    if (
                        text
                        and text not in texts
                    ):
                        texts.append(
                            text
                        )

            except Exception:
                continue

    return "\n".join(
        texts
    )
def run_fallback_ocr(image, fast_text=""):
    """
    Papildomas OCR naudojamas tik tada,
    kai greitas OCR nerado tinkamo VVKT kandidato.

    Fallback:
    - sutvarko telefono orientaciją;
    - tikrina 0 / 90 / 180 / 270 laipsnių kryptis;
    - naudoja sustiprintą kontrastą;
    - papildomai naudoja paryškintą vaizdą;
    - bando PSM 11 ir PSM 6.
    """

    base = prepare_fast_ocr_image(image)

    texts = []

    if fast_text:
        texts.append(fast_text)

    # -------------------------------------------------
    # 1. ORIENTACIJOS
    # -------------------------------------------------

    for angle in (0, 90, 180, 270):

        rotated = base.rotate(
            angle,
            expand=True
        )

        gray = ImageOps.grayscale(
            rotated
        )

        processed = ImageOps.autocontrast(
            gray
        )

        processed = ImageEnhance.Contrast(
            processed
        ).enhance(1.8)

        # PSM 11 gerai tinka išmėtytam tekstui ant pakuotės.
        try:
            text = pytesseract.image_to_string(
                processed,
                config="--psm 11"
            ).strip()

            if text:
                texts.append(text)

        except Exception:
            pass

    # -------------------------------------------------
    # 2. PAPILDOMAS BANDYMAS NORMALIA ORIENTACIJA
    # -------------------------------------------------
    # Jei prekės pavadinimas yra didesnis ir išdėstytas
    # kaip vientisas teksto blokas, PSM 6 kartais jį
    # perskaito geriau nei PSM 11.
    #
    # Šito nedarome visoms 4 orientacijoms, kad OCR
    # vėl netaptų labai lėtas.

    gray = ImageOps.grayscale(
        base
    )

    block_image = ImageOps.autocontrast(
        gray
    )

    block_image = ImageEnhance.Contrast(
        block_image
    ).enhance(2.0)

    block_image = ImageEnhance.Sharpness(
        block_image
    ).enhance(1.8)

    try:
        text = pytesseract.image_to_string(
            block_image,
            config="--psm 6"
        ).strip()

        if text:
            texts.append(text)

    except Exception:
        pass

    # -------------------------------------------------
    # PAŠALINAME PASIKARTOJANTĮ OCR TEKSTĄ
    # -------------------------------------------------

    unique_texts = list(
        dict.fromkeys(
            text
            for text in texts
            if text
        )
    )

    return "\n".join(
        unique_texts
    )


def run_ocr(image):
    """
    Suderinamumo funkcija.
    Pagal nutylėjimą atliekamas greitas OCR.
    """
    return run_fast_ocr(image)

def retry_ocr_from_package(image, previous_text, previous_candidates):
    """Pakartotinis OCR su pasvirimo korekcija; pradinis rezultatas saugomas.

    Skaito tik pareikalavus naudotojui. Priimamas tik VVKT kandidatas,
    kurio pilnas prekės pavadinimas pažodžiui perskaitytas iš vaizdo.
    Taip vien veikliosios medžiagos sutapimas netampa vaisto tapatybe.
    """
    # Jei senasis OCR jau tiksliai perskaitė preparato pavadinimą,
    # jo nekeičiam. Tai apsaugo Ibuprom, Panadol ir kitus gerus atvejus.
    previous_normalized = normalize_text(previous_text)
    for old in previous_candidates or []:
        old_name = normalize_text(old.get("name", ""))
        if len(old_name) >= 5 and re.search(
            r"(?<![a-z0-9])" + re.escape(old_name) + r"(?![a-z0-9])",
            previous_normalized,
        ) and float(old.get("name_score", 0)) >= 0.90:
            return previous_text, previous_candidates, False

    base = prepare_fast_ocr_image(image)
    # Lietuviškų vaistų pakuočių nuotraukos dažnai pasvirusios 15–35°.
    # Baltas fonas pasukant svarbus, kad Tesseract neskaitytų juodų kampų.
    # PSM 11 skaito išsklaidytus užrašus, PSM 6 – pavadinimo eilutę.
    variants = [(base, 0)]
    for angle in (25, -25, 15, -15, 35, -35):
        variants.append((base, angle))
    # Jei visas kadras nepadeda, patikriname tik centrinę dalį (be stalo).
    w, h = base.size
    if min(w, h) >= 180:
        region = base.crop((int(w * 0.07), int(h * 0.07),
                            int(w * 0.93), int(h * 0.93)))
        variants.extend((region, angle) for angle in (25, -25, 35, -35))

    previous_best_score = max(
        (float(x.get("score", 0)) for x in (previous_candidates or [])),
        default=0.0,
    )
    best = None
    for region, angle in variants:
        if angle:
            region = region.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        gray = ImageOps.autocontrast(ImageOps.grayscale(region))
        # Teksto raidės turi būti pakankamo dydžio, bet ribojame RAM.
        longest = max(gray.size)
        if longest < 1450:
            scale = min(2.0, 1450.0 / max(1, longest))
            gray = gray.resize(
                (max(1, int(gray.width * scale)),
                 max(1, int(gray.height * scale))),
                Image.Resampling.LANCZOS,
            )
        for psm in (11, 6):
            try:
                text = pytesseract.image_to_string(
                    gray, config=f"--psm {psm}", timeout=8
                ).strip()
            except Exception:
                continue
            if not text:
                continue
            normalized = normalize_text(text)
            # Pirmiausia ieškome TIK pilnai perskaitytų prekės pavadinimų.
            # Nekviečiame brangios VVKT analizės atsitiktiniam OCR triukšmui.
            tokens = set(normalized.split())
            exact_names = []
            for name in vvkt_names:
                clean_name = normalize_text(name)
                if len(clean_name) < 5 or clean_name.split()[0] not in tokens:
                    continue
                if re.search(
                    r"(?<![a-z0-9])" + re.escape(clean_name) + r"(?![a-z0-9])",
                    normalized,
                ):
                    exact_names.append(name)
            if not exact_names:
                continue
            candidates = rank_vvkt_candidates(text, top_n=5)
            for candidate in candidates:
                if candidate.get("name") not in exact_names:
                    continue
                if float(candidate.get("name_score", 0)) < 0.90:
                    continue
                if float(candidate.get("score", 0)) < 0.55:
                    continue
                if candidate.get("ocr_strength_match") is False:
                    continue
                if candidate.get("ocr_form_match") is False:
                    continue
                score = float(candidate.get("score", 0))
                # Nekeisti ankstesnio bent jau pakankamai gero rezultato
                # kitu vaistu, ypač dėl vieno atsitiktinio OCR fragmento.
                # Senasis kandidatas gali turėti aukštą balą vien dėl
                # veikliosios medžiagos (pvz., Pantoprazole-Mepha).
                # Pilnas kito pavadinimo OCR įrodymas turi pirmenybę;
                # tiksliai perskaityti seni pavadinimai apsaugoti aukščiau.
                if best is None or score > best[0]:
                    chosen_first = [candidate] + [
                        x for x in candidates if x is not candidate
                    ]
                    best = (score, text, chosen_first)
            # Aukšto patikimumo pilnas pavadinimo sutapimas – baigiame,
            # kad nereikėtų veltui vykdyti visų nuskaitymo variantų.
            if best is not None and best[0] >= 0.95:
                return best[1], best[2], True
    if best is None:
        return previous_text, previous_candidates, False
    return best[1], best[2], True



def ocr_read_brand_name(ocr_text, known_names):
    """Rodo tik pažodžiui OCR tekste aptiktą VVKT preparato pavadinimą.

    Tai NĖRA vaisto tapatybės patvirtinimas ar naujas kandidatų reitingavimas.
    Saugumo sumetimais reikalaujama pilno pavadinimo žodžių ribose.
    """
    if not ocr_text:
        return None
    # Tarp dviejų OCR eilučių gali būti atskirti to paties vardo žodžiai:
    # IBUPROM\nEXPRESS. Normalizuojame visą OCR tekstą, bet reikalaujame
    # pilno, žodžių ribomis apriboto vardo, ne fuzzy spėjimo.
    whole = normalize_text(ocr_text)
    possible = []
    for name in known_names:
        word = normalize_text(name)
        if len(word) < 5 or not re.search(r"[a-z]", word):
            continue
        expression = re.compile(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])")
        if expression.search(whole):
            possible.append(name)
    # Jei visame tekste keli pavadinimai, renkamės tik unikalų ilgiausią
    # ištisą vardą, kai kiti yra jo dalys (ne skirtingi preparatai).
    possible = list(dict.fromkeys(possible))
    if len(possible) == 1:
        return possible[0]
    if possible:
        longest = max(possible, key=lambda x: len(normalize_text(x)))
        if all(normalize_text(x) in normalize_text(longest) for x in possible):
            return longest
    return None

def ocr_strengths_unique(values):
    """Sujungia vienodą OCR stiprumą (pvz. 400mg ir 400 mg)."""
    out = []
    seen = set()
    for value in values or []:
        normalized = normalize_strength_value(value)
        if normalized not in seen:
            seen.add(normalized)
            pretty = re.sub(r"\s*([a-zA-Z]+)$", r" \1", str(value).strip())
            out.append(pretty)
    return out


def vvkt_ocr_discrepancies(candidate, detected_strengths, detected_form):
    """Nežinoma OCR reikšmė nėra neatitikimas. Tikrina tik realų prieštaravimą.

    Stiprumo neatitikimas yra tik kai rastas bent vienas stiprumas,
    tačiau nė vienas nesutampa su pasirinkta VVKT eilute. Formų
    neatitikimą skaičiuojame tik kai OCR formą iš tikrųjų nustatė.
    """
    problems = []
    strengths = ocr_strengths_unique(detected_strengths)
    candidate_strength = candidate.get("strength") or ""
    if strengths and candidate_strength and candidate_strength != "—":
        if not any(
            normalize_strength_value(value) == normalize_strength_value(candidate_strength)
            or strength_score(candidate_strength, value) >= 0.95
            for value in strengths
        ):
            problems.append("Stiprumas")
    if detected_form and candidate.get("form") and candidate["form"] != "—":
        if not forms_compatible(detected_form, candidate["form"]):
            problems.append("Farmacinė forma")
    return problems


def show_candidate_verification(candidate, detected_strengths, detected_form, slot):
    """Neužblokuoja rankinio patvirtinimo vien dėl nepatikimo OCR.

    Tikras OCR ir VVKT prieštaravimas reikalauja atskiro naudotojo
    patvirtinimo, kad jis rankiniu būdu patikrino BŪTENT šį variantą.
    """
    problems = vvkt_ocr_discrepancies(candidate, detected_strengths, detected_form)
    name = candidate.get("name") or "nenurodyta"
    if not problems:
        if not detected_strengths or not detected_form:
            st.caption(
                "OCR dalies informacijos nenustatė. Tai nėra neatitikimas. "
                "Prieš patvirtindami palyginkite pakuotę su VVKT pavadinimu, "
                "stiprumu ir farmacine forma."
            )
        return True
    st.warning(
        "OCR ir pasirinkto VVKT varianto galimas neatitikimas: "
        + ", ".join(problems)
        + ". OCR gali klysti; vien algoritmo balas nėra tapatybės įrodymas."
    )
    return st.checkbox(
        "Patikrinau ANT PAKUOTĖS, kad tai tikrai "
        f"{name} | {candidate.get('strength', '—')} | "
        f"{candidate.get('form', '—')}; patvirtinu rankiniu būdu.",
        value=False,
        key=f"{slot}_verify_{hashlib.sha256(repr((name, candidate.get('strength'), candidate.get('form'))).encode()).hexdigest()[:12]}",
    )


# =========================================================
# OCR – STIPRUMO ATPAŽINIMAS
# =========================================================

def extract_strengths(text):
    """Atpažįsta mg/ml koncentracijas, neįtraukiant pakuotės tūrio į stiprumus.

    Funkcija nekuria /ml, jeigu OCR šio santykio neperskaitė.
    """
    normalized = normalize_text(text).replace(",", ".")
    # Koncentracijų alternatyva PRIVALO būti prieš paprastą mg alternatyvą.
    pattern = (
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g)"
        r"\s*/\s*(?:\d+(?:\.\d+)?\s*)?ml\b"
        r"|\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g)\b"
    )
    found = []
    for item in re.findall(pattern, normalized):
        item = re.sub(r"\s*/\s*", "/", item)
        item = re.sub(r"\s+", " ", item).strip()
        if item not in found:
            found.append(item)
    return found


# =========================================================
# OCR – FARMACINĖS FORMOS ATPAŽINIMAS
# =========================================================

def detect_form(text):

    t = normalize_text(
        text
    )

    # Geriamoji suspensija nėra geriamasis tirpalas.
    # Formą nustatome tik iš OCR perskaityto teksto.
    if "suspensij" in t and "geriam" in t:
        return "geriamoji suspensija"

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
            cutoff=0.82
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
# OCR PAVADINIMO KANDIDATAI – PATOBULINTA VERSIJA
# =========================================================

def get_ocr_lines(text):
    """
    Grąžina prasmingas OCR eilutes.
    Pakuotės prekės ženklas dažnai būna atskiroje eilutėje.
    """

    lines = []

    for line in text.splitlines():

        line = normalize_text(line)

        if not line:
            continue

        # Ignoruojame labai trumpą triukšmą
        if len(line) < 3:
            continue

        lines.append(line)

    return list(dict.fromkeys(lines))

GENERIC_OCR_WORDS = {
    # Farmacinės formos
    "tablete",
    "tabletes",
    "tableciu",
    "tablet",
    "tablets",
    "kapsule",
    "kapsules",
    "kapsuliu",
    "capsule",
    "capsules",
    "sirupas",
    "syrup",
    "tirpalas",
    "solution",
    "gelis",
    "kremas",
    "cream",
    "tepalas",
    "milteliai",
    "granules",

    # Pakuotės / vartojimo žodžiai
    "plevele",
    "dengtos",
    "dengta",
    "geriamasis",
    "geriamieji",
    "injekcinis",
    "modified",
    "release",
    "prolonged",

    # Dažni cheminiai / ingredientų žodžiai,
    # kurie vieni nėra prekės ženklas
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

    # Matavimo vienetai
    "mg",
    "mcg",
    "ug",
    "ml"
}


def is_useful_brand_fragment(fragment):
    fragment = normalize_text(fragment)

    if not fragment:
        return False

    if len(fragment) < 4:
        return False

    # Vien skaičiai, dozės ir simboliai
    if re.fullmatch(
        r"[\d\s.,/%+-]+",
        fragment
    ):
        return False

    words = fragment.split()

    if not words:
        return False

    # Jei visas fragmentas sudarytas tik iš bendrinių
    # farmacinių žodžių, jis nėra prekės ženklo įrodymas.
    if all(
        word in GENERIC_OCR_WORDS
        for word in words
    ):
        return False

    return True
def get_brand_fragments(text):
    """
    Sukuria fragmentus, kurie labiausiai tinka
    preparato prekės ženklo paieškai.
    """

    fragments = []

    for line in get_ocr_lines(text):

        # Visa OCR eilutė
        if (
            3 <= len(line) <= 50
            and is_useful_brand_fragment(line)
        ):
            fragments.append(line)

        words = line.split()

        # Atskiri žodžiai
        for word in words:

            # Preparatų pavadinimams trumpesni nei
            # 4 simbolių fragmentai per daug nepatikimi
            if (
                len(word) >= 4
                and is_useful_brand_fragment(word)
            ):
                fragments.append(word)

        # 2 ir 3 žodžių preparatų pavadinimai
        for size in (2, 3):

            if len(words) < size:
                continue

            for i in range(
                len(words) - size + 1
            ):

                fragment = " ".join(
                    words[i:i + size]
                )

                if (
                    len(fragment) >= 5
                    and is_useful_brand_fragment(fragment)
                ):
                    fragments.append(fragment)

    return list(
        dict.fromkeys(fragments)
    )


def brand_similarity(
    drug_name,
    fragment
):
    """
    Palygina VVKT preparato pavadinimą
    su OCR fragmentu.
    """

    drug = normalize_text(drug_name)
    fragment = normalize_text(fragment)

    if not drug or not fragment:
        return 0.0

    # Tikslus atitikimas
    if drug == fragment:
        return 1.0

    # Visas preparato pavadinimas perskaitytas
    # ilgesnėje OCR eilutėje
    if (
        len(drug) >= 4
        and drug in fragment
    ):
        return 0.99

    # OCR eilutė yra preparato pavadinimo dalis
    if (
        len(fragment) >= 5
        and fragment in drug
    ):

        ratio = len(fragment) / len(drug)

        if ratio >= 0.75:
            return 0.92

    score = similarity(
        drug,
        fragment
    )

    # Labai trumpiems pavadinimams fuzzy
    # atitikimą vertiname konservatyviau
    if len(drug) <= 4:

        if score < 0.90:
            return 0.0

    elif len(drug) <= 6:

        if score < 0.78:
            return 0.0

    else:

        if score < 0.68:
            return 0.0

    return score


def find_name_candidates(
    ocr_text,
    max_names=50
):
    """
    Greita dviejų pakopų VVKT preparato pavadinimo paieška.

    1. Pirmiausia atliekama labai pigi tiksli / dalinė paieška.
    2. Fuzzy palyginimas vykdomas tik su sumažintu
       galimų VVKT pavadinimų rinkiniu.

    Taip išvengiama situacijos, kai kiekvienas OCR fragmentas
    lyginamas su visu VVKT preparatų katalogu.
    """

    fragments = get_brand_fragments(
        ocr_text
    )

    if not fragments:
        return []

    # -----------------------------------------------------
    # PARUOŠIAME PRASMINGUS OCR FRAGMENTUS
    # -----------------------------------------------------

    useful_fragments = []

    for fragment in fragments:

        fragment = normalize_text(
            fragment
        )

        if len(fragment) < 4:
            continue

        # Skaičiai / stiprumai nėra geri prekės ženklo
        # paieškos signalai.
        if re.fullmatch(
            r"[\d\s.,/%+-]+",
            fragment
        ):
            continue

        useful_fragments.append(
            fragment
        )

    useful_fragments = list(
        dict.fromkeys(
            useful_fragments
        )
    )

    if not useful_fragments:
        return []

    # -----------------------------------------------------
    # 1. GREITA TIKSLI / DALINĖ PAIEŠKA
    # -----------------------------------------------------

    exact_scores = {}

    # Normalizuojame OCR tekstą vieną kartą.
    normalized_ocr = normalize_text(
        ocr_text
    )

    for name in vvkt_names:

        drug = normalize_text(
            name
        )

        if len(drug) < 3:
            continue

        # Visas VVKT pavadinimas aiškiai matomas OCR tekste.
        if (
            len(drug) >= 4
            and drug in normalized_ocr
        ):

            exact_scores[name] = 1.0
            continue

        # Tikriname, ar OCR fragmentas yra didelė
        # preparato pavadinimo dalis.
        for fragment in useful_fragments:

            if (
                len(fragment) >= 5
                and fragment in drug
            ):

                coverage = (
                    len(fragment)
                    / max(len(drug), 1)
                )

                if coverage >= 0.75:

                    exact_scores[name] = max(
                        exact_scores.get(
                            name,
                            0.0
                        ),
                        0.92
                    )

                    break

    # Jei turime aiškių pavadinimo atitikimų,
    # brangaus fuzzy etapo apskritai nereikia.
    if exact_scores:

        ranked = sorted(
            exact_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            name
            for name, _
            in ranked[:max_names]
        ]

    # -----------------------------------------------------
    # 2. SUMAŽINAME VVKT KANDIDATŲ RINKINĮ
    # -----------------------------------------------------

    # Fuzzy palyginimui neimsime viso registro.
    # Kandidatą paliekame tik tada, kai jo pradžia / žodžio
    # pradžia bent apytiksliai sutampa su OCR fragmentu.

    reduced_names = []

    fragment_prefixes = set()

    for fragment in useful_fragments:

        for word in fragment.split():

            if len(word) >= 4:

                fragment_prefixes.add(
                    word[:3]
                )

    for name in vvkt_names:

        drug = normalize_text(
            name
        )

        if not drug:
            continue

        drug_words = drug.split()

        prefixes = {
            word[:3]
            for word in drug_words
            if len(word) >= 3
        }

        if (
            prefixes
            and fragment_prefixes
            and prefixes.intersection(
                fragment_prefixes
            )
        ):

            reduced_names.append(
                name
            )

    # OCR kartais suklysta jau pirmose raidėse.
    # Tokiu atveju naudojame konservatyvų atsarginį
    # kandidatų sąrašą pagal pirmąją raidę.

    if not reduced_names:

        first_letters = {
            fragment[0]
            for fragment in useful_fragments
            if fragment
        }

        for name in vvkt_names:

            drug = normalize_text(
                name
            )

            if (
                drug
                and drug[0] in first_letters
            ):

                reduced_names.append(
                    name
                )

            # Neleidžiame atsarginiam rinkiniui
            # vėl išaugti iki viso VVKT katalogo.
            if len(reduced_names) >= 1500:
                break

    # -----------------------------------------------------
    # 3. FUZZY TIK SUMAŽINTAM RINKINIUI
    # -----------------------------------------------------

    scores = {}

    for name in reduced_names:

        best = 0.0

        for fragment in useful_fragments:

            score = brand_similarity(
                name,
                fragment
            )

            if score > best:
                best = score

            if best >= 0.99:
                break

        if best >= 0.68:

            scores[name] = best

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
    """
    Preparato pavadinimas yra pagrindinis
    pakuotės identifikavimo požymis.
    """

    fragments = get_brand_fragments(
        ocr_text
    )

    best = 0.0

    for fragment in fragments:

        score = brand_similarity(
            name,
            fragment
        )

        best = max(
            best,
            score
        )

        if best >= 0.99:
            break

    return best


def normalize_strength_value(value):
    """
    Normalizuoja stiprumą palyginimui.
    Pvz. '30 mg' -> '30mg'
    """

    value = normalize_text(value)

    value = value.replace(
        ",",
        "."
    )

    value = re.sub(
        r"\s+",
        "",
        value
    )

    return value


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

    strength_norm = normalize_strength_value(
        strength
    )

    for item in detected:

        item_norm = normalize_strength_value(
            item
        )

        # Reikalaujame pilno stiprumo sutapimo
        # arba kad OCR reikšmė būtų aiški VVKT
        # stiprumo dalis.
        if item_norm == strength_norm:
            return 1.0

        if (
            len(item_norm) >= 3
            and item_norm in strength_norm
        ):
            return 0.95

    return 0.0


def ingredient_score(
    ingredient,
    ocr_text
):
    """
    Veiklioji medžiaga naudojama kaip
    papildomas patvirtinimas, bet ji negali
    viena pati nustelbti preparato pavadinimo.
    """

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

    if not ing:
        return 0.0

    # Tikslus veikliosios medžiagos tekstas
    if (
        len(ing) >= 5
        and ing in text
    ):
        return 1.0

    best = 0.0

    for fragment in build_ocr_fragments(
        ocr_text
    ):

        # Neleidžiame trumpiems OCR žodžiams,
        # pvz. atsitiktiniam "magnis",
        # sukurti labai stipraus įrodymo
        if len(fragment) < 7:
            continue

        score = similarity(
            ing,
            fragment
        )

        best = max(
            best,
            score
        )

    # Ingredientų fuzzy matching turi būti
    # daug griežtesnis nei anksčiau.
    if best < 0.78:
        return 0.0

    return best


def forms_compatible(ocr_form, vvkt_form):
    """Konservatyviai lygina OCR bazinę ir VVKT detalesnę formą.

    Platesnis VVKT aprašymas (pvz., skrandyje neirios minkštosios kapsulės)
    neprieštarauja OCR aptiktoms minkštosioms kapsulėms. Kapsulės ir
    tabletės nėra ekvivalentiškos. Kai OCR aptinka specifinę savybę,
    kurios VVKT nenurodo, automatiškai jos nepatvirtiname.
    """
    ocr = normalize_text(ocr_form)
    vvkt = normalize_text(vvkt_form)
    if not ocr or not vvkt:
        return False
    if ocr == vvkt:
        return True

    def category(value):
        if "kapsul" in value:
            if "minkst" in value:
                return "minkstosios kapsules"
            if "kiet" in value:
                return "kietosios kapsules"
            return "kapsules"
        if "tablet" in value:
            return "tabletes"
        return None

    ocr_category = category(ocr)
    vvkt_category = category(vvkt)
    if not ocr_category or ocr_category != vvkt_category:
        return False

    # Jei OCR atpažino specifinę savybę, VVKT taip pat ją turi nurodyti.
    qualifiers = ("skrandyje neir", "plevele dengt", "modifikuot", "pailgint")
    for qualifier in qualifiers:
        if qualifier in ocr and qualifier not in vvkt:
            return False
    # Bazinės formos atitikimas: 'minkštosios kapsulės' ->
    # 'skrandyje neirios minkštosios kapsulės' yra leidžiamas.
    if ocr == ocr_category:
        return True
    # Kitos formuluotės suderinamos tik jei OCR žodžiai randami VVKT.
    return all(token in vvkt.split() for token in ocr.split())


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

    if forms_compatible(detected, form):
        return 1.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.85

    score = similarity(
        a,
        b
    )

    if score < 0.60:
        return 0.0

    return score


# =========================================================
# VVKT KANDIDATŲ REITINGAVIMAS – PATOBULINTA VERSIJA
# =========================================================

def rank_vvkt_candidates(ocr_text, top_n=5):
    """Kandidatai vertinami pakuotės stiprumo ir formos kontekste.

    Nedaro išvados apie medikamentų suderinamumą ar vaisto tapatybę.
    Kiekviena VVKT eilutė vertinama atskirai, ne tik pavadinimo pirmoji.
    """
    if not ocr_text or not ocr_text.strip():
        return []
    candidate_names = find_name_candidates(ocr_text, max_names=100)
    found_ingredient = detect_ingredient(ocr_text)
    found_strengths = extract_strengths(ocr_text)
    found_form = detect_form(ocr_text)
    if found_ingredient and len(candidate_names) < 8:
        matched_rows = vvkt[
            vvkt["veiklioji_medz_lt"].fillna("").astype(str).apply(normalize_text)
            == normalize_text(found_ingredient)
        ]
        for value in matched_rows["preparato_pav"].dropna().astype(str).tolist():
            if value not in candidate_names:
                candidate_names.append(value)
    results = []
    seen = set()
    # OCR eilutės gali atskirti vieno prekės ženklo žodžius (IBUPROM / EXPRESS).
    # Patikriname visą VVKT pavadinimą visame OCR tekste, nekeisdami OCR.
    ocr_whole = normalize_text(ocr_text)
    for name in candidate_names:
        ns = name_score(name, ocr_text)
        full_name = normalize_text(name)
        if len(full_name.split()) >= 2 and re.search(
            r"(?<![a-z0-9])" + re.escape(full_name) + r"(?![a-z0-9])", ocr_whole
        ):
            ns = max(ns, 1.0)
        if ns < 0.55:
            continue
        rows = get_vvkt_rows(name)
        for _, row in rows.iterrows():
            strength = clean_value(row.get("stiprumas"))
            form = clean_value(row.get("farmacine_forma_lt"))
            ingredient = clean_value(row.get("veiklioji_medz_lt"))
            route = clean_value(row.get("vartojimo_budas"))
            identity = (name, strength, form, ingredient, route)
            if identity in seen:
                continue
            seen.add(identity)
            ins = ingredient_score(ingredient, ocr_text)
            ss = strength_score(strength, ocr_text)
            fs = form_score(form, ocr_text)
            # Griežtas skaitinio stiprumo sutapimas, kai OCR jį aptiko.
            strength_match = bool(found_strengths) and any(
                normalize_strength_value(x) == normalize_strength_value(strength)
                for x in found_strengths
            )
            form_match = bool(found_form) and forms_compatible(found_form, form)
            score = ns * 0.65 + ins * 0.05 + ss * 0.20 + fs * 0.10
            if found_strengths:
                score += 0.22 if strength_match else -0.38
            if found_form:
                score += 0.12 if form_match else -0.22
            results.append({
                "name": name, "ingredient": ingredient, "strength": strength,
                "form": form, "route": route, "score": max(0.0, min(score, 1.0)),
                "name_score": ns, "ingredient_score": ins,
                "strength_score": ss, "form_score": fs,
                "ocr_strength_match": strength_match if found_strengths else None,
                "ocr_form_match": form_match if found_form else None,
            })
    # Kai OCR nustatė stiprumą / formą, neprieštaraujantis variantas turi pirmenybę.
    # Kitus variantus paliekame pasirinkimui tik jei nepakanka suderintų kandidatų.
    results.sort(key=lambda r: (
        1 if r["ocr_strength_match"] is True else (0 if r["ocr_strength_match"] is None else -1),
        1 if r["ocr_form_match"] is True else (0 if r["ocr_form_match"] is None else -1),
        r["score"], r["name_score"]
    ), reverse=True)
    return results[:top_n]

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

    # =====================================================
    # OCR REZULTATO IŠSAUGOJIMAS PAGAL NUOTRAUKĄ
    # =====================================================

    uploaded_bytes = uploaded.getvalue()

    image_key = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    cached_key = st.session_state.get(
        "ocr_image_key"
    )

    # OCR atliekame tik tada, kai įkelta nauja nuotrauka.
    if cached_key != image_key:
        st.session_state["first_vvkt_drug"] = None
        st.session_state["first_vvkt_selected"] = None
        st.session_state.pop("ocr_candidate", None)

        with st.spinner(
            "Analizuojama pakuotė..."
        ):

            # =============================================
            # 1 ETAPAS – GREITAS TESSERACT
            # =============================================

            fast_ocr_text = run_fast_ocr(
                image
            )

            fast_candidates = rank_vvkt_candidates(
                fast_ocr_text,
                top_n=5
            )

            fast_is_strong = False

            if fast_candidates:

                top_fast = fast_candidates[0]

                fast_is_strong = (
                    top_fast["name_score"] >= 0.90
                    and top_fast["score"] >= 0.75
                )

            # =============================================
            # JEI TESSERACT PAKANKAMAI STIPRUS – STOP
            # =============================================

            if fast_is_strong:

                ocr_text = fast_ocr_text
                ranked_candidates = fast_candidates
                ocr_engine = "Tesseract"

            else:

                # =========================================
                # 2 ETAPAS – PADDLEOCR
                # =========================================

                paddle_text = run_paddle_ocr(
                    image
                )

                combined_parts = []

                if fast_ocr_text:
                    combined_parts.append(
                        fast_ocr_text
                    )

                if paddle_text:
                    combined_parts.append(
                        paddle_text
                    )

                ocr_text = "\n".join(
                    dict.fromkeys(
                        combined_parts
                    )
                )

                ranked_candidates = rank_vvkt_candidates(
                    ocr_text,
                    top_n=5
                )

                if paddle_text:
                    ocr_engine = "Tesseract + PaddleOCR"
                else:
                    ocr_engine = "Tesseract"

        # =============================================
        # IŠSAUGOME ŠIOS NUOTRAUKOS REZULTATĄ
        # =============================================

        st.session_state[
            "ocr_image_key"
        ] = image_key

        st.session_state[
            "ocr_text_cached"
        ] = ocr_text

        st.session_state[
            "ocr_candidates_cached"
        ] = ranked_candidates

        st.session_state[
            "ocr_engine_cached"
        ] = ocr_engine

    else:

        # =============================================
        # TA PATI NUOTRAUKA – OCR NEKARTOJAME
        # =============================================

        ocr_text = st.session_state.get(
            "ocr_text_cached",
            ""
        )

        ranked_candidates = st.session_state.get(
            "ocr_candidates_cached",
            []
        )

        ocr_engine = st.session_state.get(
            "ocr_engine_cached",
            "Tesseract"
        )

    # Pasirenkamas, saugus pakartotinis OCR: ankstesnio rezultato nekeičia,
    # kol neaptiktas aiškus VVKT pavadinimas be prieštaravimų.
    if st.session_state.get("first_vvkt_drug") is None:
        if st.button("🔁 Pakartoti išsamesnį pirmos pakuotės OCR", key="retry_first_ocr"):
            with st.spinner("Papildomai skaitoma pakuotė..."):
                better_text, better_candidates, improved = retry_ocr_from_package(
                    image, ocr_text, ranked_candidates
                )
            if improved:
                ocr_text, ranked_candidates = better_text, better_candidates
                ocr_engine = "Pakartotinis Tesseract"
                st.session_state["ocr_text_cached"] = ocr_text
                st.session_state["ocr_candidates_cached"] = ranked_candidates
                st.session_state["ocr_engine_cached"] = ocr_engine
                st.session_state.pop("ocr_candidate", None)
                st.success("Papildomas OCR rado aiškesnį VVKT pavadinimą. Patikrinkite kandidatą.")
            else:
                st.info("Papildomas OCR rezultato nepagerino; ankstesnis atpažinimas išsaugotas. Jei reikia, naudokite rankinę VVKT paiešką.")

    strengths = extract_strengths(
        ocr_text
    )

    detected_form = detect_form(
        ocr_text
    )

    detected_ingredient = detect_ingredient(
        ocr_text
    )
    with st.expander("🔍 OCR perskaitytas tekstas"):
        st.caption(
            f"Naudotas OCR: {ocr_engine}"
        )
        st.text(
            ocr_text
            if ocr_text
            else "OCR teksto neaptiko."
        )
    st.markdown(
        "#### 🧾 OCR iš pirmos pakuotės perskaityta informacija"
    )

    st.write(
        "**Pavadinimas (perskaitytas OCR):**",
        ocr_read_brand_name(ocr_text, vvkt_names)
        or "OCR pavadinimo patikimai nenustatė"
    )

    st.write(
        "**Veiklioji medžiaga:**",
        detected_ingredient
        or "OCR nenustatė; galimo VVKT kandidato duomenys pateikiami žemiau"
    )

    st.write(
        "**Stiprumas:**",
        ", ".join(ocr_strengths_unique(strengths)[:3])
        if strengths
        else "OCR nenustatė; galimo VVKT kandidato stiprumas pateikiamas žemiau"
    )

    st.write(
        "**Farmacinė forma:**",
        detected_form
        or "OCR nenustatė; galimo VVKT kandidato duomenys pateikiami žemiau"
    )

    st.markdown(
        "#### 🔎 Galimi VVKT preparatai (duomenys nepatvirtinti, kol nepasirinkote preparato)"
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
                f"{candidate['strength']} | {candidate['form']} | "
                f"{candidate['score'] * 100:.0f}% algoritmo balas"
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

        if (ocr_read_brand_name(ocr_text, vvkt_names) == candidate.get("name")
                and float(candidate.get("score", 0)) >= 0.75):
            st.success(f"💊 VVKT kandidatas (patikrinkite): **{candidate['name']}**")
        else:
            st.warning(f"VVKT paieškos pasiūlymas, tapatybė nepatvirtinta: **{candidate['name']}**")

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

        first_confirmed_by_user = show_candidate_verification(
            candidate, strengths, detected_form, "first"
        )
        if st.button(
            "✅ Patvirtinti preparatą",
            type="primary",
            disabled=not first_confirmed_by_user
        ):

            st.session_state[
                "first_vvkt_drug"
            ] = candidate["name"]
            st.session_state["first_vvkt_selected"] = dict(candidate)

            st.success(
                f"✅ Patvirtinta: {candidate['name']}"
            )

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

        first_options = manual_vvkt_options(manual_first)
        first_variant = st.selectbox(
            "Pasirinkite tikslų stiprumą ir farmacinę formą",
            list(first_options),
            key="manual_first_variant",
        )
        first_chosen = first_options[first_variant]
        show_selected_vvkt_info(first_chosen)

        if st.button(
            "✅ Naudoti šį preparatą"
        ):

            st.session_state[
                "first_vvkt_drug"
            ] = manual_first
            st.session_state["first_vvkt_selected"] = dict(first_chosen)

            st.rerun()


# =========================================================
# 3. PATVIRTINTAS PREPARATAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '✅ 3. Patvirtintas pirmas rezultatas'
    '</div>',
    unsafe_allow_html=True
)

first = st.session_state.get(
    "first_vvkt_drug"
)

if first:

    st.success(
        f"💊 Patvirtintas pirmas preparatas: {first}"
    )

    chosen = st.session_state.get("first_vvkt_selected")
    if chosen and chosen.get("name") == first:
        show_selected_vvkt_info(chosen)
    else:
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
    '💊 4. Įkelkite antro vaisto pakuotės nuotrauką'
    '</div>',
    unsafe_allow_html=True
)

second_uploaded = st.file_uploader(
    "Antro vaisto pakuotės nuotrauka",
    type=["jpg", "jpeg", "png"],
    key="second_drug_image"
)

if second_uploaded:

    second_image = ImageOps.exif_transpose(
        Image.open(second_uploaded)
    ).convert("RGB")

    st.image(
        second_image,
        caption="Įkelta antro vaisto pakuotė",
        width=430
    )

    second_bytes = second_uploaded.getvalue()

    second_image_key = hashlib.sha256(
        second_bytes
    ).hexdigest()

    second_cached_key = st.session_state.get(
        "second_ocr_image_key"
    )

    # OCR vykdome tik naujai antro vaisto nuotraukai.
    if second_cached_key != second_image_key:
        st.session_state.pop("second_ocr_candidate", None)

        with st.spinner(
            "Analizuojama antro vaisto pakuotė..."
        ):

            # 1 etapas – greitas Tesseract
            second_fast_text = run_fast_ocr(
                second_image
            )

            second_fast_candidates = rank_vvkt_candidates(
                second_fast_text,
                top_n=5
            )

            second_fast_is_strong = False

            if second_fast_candidates:

                second_top_fast = second_fast_candidates[0]

                second_fast_is_strong = (
                    second_top_fast["name_score"] >= 0.90
                    and second_top_fast["score"] >= 0.75
                )

            if second_fast_is_strong:

                second_ocr_text = second_fast_text

                second_ranked_candidates = (
                    second_fast_candidates
                )

                second_ocr_engine = "Tesseract"

            else:

                # 2 etapas – PaddleOCR
                second_paddle_text = run_paddle_ocr(
                    second_image
                )

                second_combined_parts = []

                if second_fast_text:
                    second_combined_parts.append(
                        second_fast_text
                    )

                if second_paddle_text:
                    second_combined_parts.append(
                        second_paddle_text
                    )

                second_ocr_text = "\n".join(
                    dict.fromkeys(
                        second_combined_parts
                    )
                )

                second_ranked_candidates = (
                    rank_vvkt_candidates(
                        second_ocr_text,
                        top_n=5
                    )
                )

                if second_paddle_text:
                    second_ocr_engine = (
                        "Tesseract + PaddleOCR"
                    )
                else:
                    second_ocr_engine = "Tesseract"

        # Išsaugome tik antro vaisto OCR rezultatą.
        st.session_state[
            "second_ocr_image_key"
        ] = second_image_key

        st.session_state[
            "second_ocr_text_cached"
        ] = second_ocr_text

        st.session_state[
            "second_ocr_candidates_cached"
        ] = second_ranked_candidates

        st.session_state[
            "second_ocr_engine_cached"
        ] = second_ocr_engine

        # Nauja antro vaisto nuotrauka panaikina ankstesnį
        # patvirtintą antrą preparatą.
        st.session_state[
            "second_vvkt_drug"
        ] = None
        st.session_state["second_vvkt_selected"] = None

    else:

        # Ta pati nuotrauka – OCR nekartojame.
        second_ocr_text = st.session_state.get(
            "second_ocr_text_cached",
            ""
        )

        second_ranked_candidates = st.session_state.get(
            "second_ocr_candidates_cached",
            []
        )

        second_ocr_engine = st.session_state.get(
            "second_ocr_engine_cached",
            "Tesseract"
        )

    if st.session_state.get("second_vvkt_drug") is None:
        if st.button("🔁 Pakartoti išsamesnį antros pakuotės OCR", key="retry_second_ocr"):
            with st.spinner("Papildomai skaitoma antra pakuotė..."):
                better_text, better_candidates, improved = retry_ocr_from_package(
                    second_image, second_ocr_text, second_ranked_candidates
                )
            if improved:
                second_ocr_text, second_ranked_candidates = better_text, better_candidates
                second_ocr_engine = "Pakartotinis Tesseract"
                st.session_state["second_ocr_text_cached"] = second_ocr_text
                st.session_state["second_ocr_candidates_cached"] = second_ranked_candidates
                st.session_state["second_ocr_engine_cached"] = second_ocr_engine
                st.session_state.pop("second_ocr_candidate", None)
                st.success("Papildomas OCR rado aiškesnį VVKT pavadinimą. Patikrinkite kandidatą.")
            else:
                st.info("Papildomas OCR rezultato nepagerino; ankstesnis atpažinimas išsaugotas. Jei reikia, naudokite rankinę VVKT paiešką.")

    second_strengths = extract_strengths(
        second_ocr_text
    )

    second_detected_form = detect_form(
        second_ocr_text
    )

    second_detected_ingredient = detect_ingredient(
        second_ocr_text
    )

    with st.expander(
        "🔍 Antro vaisto OCR perskaitytas tekstas"
    ):

        st.caption(
            f"Naudotas OCR: {second_ocr_engine}"
        )

        st.text(
            second_ocr_text
            if second_ocr_text
            else "OCR teksto neaptiko."
        )

    st.markdown(
        "#### 🧾 OCR iš antros pakuotės perskaityta informacija"
    )

    st.write(
        "**Pavadinimas (perskaitytas OCR):**",
        ocr_read_brand_name(second_ocr_text, vvkt_names)
        or "OCR pavadinimo patikimai nenustatė"
    )

    st.write(
        "**Veiklioji medžiaga:**",
        second_detected_ingredient
        or "OCR nenustatė; galimo VVKT kandidato sudėtis pateikiama žemiau"
    )

    st.write(
        "**Stiprumas:**",
        ", ".join(ocr_strengths_unique(second_strengths)[:3])
        if second_strengths
        else "OCR nenustatė; galimo VVKT kandidato stiprumas pateikiamas žemiau"
    )

    st.write(
        "**Farmacinė forma:**",
        second_detected_form
        or "OCR nenustatė; galimo VVKT kandidato duomenys pateikiami žemiau"
    )

    st.markdown(
        "#### 🔎 Galimi VVKT preparatai (duomenys nepatvirtinti, kol nepasirinkote preparato)"
    )

    if second_ranked_candidates:

        second_labels = {}

        for index, candidate in enumerate(
            second_ranked_candidates,
            start=1
        ):

            label = (
                f"{index}. "
                f"{candidate['name']} | "
                f"{candidate['strength']} | {candidate['form']} | "
                f"{candidate['score'] * 100:.0f}% algoritmo balas"
            )

            second_labels[label] = candidate

        second_selected_label = st.selectbox(
            "Pasirinkite antrą preparatą",
            list(second_labels.keys()),
            key="second_ocr_candidate"
        )

        second_candidate = second_labels[
            second_selected_label
        ]

        if (ocr_read_brand_name(second_ocr_text, vvkt_names) == second_candidate.get("name")
                and float(second_candidate.get("score", 0)) >= 0.75):
            st.success(f"💊 VVKT kandidatas (patikrinkite): **{second_candidate['name']}**")
        else:
            st.warning(f"VVKT paieškos pasiūlymas, tapatybė nepatvirtinta: **{second_candidate['name']}**")

        second_c1, second_c2, second_c3 = st.columns(3)

        second_c1.metric(
            "Bendras atitikimas",
            f"{second_candidate['score'] * 100:.0f}%"
        )

        second_c2.metric(
            "Pavadinimas",
            f"{second_candidate['name_score'] * 100:.0f}%"
        )

        second_c3.metric(
            "Veiklioji medžiaga",
            f"{second_candidate['ingredient_score'] * 100:.0f}%"
        )

        st.write(
            "**Veiklioji medžiaga (VVKT kandidatas, dar nepatvirtinta):**",
            second_candidate["ingredient"]
        )

        st.write(
            "**Stiprumas:**",
            second_candidate["strength"]
        )

        st.write(
            "**Farmacinė forma:**",
            second_candidate["form"]
        )

        st.caption(
            "Atitikimo procentas yra paieškos algoritmo "
            "balas, o ne statistinė AI tikimybė."
        )

        second_confirmed_by_user = show_candidate_verification(
            second_candidate, second_strengths, second_detected_form, "second"
        )
        if st.button(
            "✅ Patvirtinti antrą preparatą",
            type="primary",
            disabled=not second_confirmed_by_user
        ):

            st.session_state[
                "second_vvkt_drug"
            ] = second_candidate["name"]
            st.session_state["second_vvkt_selected"] = dict(second_candidate)

            st.success(
                f"✅ Patvirtinta: "
                f"{second_candidate['name']}"
            )

    else:

        st.warning(
            "Patikimo antro preparato VVKT kandidato "
            "automatiškai parinkti nepavyko."
        )


# =========================================================
# RANKINĖ ANTRO PREPARATO PAIEŠKA
# =========================================================

st.markdown(
    "#### 🔎 Arba ieškokite antro preparato rankiniu būdu"
)

second_query = st.text_input(
    "Ieškoti antro preparato",
    placeholder="Pvz. Dolmen, NO-SPA...",
    key="manual_second_query"
)

if second_query:

    second_matches = find_names(
        second_query
    )

    if second_matches:

        manual_second = st.selectbox(
            "Rasti antro preparato variantai",
            second_matches,
            key="manual_second"
        )

        second_options = manual_vvkt_options(manual_second)
        second_variant = st.selectbox(
            "Pasirinkite tikslų stiprumą ir farmacinę formą",
            list(second_options),
            key="manual_second_variant",
        )
        second_chosen = second_options[second_variant]
        show_selected_vvkt_info(second_chosen)

        if st.button(
            "✅ Naudoti šį antrą preparatą"
        ):

            st.session_state[
                "second_vvkt_drug"
            ] = manual_second
            st.session_state["second_vvkt_selected"] = dict(second_chosen)

            st.success(
                f"✅ Patvirtinta: {manual_second}"
            )


# =========================================================
# 5. PATVIRTINTAS ANTRAS REZULTATAS
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '✅ 5. Patvirtintas antras rezultatas'
    '</div>',
    unsafe_allow_html=True
)

# Patvirtintas antras preparatas.
second = st.session_state.get(
    "second_vvkt_drug"
)

if second:

    st.success(
        f"💊 Patvirtintas antras preparatas: {second}"
    )

    chosen = st.session_state.get("second_vvkt_selected")
    if chosen and chosen.get("name") == second:
        show_selected_vvkt_info(chosen)
    else:
        show_vvkt_info(second)
else:
    st.info("Dar nepatvirtintas antrasis preparatas.")


# Demonstracinis perspėjimas automatiškai perskaičiuojamas keičiant pasirinkimą.
# Vienareikšmiškumo reikalavimas: skirtingų VVKT eilučių reikšmės nespėjamos.
def _demo_info(name, quantity, chosen=None):
    if chosen and chosen.get("name") == name:
        return {"name": name, "ingredient": chosen.get("ingredient"), "strength": chosen.get("strength"), "form": chosen.get("form"), "quantity": quantity}
    rows = get_vvkt_rows(name)
    if rows.empty:
        return {"name": name, "ingredient": None, "strength": None, "form": None, "quantity": quantity}
    result = {"name": name, "quantity": quantity}
    for target, column in (("ingredient", "veiklioji_medz_lt"), ("strength", "stiprumas"), ("form", "farmacine_forma_lt")):
        values = {str(v).strip() for v in rows[column].dropna() if str(v).strip()}
        result[target] = next(iter(values)) if len(values) == 1 else None
    return result

if first and second:
    st.markdown("### 🛡️ 6. Vaistų derinio patikra")
    st.caption("Patikra skirta universiteto demonstracijai, ne medicininiam suderinamumui nustatyti.")
    q1, q2 = st.columns(2)
    with q1:
        first_qty = st.number_input(f"{first}: vienetų kiekis", min_value=1, max_value=100, value=1, step=1, key="demo_qty_first")
    with q2:
        second_qty = st.number_input(f"{second}: vienetų kiekis", min_value=1, max_value=100, value=1, step=1, key="demo_qty_second")
    demo_items = [_demo_info(first, int(first_qty), st.session_state.get("first_vvkt_selected")), _demo_info(second, int(second_qty), st.session_state.get("second_vvkt_selected"))]
    demo_warnings, demo_notes, demo_components = analyze_demo_risk(demo_items)
    st.markdown("**Atpažintos veikliosios medžiagos**")
    for comp in demo_components:
        mg_text = f"{comp['mg']:g} mg" if comp['mg'] is not None else "kiekis neapskaičiuotas"
        st.write(f"• {comp['name']}: {comp['ingredient']} – {mg_text}")
    st.markdown("**Techninė dubliavimo patikra (ne sąveikos vertinimas)**")
    for warning in demo_warnings:
        st.warning(warning)
    for note in demo_notes:
        st.info(note)
    if not demo_warnings and not demo_notes:
        st.caption("Automatinė dubliavimo taisyklė šiai porai nesuveikė. Tai nieko nepasako apie kitokias vaistų sąveikas.")
    st.caption("Tikrinamas galimas veikliųjų medžiagų dubliavimas. Skirtingų medžiagų mg sumos nėra medicininis rizikos kriterijus. Perspėjimo nebuvimas nereiškia, kad vaistus saugu vartoti kartu.")
    current_signature = (first, second, int(first_qty), int(second_qty), tuple(demo_warnings))
    if st.session_state.get("demo_ack_signature") != current_signature and demo_warnings:
        @st.dialog("⚠️ Dėmesio! Galima vaistų vartojimo rizika!")
        def _demo_popup():
            st.warning("Pasirinktas vaistų derinys atitinka vieną ar kelis demonstracinius perspėjimo kriterijus. Kartu vartojant šiuos vaistus gali kilti nepageidaujamų reakcijų arba dozės dubliavimo rizika. Prieš vartodami šiuos vaistus kartu, pasitarkite su gydytoju arba vaistininku.")
            st.caption("Tai nėra medicininė sąveikų patikra ar saugumo patvirtinimas.")
            if st.button("Gerai", key="demo_ack_button"):
                st.session_state["demo_ack_signature"] = current_signature
                st.rerun()
        _demo_popup()


# =========================================================
# 7. FARMAKOLOGINĖ ANALIZĖ
# =========================================================

st.divider()

st.markdown(
    '<div class="step">'
    '🔬 7. Vaistų derinio analizė'
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

        # Pasirinkto kandidato duomenys turi pirmenybę prieš paiešką vien pagal pavadinimą.
        first_selected = st.session_state.get("first_vvkt_selected")
        second_selected = st.session_state.get("second_vvkt_selected")
        if not (first_selected and first_selected.get("name") == first):
            first_selected = None
        if not (second_selected and second_selected.get("name") == second):
            second_selected = None
        first_row = None if first_selected else get_vvkt_row(first)
        second_row = None if second_selected else get_vvkt_row(second)

        ingredient_1 = (first_selected.get("ingredient") if first_selected
                        else get_vvkt_ingredient(first))
        ingredient_2 = (second_selected.get("ingredient") if second_selected
                        else get_vvkt_ingredient(second))
        if not first_selected or not second_selected:
            st.info("Rankiniu būdu pasirinktam vaistui VVKT gali turėti kelis stiprumus. "
                    "Jeigu variantas nepatvirtintas tiksliai, būtina papildoma patikra.")

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

            if first_selected:
                st.write("**Stiprumas:**", first_selected.get("strength") or "nenurodyta")
                st.write("**Farmacinė forma:**", first_selected.get("form") or "nenurodyta")
                st.caption("Duomenys iš konkretaus patvirtinto VVKT kandidato.")
            elif first_row is not None:
                st.write("**Stiprumas:**", clean_value(first_row.get("stiprumas")))
                st.write("**Farmacinė forma:**", clean_value(first_row.get("farmacine_forma_lt")))

        with col2:

            st.markdown(
                f"#### {second}"
            )

            st.write(
                "**Veiklioji medžiaga:**",
                ingredient_2 or "—"
            )

            if second_selected:
                st.write("**Stiprumas:**", second_selected.get("strength") or "nenurodyta")
                st.write("**Farmacinė forma:**", second_selected.get("form") or "nenurodyta")
                st.caption("Duomenys iš konkretaus patvirtinto VVKT kandidato.")
            elif second_row is not None:
                st.write("**Stiprumas:**", clean_value(second_row.get("stiprumas")))
                st.write("**Farmacinė forma:**", clean_value(second_row.get("farmacine_forma_lt")))

        # =================================================
        # SĄVEIKOS REZULTATAS
        # =================================================
        # Sąveikos modulio rezultatas yra vienintelis šio skyriaus
        # aiškinimas. Atskirų nepatikrintų PCS tekstų ir tuščių
        # farmakologinių profilių čia nebedubliuojame.
        show_pair_result(ingredient_1, ingredient_2)

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
