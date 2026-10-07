import io
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Svarbu: importuojame TĄ PAČIĄ logiką, kuri naudojama pagrindinėje programoje.
import app as core


st.title("Proposed metodo eksperimentas")

st.write(
    "Šis puslapis automatiškai testuoja Tesseract → PaddleOCR → VVKT "
    "atpažinimo grandinę su ZIP faile esančiomis vaistų nuotraukomis."
)

st.warning(
    "Eksperimentui naudok ZIP failą, kuriame nuotraukų failų pavadinimai "
    "prasideda tikruoju vaisto pavadinimu."
)

uploaded_zip = st.file_uploader(
    "Įkelk testavimo nuotraukų ZIP",
    type=["zip"]
)


def expected_name_from_filename(filename):
    """
    Tikras vaisto pavadinimas gaunamas iš failo pavadinimo.
    Pašalinamas tik paskutinis numeris, pvz.:
    'IBUPROM 1.jpg' -> 'IBUPROM'
    'Ferrum Lek 2.jpg' -> 'Ferrum Lek'
    """
    stem = Path(filename).stem.strip()

    parts = stem.rsplit(" ", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].strip()

    return stem


def norm(text):
    return core.normalize_text(text)


def same_name(expected, predicted):
    return norm(expected) == norm(predicted)


if uploaded_zip is not None:

    if st.button("Paleisti Proposed eksperimentą", type="primary"):

        results = []

        zip_bytes = uploaded_zip.getvalue()

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:

            image_files = [
                name for name in zf.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower()
                in [".jpg", ".jpeg", ".png", ".webp"]
            ]

            st.write(f"Rasta nuotraukų: **{len(image_files)}**")

            progress = st.progress(0)
            status = st.empty()

            for index, filename in enumerate(image_files, start=1):

                status.write(
                    f"Apdorojama {index}/{len(image_files)}: {filename}"
                )

                expected = expected_name_from_filename(filename)

                raw = zf.read(filename)

                try:
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as exc:
                    results.append({
                        "file": filename,
                        "expected": expected,
                        "engine": "ERROR",
                        "top1": "",
                        "top1_correct": False,
                        "top3_correct": False,
                        "score": 0.0,
                        "name_score": 0.0,
                        "ingredient": "",
                        "strength": "",
                        "form": "",
                        "runtime_seconds": 0.0,
                        "error": str(exc),
                    })
                    continue

                start = time.perf_counter()

                # -----------------------------------------
                # 1. TESSERACT
                # -----------------------------------------

                try:
                    tess_text = core.run_fast_ocr(image)
                except Exception as exc:
                    tess_text = ""
                    tess_error = str(exc)
                else:
                    tess_error = ""

                tess_candidates = core.rank_vvkt_candidates(
                    tess_text,
                    top_n=5
                )

                strong_tesseract = False

                if tess_candidates:
                    best = tess_candidates[0]

                    strong_tesseract = (
                        best.get("name_score", 0) >= 0.90
                        and best.get("score", 0) >= 0.75
                    )

                # -----------------------------------------
                # 2. PADDLEOCR FALLBACK
                # -----------------------------------------

                engine = "Tesseract"
                final_text = tess_text
                final_candidates = tess_candidates
                paddle_error = ""

                if not strong_tesseract:

                    engine = "Tesseract + PaddleOCR"

                    try:
                        paddle_text = core.run_paddle_ocr(image)

                        if paddle_text:
                            final_text = (
                                str(tess_text).strip()
                                + "\n"
                                + str(paddle_text).strip()
                            ).strip()

                            final_candidates = (
                                core.rank_vvkt_candidates(
                                    final_text,
                                    top_n=5
                                )
                            )

                    except Exception as exc:
                        paddle_error = str(exc)

                runtime = time.perf_counter() - start

                # -----------------------------------------
                # 3. REZULTATAS
                # -----------------------------------------

                top1 = ""
                score = 0.0
                name_score = 0.0
                ingredient = ""
                strength = ""
                form = ""

                if final_candidates:
                    best = final_candidates[0]

                    top1 = best.get("name", "")
                    score = best.get("score", 0.0)
                    name_score = best.get("name_score", 0.0)
                    ingredient = best.get("ingredient", "")
                    strength = best.get("strength", "")
                    form = best.get("form", "")

                top_names = [
                    candidate.get("name", "")
                    for candidate in final_candidates[:3]
                ]

                top1_correct = same_name(
                    expected,
                    top1
                )

                top3_correct = any(
                    same_name(expected, name)
                    for name in top_names
                )

                results.append({
                    "file": filename,
                    "expected": expected,
                    "engine": engine,
                    "top1": top1,
                    "top1_correct": top1_correct,
                    "top3_correct": top3_correct,
                    "score": round(float(score), 4),
                    "name_score": round(float(name_score), 4),
                    "ingredient": ingredient,
                    "strength": strength,
                    "form": form,
                    "runtime_seconds": round(runtime, 3),
                    "tesseract_error": tess_error,
                    "paddle_error": paddle_error,
                })

                progress.progress(
                    index / len(image_files)
                )

        df = pd.DataFrame(results)

        st.success("Eksperimentas baigtas.")

        st.dataframe(
            df,
            use_container_width=True
        )

        if len(df) > 0:

            top1_accuracy = df["top1_correct"].mean()
            top3_accuracy = df["top3_correct"].mean()

            paddle_rate = (
                df["engine"]
                .astype(str)
                .str.contains("PaddleOCR")
                .mean()
            )

            mean_runtime = df["runtime_seconds"].mean()
            median_runtime = df["runtime_seconds"].median()

            st.subheader("Pagrindinės metrikos")

            metrics = pd.DataFrame({
                "Metrika": [
                    "Nuotraukų skaičius",
                    "TOP-1 accuracy",
                    "TOP-3 hit rate",
                    "PaddleOCR fallback rate",
                    "Vidutinis laikas, s",
                    "Mediana, s",
                ],
                "Reikšmė": [
                    len(df),
                    f"{top1_accuracy:.2%}",
                    f"{top3_accuracy:.2%}",
                    f"{paddle_rate:.2%}",
                    f"{mean_runtime:.3f}",
                    f"{median_runtime:.3f}",
                ],
            })

            st.table(metrics)

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Atsisiųsti proposed_benchmark_results.csv",
            data=csv_data,
            file_name="proposed_benchmark_results.csv",
            mime="text/csv",
        )
