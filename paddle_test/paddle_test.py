import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from paddleocr import PaddleOCR


st.set_page_config(
    page_title="PaddleOCR testas",
    page_icon="🔬"
)

st.title("🔬 PaddleOCR vaistų pakuotės testas")

st.caption(
    "Atskiras eksperimentas. "
    "Pagrindinis app.py nekeičiamas."
)


@st.cache_resource
def load_paddle_ocr():
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )


uploaded = st.file_uploader(
    "Įkelkite vaisto pakuotės nuotrauką",
    type=["jpg", "jpeg", "png"]
)


if uploaded:

    image = ImageOps.exif_transpose(
        Image.open(uploaded)
    ).convert("RGB")

    st.image(
        image,
        caption="Testuojama nuotrauka",
        width=430
    )

    if st.button(
        "🔬 Paleisti PaddleOCR",
        type="primary"
    ):

        with st.spinner(
            "PaddleOCR analizuoja nuotrauką..."
        ):

            start = time.perf_counter()

            ocr = load_paddle_ocr()

            result = ocr.predict(
                np.asarray(image)
            )

            elapsed = (
                time.perf_counter()
                - start
            )

        st.success(
            f"OCR baigtas per {elapsed:.2f} s"
        )

        st.subheader("PaddleOCR rezultatas")

        found_text = []

        for item in result:

            data = getattr(
                item,
                "json",
                None
            )

            if callable(data):
                data = data()

            if isinstance(data, dict):

                res = data.get(
                    "res",
                    data
                )

                texts = res.get(
                    "rec_texts",
                    []
                )

                for text in texts:

                    text = str(text).strip()

                    if text:
                        found_text.append(text)

        if found_text:

            st.text(
                "\n".join(found_text)
            )

        else:

            st.warning(
                "PaddleOCR teksto negrąžino."
            )

            st.write(
                "Žalias PaddleOCR rezultatas:"
            )

            st.write(result)
