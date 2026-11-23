"""Edukaciniai konkrečių veikliųjų medžiagų porų aprašai (ne medicininė rekomendacija)."""
import csv
from pathlib import Path
from demo_saveikos import normalize

DATA_FILE = Path(__file__).with_name("saltiniais_pagristos_10_poru.csv")


def find_documented_pairs(first, second, data_path=None):
    """Exact ingredient-pair lookup, order independent; never infer an interaction."""
    a, b = normalize(first), normalize(second)
    missing = {"", "—", "nenurodyta", "nan", "none"}
    if a in missing or b in missing or a == b:
        return []

    path = Path(data_path) if data_path else DATA_FILE
    with path.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))

    return [r for r in records if
            {a, b} == {normalize(r["veiklioji_1"]), normalize(r["veiklioji_2"])}]


def show_pair_result(first, second):
    """Tik UI; OCR ir VVKT kodą paliekame nepakitusį."""
    import streamlit as st
    records = find_documented_pairs(first, second)
    st.markdown("### 🧩 Dviejų vaistų sąveikos rezultatas")
    if not records:
        st.info("Šiai konkrečiai veikliųjų medžiagų porai mūsų mokomojoje bazėje įrašo nėra. Tai nereiškia, kad sąveikos nėra.")
        st.caption("Mokomoji bazė šiuo metu apima 10 šaltiniais pagrįstų porų, o ne visas registruotų vaistų sąveikas.")
        return False
    for r in records:
        st.markdown("**Galima sąveika:** " + r["saveika"])
        st.markdown("**Reikšmingumas pagal nurodytą PCS / šaltinį:** " + r["kategorija"])
        st.write("**Kodėl:**", r["mechanizmas"])
        st.write("**Galimos pasekmės:**", r["galimos_pasekmes"])
        st.write("**Svarbios sąlygos:**", r["salygos"])
        st.markdown("**Šaltinis:** [Oficiali PCS arba NHS informacija](" + r["saltinis_url"] + ")")
    st.caption("Edukacinis viešai prieinamų šaltinių apibendrinimas. UK PCS nebūtinai atitinka konkretaus Lietuvoje registruoto preparato PCS. Dėl vartojimo sprendimų kreipkitės į gydytoją ar vaistininką.")
    return True
