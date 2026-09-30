
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import hog
from difflib import get_close_matches
ROOT = Path(__file__).parent
st.set_page_config(page_title="AI vaistų atpažinimas", page_icon="💊", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "hog_logreg.joblib")

@st.cache_data
def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))

@st.cache_data
def load_interactions():
    return pd.read_csv(ROOT / "interactions.csv")
@st.cache_data
def load_vvkt():
    return pd.read_csv(ROOT / "PreparatasPakuote.csv")
model = load_model()
classes = list(model.classes_)
drug_data = load_json("drug_profiles.json")
ingredient_profiles = load_json("ingredient_profiles.json")
interactions = load_interactions()
vvkt = load_vvkt()
vvkt = vvkt[vvkt["preparato_pav"].notna()].copy()
vvkt["preparato_pav"] = vvkt["preparato_pav"].astype(str).str.strip()
vvkt_names = sorted(vvkt["preparato_pav"].drop_duplicates().tolist())
st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
h1 {text-align:center;}
.step {font-size:1.1rem; font-weight:700;}
.small {opacity:.75;}
</style>
""", unsafe_allow_html=True)

st.title("💊 AI vaistų atpažinimas ir sąveikų paaiškinimas")
st.caption("Edukacinis HOG + Logistic Regression prototipas")

st.warning("⚠️ Edukacinis prototipas. Informacija nėra individuali medicininė rekomendacija ir nepakeičia gydytojo ar vaistininko konsultacijos.")

st.markdown('<div class="step">📷 1. Įkelkite vaisto pakuotės nuotrauką</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Vaisto pakuotės nuotrauka", type=["jpg","jpeg","png"])

if "first_drug" not in st.session_state:
    st.session_state.first_drug = None

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    c1,c2 = st.columns(2)
    with c1:
        st.image(image, caption="Įkelta pakuotė", use_container_width=True)
    with c2:
        arr = np.asarray(image.convert("L").resize((128,128)),dtype=np.float32)/255.0
        features = hog(arr, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
                       block_norm="L2-Hys", transform_sqrt=True, feature_vector=True).reshape(1,-1)
        probs = model.predict_proba(features)[0]
        top = np.argsort(probs)[::-1][:3]
        st.markdown("### 🤖 2. AI modelis atpažins vaistą")
        for i,idx in enumerate(top,1):
            st.write(f"**{i}. {model.classes_[idx]}** — {probs[idx]*100:.1f}% modelio tikimybės įvertis")
        suggested = model.classes_[top[0]]
        if st.button("Patvirtinti Top-1"):
            st.session_state.first_drug = suggested

st.divider()
st.markdown("### 🔎 Patobulinta paieška VVKT vaistų kataloge")

vvkt_query = st.text_input(
    "Įveskite vaisto pavadinimą",
    placeholder="Pvz. Atacand"
)

if vvkt_query:
    matches = [
        name for name in vvkt_names
        if vvkt_query.lower() in name.lower()
    ][:20]

    if matches:
        st.selectbox("Rasti preparatai", matches)
    else:
        st.info("Pagal įvestą pavadinimą preparatų nerasta.")
st.divider()
st.markdown('<div class="step">✅ 3. Patvirtinkite arba pataisykite rezultatą</div>', unsafe_allow_html=True)

default_index = classes.index(st.session_state.first_drug) if st.session_state.first_drug in classes else 0
first = st.selectbox("Pirmasis vaistas", classes, index=default_index)
st.session_state.first_drug = first

st.markdown('<div class="step">💊 4. Pasirinkite antrą vaistą</div>', unsafe_allow_html=True)
second = st.selectbox("Antrasis vaistas", classes, index=1 if len(classes)>1 else 0)

st.divider()
st.markdown('<div class="step">🔬 5. Patikrinkite sąveikos informaciją</div>', unsafe_allow_html=True)

def route_text(routes):
    return ", ".join(routes) if routes else "vartojimo būdas neįvestas"

def show_drug(name):
    d = drug_data.get(name, {})
    st.markdown(f"**{name}**")
    st.write("Vartojimo būdas:", route_text(d.get("routes",[])))
    ings = d.get("ingredients",[])
    if not ings:
        st.info("Šio preparato veikliųjų medžiagų profilį dar reikia patikrinti pagal konkretaus preparato informacinį lapelį.")
        return
    for ing in ings:
        p = ingredient_profiles.get(ing)
        if p:
            st.write(f"• **{ing}** — {p['group']}. {p['effect']}")
        else:
            st.write(f"• **{ing}**")

if st.button("🔬 Patikrinti sąveiką", type="primary"):
    if first == second:
        st.error("Pasirinkite du skirtingus preparatus.")
    else:
        st.subheader(f"💊 {first} + {second}")
        st.markdown("### 🧪 Veikliosios medžiagos")
        a = drug_data.get(first,{})
        b = drug_data.get(second,{})
        show_drug(first)
        show_drug(second)

        st.markdown("### 🔬 Kaip šie vaistai veikia kartu")
        mask = ((interactions.drug_a==first)&(interactions.drug_b==second)) | ((interactions.drug_a==second)&(interactions.drug_b==first))
        found = interactions[mask]
        if not found.empty:
            for _,row in found.iterrows():
                st.warning(row["description"])
                st.caption("Šaltinis: " + row["source"])
        else:
            ings1=a.get("ingredients",[]); ings2=b.get("ingredients",[])
            known1=[ingredient_profiles[x]["effect"] for x in ings1 if x in ingredient_profiles]
            known2=[ingredient_profiles[x]["effect"] for x in ings2 if x in ingredient_profiles]
            st.info("Šiai porai nėra įrašytos konkrečios porinės taisyklės mūsų prototipo bazėje. Toliau pateikiamas veikimo mechanizmų palyginimas; tai nėra teiginys, kad derinys yra saugus.")
            if known1: st.write(f"**{first}:** " + " ".join(known1))
            if known2: st.write(f"**{second}:** " + " ".join(known2))
            if not known1 or not known2:
                st.warning("Bent vieno preparato sudėtis šiame prototipe dar nėra pakankamai suprofiliuota, todėl automatinės sąveikos išvados neteikiamos.")

        st.markdown("### 💉 Vartojimo būdas")
        st.write(f"{first}: {route_text(a.get('routes',[]))}")
        st.write(f"{second}: {route_text(b.get('routes',[]))}")

st.divider()
st.caption("Modelio klasės: 24. HOG apdorojimas: 128×128, 9 orientacijos, 8×8 pikselių ląstelė, 2×2 blokas.")
