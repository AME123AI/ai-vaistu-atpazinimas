"""Atskiras informacinis Streamlit puslapis; nekeičia pagrindinės app.py logikos."""
from pathlib import Path
import pandas as pd
import streamlit as st

from kolegu_preparatu_info import load_info_catalog, match_product
from preparato_duomenu_pilnumas import load_catalog, check_drug

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Kolegų preparatų informacija", page_icon="📋", layout="wide")
st.title("📋 Kolegų preparatų informacija")
st.info("Edukacinis bandymų puslapis. Kolegų duomenų medicininis tikslumas nėra nepriklausomai patvirtintas. Nevertina vaistų poros sąveikos.")

info = load_info_catalog(str(ROOT / "kolegu_preparatu_informacija.csv"))
texts = load_catalog(ROOT / "saveikos_rezultatai_441.csv")
if not info:
    st.error("Preparatų informacijos katalogas nerastas arba jo struktūra netinkama.")
    st.stop()

# Kiekviena parinktis saugo tikslius 4 požymius. Nesiejame tik pagal pavadinimą.
options = sorted(
    {(r.get('pavadinimas',''),r.get('veiklioji',''),r.get('stiprumas',''),r.get('forma',''))
     for r in info if all((r.get(x) or '').strip() for x in ('pavadinimas','veiklioji','stiprumas','forma'))},
    key=lambda x: (x[0].casefold(),x[2].casefold(),x[3].casefold()),
)
if not options:
    st.warning("Kataloge nėra pakankamų preparatų identifikavimo duomenų.")
    st.stop()

chosen = st.selectbox(
    "Pasirink preparatą (pavadinimas, stiprumas, forma, veiklioji medžiaga)",
    options,
    format_func=lambda x: f"{x[0]} | {x[2]} | {x[3]} | {x[1]}",
)
name, ingredient, strength, form = chosen
row = match_product(info,name,ingredient,strength,form)
if not row:
    st.warning("Vienareikšmiškai susieti įrašo nepavyko. Informacija nerodoma.")
    st.stop()

st.subheader(name)
st.write("**Veiklioji medžiaga:**", ingredient)
st.write("**Stiprumas:**", strength)
st.write("**Forma:**", form)
st.write("**Informacijos kilmės statusas:**", row.get('duomenu_busena') or 'Nenurodytas')

# Net 'NURODYTAS_KONKRETUS_DOKUMENTAS' reiškia tik nuorodą, ne klinikinę validaciją.
status = check_drug(name, ingredient, strength, form, texts)
if not status['enough']:
    st.warning("⚠️ Mažai duomenų. Reikia pasitarti su gydytoju ar vaistininku.")
    st.caption(status['reason'])
else:
    st.info("Rastas bandomasis 4.5 skyriaus tekstas, tačiau jo tikslumas ir atitikimas preparatui dar nepatvirtinti.")

st.markdown("### Papildoma kolegų surinkta informacija")
if row.get('duomenu_busena') != 'NURODYTAS_KONKRETUS_DOKUMENTAS':
    st.warning("Toliau pateikiamas bendrinis arba neišsamus aprašymas. Tai nėra individualiai patvirtinta šio preparato PCS informacija.")

for field,label in (
    ('vartojimas','Vartojimo būdas'),
    ('indikacijos','Indikacijos'),
    ('kontraindikacijos','Kontraindikacijos'),
    ('ispejimai','Įspėjimai'),
    ('mechanizmas','Veikimo mechanizmas'),
):
    val = (row.get(field) or '').strip()
    if val and val.casefold() not in {'nan','none','nerasta','—','-'}:
        st.markdown(f"**{label}**")
        st.write(val)

st.markdown("### Nuorodos patikrai")
for field,label in [('vvkt_url','VVKT preparato kortelė'),('dokumento_url','Nurodytas dokumentas')]:
    url=(row.get(field) or '').strip()
    if url.startswith(('https://','http://')):
        st.markdown(f"[{label}]({url})")
st.caption("Nurodytos nuorodos ir ištraukos automatiškai nepatvirtina nei konkrečios poros sąveikos, nei informacijos tinkamumo konkrečiam pacientui.")
