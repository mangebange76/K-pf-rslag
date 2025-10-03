# app.py
from __future__ import annotations
import pandas as pd
import streamlit as st

st.set_page_config(page_title="K-pf-rslag", layout="wide")
st.title("K-pf-rslag")

# Importera din vy (om du har den)
try:
    from stockapp.manual_collect import manual_collect_view
except Exception:
    manual_collect_view = None  # type: ignore

# Sheets-helpers (URL-baserad, enligt din gamla app)
from stockapp.sheets import get_ws, ws_read_df, save_dataframe, list_sheet_names

def _load_df(worksheet_name: str | None) -> pd.DataFrame:
    try:
        ws = get_ws(worksheet_name=worksheet_name)
        return ws_read_df(ws)
    except Exception as e:
        st.warning(f"🚫 Kunde inte läsa från Google Sheet: {e}")
        return pd.DataFrame()

def _save_df(df: pd.DataFrame, worksheet_name: str | None) -> None:
    try:
        save_dataframe(df, worksheet_name=worksheet_name)
        st.success("Sparat till Google Sheets.")
    except Exception as e:
        st.warning(f"⚠️ Kunde inte spara: {e}")

# Sidopanel: välj blad
with st.sidebar:
    st.header("Google Sheets")
    blad = []
    try:
        blad = list_sheet_names()
    except Exception as e:
        st.info(f"Kunde inte lista blad: {e}")
    default_name = st.secrets.get("WORKSHEET_NAME") or "Blad1"
    if blad and default_name in blad:
        idx = blad.index(default_name)
    else:
        idx = 0 if blad else 0
    ws_name = st.selectbox("Välj blad:", blad or [default_name], index=idx)
    if st.button("🔄 Läs in"):
        st.session_state["_df_ref"] = _load_df(ws_name)
        st.toast(f"Inläst '{ws_name}'", icon="✅")

    uploaded = st.file_uploader("Importera CSV (ersätter vy)", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state["_df_ref"] = pd.read_csv(uploaded)
            st.success("CSV inläst.")
        except Exception as e:
            st.error(f"Kunde inte läsa CSV: {e}")

    if st.button("💾 Spara vy"):
        _save_df(st.session_state.get("_df_ref", pd.DataFrame()), ws_name)

# Init-läsning
if "_df_ref" not in st.session_state:
    st.session_state["_df_ref"] = _load_df(st.secrets.get("WORKSHEET_NAME") or "Blad1")

# Flikar
tab_data, tab_collect = st.tabs(["📄 Data", "🧩 Manuell insamling"])

with tab_data:
    df = st.session_state.get("_df_ref", pd.DataFrame())
    if df.empty:
        st.info("Ingen data att visa.")
    else:
        st.dataframe(df, use_container_width=True)

with tab_collect:
    if manual_collect_view is None:
        st.info("Insamlingsvyn saknas i detta exempel.")
    else:
        df_in = st.session_state.get("_df_ref", pd.DataFrame())
        df_out = manual_collect_view(df_in)
        if isinstance(df_out, pd.DataFrame) and not df_out.equals(df_in):
            st.session_state["_df_ref"] = df_out
            st.success("Uppdaterade sessionens data.")
