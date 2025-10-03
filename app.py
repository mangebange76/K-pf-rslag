# app.py
from __future__ import annotations
import pandas as pd
import streamlit as st

st.set_page_config(page_title="K-pf-rslag", layout="wide")
st.title("K-pf-rslag")

# ── Importer ────────────────────────────────────────────────────────────────
try:
    from stockapp.manual_collect import manual_collect_view
except Exception as e:
    st.error(f"Kunde inte importera manual_collect: {e}")
    manual_collect_view = None  # type: ignore

_sheets_ok = True
try:
    from stockapp.sheets import get_ws, ws_read_df, save_dataframe
except Exception as e:
    _sheets_ok = False
    st.error(f"Kunde inte ladda sheets-modulen: {e}")
    def get_ws(*_, **__): raise RuntimeError(f"Sheets-modulen saknas: {e}")
    def ws_read_df(*_, **__): raise RuntimeError(f"Sheets-modulen saknas: {e}")
    def save_dataframe(*_, **__): raise RuntimeError(f"Sheets-modulen saknas: {e}")

try:
    from stockapp.storage import hamta_data  # valfri
except Exception:
    hamta_data = None  # type: ignore

# ── Hjälpare ───────────────────────────────────────────────────────────────
def _load_df_from_sheets() -> pd.DataFrame:
    try:
        ws = get_ws()
        return ws_read_df(ws)
    except Exception as e:
        st.warning(f"🚫 Kunde inte läsa data från Google Sheet: {e}")
        return pd.DataFrame()

def _ensure_df_in_state() -> None:
    if "_df_ref" in st.session_state:
        return
    df = pd.DataFrame()
    # Försök via storage.hamta_data (utan truthiness-bugg)
    if callable(hamta_data):
        try:
            tmp = hamta_data()
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                df = tmp
        except Exception as e:
            st.info(f"Info: hamta_data() misslyckades: {e}")
    # Annars, Sheets
    if df.empty and _sheets_ok:
        df = _load_df_from_sheets()
    st.session_state["_df_ref"] = df

def _save_df_via_sheets(df: pd.DataFrame) -> None:
    if not _sheets_ok:
        st.info("Ingen sheets-modul – hoppar över skrivning.")
        return
    try:
        save_dataframe(df)
        st.success("Sparat till Google Sheets.")
    except Exception as e:
        st.warning(f"⚠️ Kunde inte spara via sheets-modulen: {e}")

# ── Init-data ──────────────────────────────────────────────────────────────
_ensure_df_in_state()

# ── Sidopanel ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Åtgärder")
    if st.button("🔄 Läs in från Sheets igen"):
        st.session_state["_df_ref"] = _load_df_from_sheets()
        st.toast("Inläst från Sheets.", icon="✅")

    uploaded = st.file_uploader("Importera CSV (ersätter nuvarande vy)", type=["csv"])
    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            st.session_state["_df_ref"] = df_new
            st.success("CSV inläst till vy.")
        except Exception as e:
            st.error(f"Kunde inte läsa CSV: {e}")

    if st.button("💾 Spara nuvarande vy till Sheets"):
        _save_df_via_sheets(st.session_state.get("_df_ref", pd.DataFrame()))

# ── Flikar ─────────────────────────────────────────────────────────────────
tab_data, tab_collect = st.tabs(["📄 Data", "🧩 Manuell insamling"])

with tab_data:
    df = st.session_state.get("_df_ref", pd.DataFrame())
    if df is None or df.empty:
        st.info("Ingen data att visa.")
    else:
        st.dataframe(df, use_container_width=True)

with tab_collect:
    if manual_collect_view is None:
        st.error("manual_collect_view saknas – kan inte visa insamlingsvyn.")
    else:
        df_in = st.session_state.get("_df_ref", pd.DataFrame())
        df_out = manual_collect_view(df_in)
        if isinstance(df_out, pd.DataFrame) and not df_out.equals(df_in):
            st.session_state["_df_ref"] = df_out
            st.success("Vyn returnerade uppdaterat DataFrame – uppdaterade sessionens data.")

st.caption("Build OK • Enkel Sheets-koppling (GOOGLE_CREDENTIALS + SPREADSHEET_ID).")
