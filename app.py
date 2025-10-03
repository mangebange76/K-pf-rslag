# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st

# ── App-setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="K-pf-rslag", layout="wide")

# ── Importer (robusta) ─────────────────────────────────────────────────────
# manual_collect är central för insamlingsflödet
try:
    from stockapp.manual_collect import manual_collect_view
except Exception as e:
    st.error(f"Kunde inte importera manual_collect: {e}")
    manual_collect_view = None  # type: ignore

# Sheets-stöd (inkl. runtime-override + diagnos)
_sheets_ok = True
try:
    from stockapp.sheets import (
        get_ws,
        ws_read_df,
        ws_write_df,
        save_dataframe,
        set_runtime_service_account,
        secrets_diagnose,
    )
except Exception as e:
    _sheets_ok = False
    # Fallback-dummies så appen kan laddas även om sheets-modulen inte är på plats
    def get_ws(*_, **__):
        raise RuntimeError(f"Sheets-modulen saknas eller kunde inte importeras: {e}")

    def ws_read_df(*_, **__):
        raise RuntimeError(f"Sheets-modulen saknas eller kunde inte importeras: {e}")

    def ws_write_df(*_, **__):
        raise RuntimeError(f"Sheets-modulen saknas eller kunde inte importeras: {e}")

    def save_dataframe(*_, **__):
        raise RuntimeError(f"Sheets-modulen saknas eller kunde inte importeras: {e}")

    def set_runtime_service_account(*_, **__):
        raise RuntimeError("set_runtime_service_account saknas i sheets.py (uppdatera filen enligt senaste instruktion).")

    def secrets_diagnose():
        return {"info": "secrets_diagnose saknas i sheets.py (uppdatera filen enligt senaste instruktion)."}

# storage (om du har egen läslogik där)
try:
    from stockapp.storage import hamta_data  # valfri; används om den finns
except Exception:
    hamta_data = None  # type: ignore


# ── Hjälpare ───────────────────────────────────────────────────────────────
def _load_df_from_sheets() -> pd.DataFrame:
    """Försök läsa hela kalkylbladet via sheets-modulen."""
    try:
        ws = get_ws()  # använder secrets för Spreadsheet-ID/blad
        df = ws_read_df(ws)
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        return df
    except Exception as e:
        st.warning(f"🚫 Kunde inte läsa data från Google Sheet: {e}")
        return pd.DataFrame()


def _ensure_df_in_state() -> None:
    """Lägg in start-DataFrame i sessionen om det saknas."""
    if "_df_ref" in st.session_state:
        return

    df = pd.DataFrame()
    # 1) Prova storage.hamta_data() om den finns
    if callable(hamta_data):
        try:
            df = hamta_data() or pd.DataFrame()
        except Exception as e:
            st.info(f"Info: hamta_data() misslyckades: {e}")

    # 2) Annars, prova läsa direkt från Sheets
    if df.empty and _sheets_ok:
        df = _load_df_from_sheets()

    st.session_state["_df_ref"] = df


def _save_df_via_sheets(df: pd.DataFrame) -> None:
    """Skriv hela DataFrame till bladet (om sheets-modulen finns)."""
    if not _sheets_ok:
        st.info("Ingen sheets-modul tillgänglig – hoppar över skrivning.")
        return
    try:
        save_dataframe(df)
        st.success("Sparat till Google Sheets.")
    except Exception as e:
        st.warning(f"⚠️ Kunde inte spara via sheets-modulen: {e}")


# ── Sidhuvud ───────────────────────────────────────────────────────────────
st.title("K-pf-rslag")

# ── Tillfällig felsökning för Google Sheets (runtime override + diagnos) ───
with st.expander("🛠 Google Sheets – felsökning (tillfällig)"):
    st.caption(
        "Om secrets strular kan du klistra in ditt **Service Account JSON** här "
        "så används det bara för den här sessionen (lagras inte)."
    )
    pasted = st.text_area(
        "Klistra in Service Account JSON (eller base64-JSON / key=value-format):",
        height=140,
        key="sa_paste_area",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Använd inklistad SA-JSON för denna session"):
            try:
                set_runtime_service_account(pasted)
                st.success("Service account satt för den här sessionen ✅")
            except Exception as e:
                st.error(f"Kunde inte tolka SA-JSON: {e}")
    with col_b:
        if st.button("Visa secrets-nycklar (diagnos)"):
            try:
                diag = secrets_diagnose()
                st.json(diag)  # visar endast nyckelNAMN/struktur – aldrig hemligheter
            except Exception as e:
                st.error(f"Kunde inte hämta secrets-diagnos: {e}")

# ── Data-initialisering ────────────────────────────────────────────────────
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

# ── Huvudinnehåll (flikar) ─────────────────────────────────────────────────
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
        # Om vyn returnerar ett uppdaterat DF: lagra tillbaka
        if isinstance(df_out, pd.DataFrame) and not df_out.equals(df_in):
            st.session_state["_df_ref"] = df_out
            st.success("Vyn returnerade uppdaterat DataFrame – uppdaterade sessionens data.")

# ── Fotnot / status ────────────────────────────────────────────────────────
st.caption("Build OK • Om spar/läsning från Sheets strular – använd felsökningssektionen ovan eller prova 'Läs in från Sheets igen'.")
