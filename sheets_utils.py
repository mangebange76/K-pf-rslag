# sheets_utils.py
# ----------------------------------------------------------
# Läs/skriv mot Google Sheets med snälla fallbacks lokalt.
# Kräver st.secrets med antingen:
#   - gcp_service_account (standard Streamlit-format) OCH
#   - sheets: { main_id: "<Google Sheet ID>",
#               data_ws: "Data",
#               rates_ws: "Rates",
#               logs_ws: "FetchLog",
#               snapshots_prefix: "Snap-" }
# Om inget av detta finns -> jobbar i minnet + enkla lokala filer.
# ----------------------------------------------------------

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# ----------------- Konfiguration -----------------

_DEFAULT_SHEET_CONF = {
    "main_id": "",
    "data_ws": "Data",
    "rates_ws": "Rates",
    "logs_ws": "FetchLog",
    "snapshots_prefix": "Snap-",
}

_LOCAL_DATA = Path("local_data.csv")
_LOCAL_RATES = Path("local_rates.json")
_LOCAL_LOGS = Path("local_logs.json")


# ----------------- Hjälpare: tid -----------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ----------------- Hjälpare: Sheets -----------------

@st.cache_resource(show_spinner=False)
def _gspread_client():
    """Returnera gspread-klient eller None."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        sa = st.secrets.get("gcp_service_account", None)
        if not sa:
            return None

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc
    except Exception:
        return None


def _sheet_conf() -> dict:
    try:
        conf = dict(_DEFAULT_SHEET_CONF)
        conf.update(st.secrets.get("sheets", {}))
        return conf
    except Exception:
        return dict(_DEFAULT_SHEET_CONF)


def _open_sheet():
    gc = _gspread_client()
    if not gc:
        return None
    conf = _sheet_conf()
    sid = conf.get("main_id", "")
    if not sid:
        return None
    try:
        return gc.open_by_key(sid)
    except Exception:
        return None


def _get_or_create_ws(sh, name: str, rows: int = 1000, cols: int = 40):
    try:
        return sh.worksheet(name)
    except Exception:
        try:
            return sh.add_worksheet(title=name, rows=str(rows), cols=str(cols))
        except Exception:
            return None


def _write_df_to_ws(ws, df: pd.DataFrame):
    """Skriv DataFrame till worksheet (raderar allt först)."""
    try:
        ws.clear()
        if df.empty:
            ws.update("A1", [[""]])
            return True
        # Konvertera till listor
        values = [list(df.columns)]
        values += df.astype(object).where(pd.notnull(df), "").values.tolist()
        ws.update("A1", values)
        return True
    except Exception:
        return False


def _read_ws_to_df(ws) -> pd.DataFrame:
    try:
        values = ws.get_all_values()
    except Exception:
        return pd.DataFrame()
    if not values:
        return pd.DataFrame()
    header, *rows = values
    if not header:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=header)


# ----------------- DATA -----------------

@st.cache_data(show_spinner=False)
def hamta_data() -> pd.DataFrame:
    """Läs huvudtabellen från Sheets. Fallback: lokalt/minnet."""
    sh = _open_sheet()
    conf = _sheet_conf()
    if sh:
        ws = _get_or_create_ws(sh, conf["data_ws"])
        if ws:
            df = _read_ws_to_df(ws)
            if not df.empty:
                return df
    # Fallback lokalt
    if _LOCAL_DATA.exists():
        try:
            return pd.read_csv(_LOCAL_DATA)
        except Exception:
            pass
    # Fallback minne
    return st.session_state.get("df_cached", pd.DataFrame())


def spara_data(df: pd.DataFrame) -> bool:
    """Spara huvudtabellen till Sheets. Fallback: lokalt + minnet."""
    ok = False
    sh = _open_sheet()
    conf = _sheet_conf()
    if sh:
        ws = _get_or_create_ws(sh, conf["data_ws"])
        if ws:
            ok = _write_df_to_ws(ws, df)
    # Fallback lokalt + minne oavsett
    try:
        df.to_csv(_LOCAL_DATA, index=False)
    except Exception:
        pass
    st.session_state["df_cached"] = df.copy()
    return ok


# ----------------- VALUTOR -----------------

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser() -> Dict[str, float]:
    """Läs senaste sparade växelkurser från Sheets, annars lokalt/minnet."""
    sh = _open_sheet()
    conf = _sheet_conf()
    if sh:
        ws = _get_or_create_ws(sh, conf["rates_ws"])
        if ws:
            df = _read_ws_to_df(ws)
            if not df.empty and "Code" in df.columns and "Rate" in df.columns:
                try:
                    d = {r["Code"]: float(r["Rate"]) for _, r in df.iterrows() if r["Code"]}
                    if d:
                        return d
                except Exception:
                    pass
    # Fallback lokalt
    if _LOCAL_RATES.exists():
        try:
            return json.loads(_LOCAL_RATES.read_text())
        except Exception:
            pass
    # Fallback minne
    return st.session_state.get("saved_rates", {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0})


def spara_valutakurser(rates: Dict[str, float]) -> bool:
    """Spara växelkurser till Sheets. Fallback: lokalt + minnet."""
    ok = False
    sh = _open_sheet()
    conf = _sheet_conf()
    df = pd.DataFrame(
        [{"Code": k, "Rate": float(v)} for k, v in sorted(rates.items())]
    )
    if sh:
        ws = _get_or_create_ws(sh, conf["rates_ws"])
        if ws:
            ok = _write_df_to_ws(ws, df)
    # Fallback lokalt + minne oavsett
    try:
        _LOCAL_RATES.write_text(json.dumps(rates))
    except Exception:
        pass
    st.session_state["saved_rates"] = dict(rates)
    return ok


def hamta_valutakurs(valuta: str, user_rates: Dict[str, float] | None = None) -> float:
    """Hämta kurs för given valuta, prioriterar user_rates → sparat → 1.0."""
    if user_rates and valuta in user_rates:
        return float(user_rates[valuta])
    saved = las_sparade_valutakurser()
    return float(saved.get(valuta, 1.0))


# ----------------- SNAPSHOTS -----------------

def skapa_snapshot_om_saknas(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Skapar ett dags-snapshot som ny worksheet "Snap-YYYY-MM-DD" om det saknas.
    Returnerar (ok, meddelande).
    """
    sh = _open_sheet()
    if not sh:
        return False, "Snapshot hoppades över (ingen Sheets-anslutning)."

    conf = _sheet_conf()
    prefix = conf.get("snapshots_prefix", "Snap-")
    today_name = prefix + datetime.now().strftime("%Y-%m-%d")

    try:
        # Finns redan?
        for ws in sh.worksheets():
            if ws.title == today_name:
                return False, f"Snapshot '{today_name}' fanns redan."

        ws = _get_or_create_ws(sh, today_name, rows=max(1000, len(df) + 10), cols=max(20, len(df.columns) + 5))
        if not ws:
            return False, "Kunde inte skapa snapshot-ark."
        ok = _write_df_to_ws(ws, df)
        return (True, f"Snapshot '{today_name}' skapad.") if ok else (False, "Misslyckades skriva snapshot.")
    except Exception:
        return False, "Snapshot misslyckades (okänt fel)."


# ----------------- LOGG -----------------

def spara_hamtlogg(logs: List[dict]) -> Tuple[bool, str]:
    """
    Sparar hämtningslogg (st.session_state['fetch_logs']) till logs_ws.
    """
    if not logs:
        return False, "Ingen logg att spara."

    sh = _open_sheet()
    conf = _sheet_conf()
    df = pd.DataFrame(logs)

    if sh:
        ws = _get_or_create_ws(sh, conf["logs_ws"])
        if ws:
            ok = _write_df_to_ws(ws, df)
            if ok:
                return True, "Hämtningslogg sparad till Sheets."
            return False, "Kunde inte skriva logg till Sheets."

    # Fallback lokalt
    try:
        _LOCAL_LOGS.write_text(json.dumps(logs))
        return True, "Hämtningslogg sparad lokalt."
    except Exception:
        return False, "Kunde inte spara logg."
