# sheets_utils.py
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

# ============================================
# Bakåtkompatibel Sheets-hantering
#  - Stöd för gamla secrets: GOOGLE_CREDENTIALS + SHEET_URL
#  - Stöd för nya secrets:   gcp_service_account + sheets.main_id
#  - Stöd för svenska bladnamn: "Blad1", "Valutakurser"
#  - Lokal fallback: CSV/JSON om Sheets ligger nere
# ============================================

# Lokala fallback-filer (lagras i appens arbetskatalog)
_LOCAL_DATA = Path("local_data.csv")
_LOCAL_RATES = Path("local_rates.json")
_LOCAL_LOGS = Path("local_logs.json")

# Standardkonfig (kan överskridas i st.secrets["sheets"])
_DEFAULT_SHEET_CONF = {
    "main_id": "",            # Sheet-ID (ny modell)
    "data_ws": "Data",        # Primärt datablad (ny modell)
    "rates_ws": "Rates",      # Valutablad (ny modell)
    "logs_ws": "FetchLog",    # Loggblad
    "snapshots_prefix": "Snap-",
}

# Kandidatnamn för bakåtkompatibilitet
_DATA_WS_CANDIDATES = ["Data", "Blad1"]          # Nytt först, sedan svenskt default
_RATES_WS_CANDIDATES = ["Rates", "Valutakurser"] # Nytt först, sedan svenskt default

# --------- Hjälpare ---------

def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def _read_secrets(path: str, default=None):
    try:
        ref = st.secrets
        for part in path.split("."):
            ref = ref.get(part, {})
        return ref if ref else default
    except Exception:
        return default

def _sheet_conf() -> dict:
    # Läs ny konfig (kan saknas)
    conf = dict(_DEFAULT_SHEET_CONF)
    try:
        cfg = st.secrets.get("sheets", {})
        if isinstance(cfg, dict):
            conf.update(cfg)
    except Exception:
        pass
    # Lägg även till legacy nycklar om de finns (för status/debug)
    try:
        if "SHEET_URL" in st.secrets:
            conf["legacy_url"] = st.secrets["SHEET_URL"]
        if "SHEET_ID" in st.secrets:
            conf["legacy_id"] = st.secrets["SHEET_ID"]
    except Exception:
        pass
    return conf

def _legacy_sa_info() -> Optional[dict]:
    # Gamla appens servicekonto: st.secrets["GOOGLE_CREDENTIALS"]
    try:
        return st.secrets.get("GOOGLE_CREDENTIALS", None)
    except Exception:
        return None

def _new_sa_info() -> Optional[dict]:
    # Ny appens servicekonto: st.secrets["gcp_service_account"]
    try:
        return st.secrets.get("gcp_service_account", None)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _gspread_client():
    """Skapar gspread-klient från antingen legacy eller ny SA-info."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        sa_info = _legacy_sa_info() or _new_sa_info()
        if not sa_info:
            return None

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None

def _open_sheet():
    """
    Försök öppna kalkylark i robust ordning:
      1) Legacy URL: st.secrets["SHEET_URL"]  -> open_by_url
      2) Nytt sheet_id: st.secrets["sheets"]["main_id"] -> open_by_key
      3) Legacy ID: st.secrets["SHEET_ID"] -> open_by_key
    """
    gc = _gspread_client()
    if not gc:
        return None, "no_client"

    conf = _sheet_conf()
    # 1) legacy URL
    legacy_url = conf.get("legacy_url") or st.secrets.get("SHEET_URL", None)
    if legacy_url:
        try:
            sh = gc.open_by_url(legacy_url)
            return sh, "by_url_legacy"
        except Exception:
            pass

    # 2) nytt ID
    main_id = conf.get("main_id", "")
    if main_id:
        try:
            sh = gc.open_by_key(main_id)
            return sh, "by_key_new"
        except Exception:
            pass

    # 3) legacy ID
    legacy_id = conf.get("legacy_id") or st.secrets.get("SHEET_ID", "")
    if legacy_id:
        try:
            sh = gc.open_by_key(legacy_id)
            return sh, "by_key_legacy"
        except Exception:
            pass

    return None, "open_failed"

def _first_existing_ws(sh, candidates: List[str]):
    """Returnera första existerande worksheet av listan, annars None."""
    try:
        titles = [ws.title for ws in sh.worksheets()]
    except Exception:
        titles = []
    for name in candidates:
        if name and name in titles:
            try:
                return sh.worksheet(name)
            except Exception:
                continue
    return None

def _get_or_create_ws(sh, preferred_name: str, candidates: List[str], rows: int = 1000, cols: int = 40):
    """
    Hitta första existerande av kandidaterna. Om ingen finns:
    skapa ark med preferred_name.
    """
    ws = _first_existing_ws(sh, candidates)
    if ws:
        return ws
    name = preferred_name or (candidates[0] if candidates else "Data")
    try:
        return sh.add_worksheet(title=name, rows=str(rows), cols=str(cols))
    except Exception:
        return None

def _write_df_to_ws(ws, df: pd.DataFrame) -> bool:
    """Skriv ett DataFrame till ws. Returnerar True/False, kraschar inte."""
    try:
        ws.clear()
        if df.empty:
            ws.update("A1", [[""]])
            return True
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
    df = pd.DataFrame(rows, columns=header)
    return df

# ---------- STATUS/DIAGNOSTIK ----------

def sheets_status() -> dict:
    conf = _sheet_conf()
    sa = bool(_legacy_sa_info() or _new_sa_info())
    sh, method = _open_sheet()
    ok = bool(sh)
    using = "sheets" if ok else "local"
    msg = ""
    if not sa:
        msg = "Servicekonto (GOOGLE_CREDENTIALS eller gcp_service_account) saknas i secrets."
    elif not ok:
        if method == "open_failed":
            msg = "Kunde inte öppna kalkylarket: kontrollera SHEET_URL / sheets.main_id / SHEET_ID och att arket är delat med servicekontot."
        elif method == "no_client":
            msg = "Kunde inte skapa klient. Kontrollera servicekontot i secrets."
    # Vilka bladnamn appen letar efter:
    data_ws_pref = _sheet_conf().get("data_ws") or _DATA_WS_CANDIDATES[0]
    rates_ws_pref = _sheet_conf().get("rates_ws") or _RATES_WS_CANDIDATES[0]
    return {
        "ok": ok,
        "using": using,
        "method": method,
        "message": msg,
        "preferred_data_ws": data_ws_pref,
        "preferred_rates_ws": rates_ws_pref,
        "candidates_data": _DATA_WS_CANDIDATES,
        "candidates_rates": _RATES_WS_CANDIDATES,
    }

def reset_sheets_client():
    try:
        st.cache_resource.clear()
    except Exception:
        pass

# ---------- DATA ----------

@st.cache_data(show_spinner=False)
def hamta_data() -> pd.DataFrame:
    """
    Läs data-bladet:
      - Försök "Data" eller "Blad1" (i den ordningen), eller preferred från secrets.
      - Lokal fallback om Sheets inte nås.
    """
    sh, _ = _open_sheet()
    if sh:
        pref = _sheet_conf().get("data_ws", "") or _DATA_WS_CANDIDATES[0]
        ws = _get_or_create_ws(sh, pref, [pref] + [c for c in _DATA_WS_CANDIDATES if c != pref])
        if ws:
            df = _read_ws_to_df(ws)
            if not df.empty:
                return df

    # Lokal fallback
    if _LOCAL_DATA.exists():
        try:
            return pd.read_csv(_LOCAL_DATA)
        except Exception:
            pass
    # Session fallback
    return st.session_state.get("df_cached", pd.DataFrame())

def spara_data(df: pd.DataFrame) -> bool:
    ok = False
    sh, _ = _open_sheet()
    if sh:
        pref = _sheet_conf().get("data_ws", "") or _DATA_WS_CANDIDATES[0]
        ws = _get_or_create_ws(sh, pref, [pref] + [c for c in _DATA_WS_CANDIDATES if c != pref], rows=max(1000, len(df)+10), cols=max(20, len(df.columns)+5))
        if ws:
            ok = _write_df_to_ws(ws, df)
    try:
        df.to_csv(_LOCAL_DATA, index=False)
    except Exception:
        pass
    st.session_state["df_cached"] = df.copy()
    return ok

# ---------- VALUTOR ----------

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser() -> Dict[str, float]:
    """
    Läs valutakurser från Rates/Valutakurser:
      - Stöd för header "Code,Rate" (nytt) OCH "Valuta,Kurs" (gammalt).
    """
    sh, _ = _open_sheet()
    if sh:
        pref = _sheet_conf().get("rates_ws", "") or _RATES_WS_CANDIDATES[0]
        ws = _first_existing_ws(sh, [pref] + [c for c in _RATES_WS_CANDIDATES if c != pref])
        if ws:
            df = _read_ws_to_df(ws)
            if not df.empty:
                # Försök nytt schema
                if set(["Code","Rate"]).issubset(df.columns):
                    try:
                        d = {r["Code"]: float(r["Rate"]) for _, r in df.iterrows() if str(r["Code"]).strip()}
                        if d:
                            return d
                    except Exception:
                        pass
                # Försök gammalt schema
                if set(["Valuta","Kurs"]).issubset(df.columns):
                    out = {}
                    for _, r in df.iterrows():
                        cur = str(r.get("Valuta", "")).upper().strip()
                        val = str(r.get("Kurs", "")).replace(",", ".").strip()
                        try:
                            out[cur] = float(val)
                        except Exception:
                            pass
                    if out:
                        return out

    # Lokal fallback
    if _LOCAL_RATES.exists():
        try:
            return json.loads(_LOCAL_RATES.read_text())
        except Exception:
            pass

    # Defaults
    return st.session_state.get("saved_rates", {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0})

def spara_valutakurser(rates: Dict[str, float]) -> bool:
    """
    Skriv valutakurser:
      - Om blad redan finns och har "Valuta,Kurs" → skriv i gammalt format
      - Annars skriv "Code,Rate" (nytt format)
    """
    ok = False
    sh, _ = _open_sheet()
    df_new = pd.DataFrame([{"Code": k, "Rate": float(v)} for k, v in sorted(rates.items())])
    df_old = pd.DataFrame([["Valuta","Kurs"]] + [[k, str(v)] for k, v in sorted(rates.items())])

    if sh:
        pref = _sheet_conf().get("rates_ws", "") or _RATES_WS_CANDIDATES[0]
        # Finns befintligt ark av någon kandidat?
        ws = _first_existing_ws(sh, [pref] + [c for c in _RATES_WS_CANDIDATES if c != pref])
        if ws:
            # Läs första raden för att avgöra schema
            try:
                vals = ws.get_all_values()
                header = vals[0] if vals else []
            except Exception:
                header = []
            if header and "Valuta" in header and "Kurs" in header:
                ok = _write_df_to_ws(ws, pd.DataFrame(df_old.values[1:], columns=df_old.values[0]))
            else:
                ok = _write_df_to_ws(ws, df_new)
        else:
            # Skapa nytt ark i nytt format (Code/Rate)
            ws2 = _get_or_create_ws(sh, pref, [pref], rows=20, cols=5)
            if ws2:
                ok = _write_df_to_ws(ws2, df_new)

    # Lokal kopia
    try:
        _LOCAL_RATES.write_text(json.dumps(rates))
    except Exception:
        pass
    st.session_state["saved_rates"] = dict(rates)
    return ok

def hamta_valutakurs(valuta: str, user_rates: Dict[str, float] | None = None) -> float:
    if user_rates and valuta in user_rates:
        return float(user_rates[valuta])
    saved = las_sparade_valutakurser()
    return float(saved.get(valuta, 1.0))

# ---------- SNAPSHOTS ----------

def _snapshot_ws_name() -> str:
    pref = _sheet_conf().get("snapshots_prefix", "Snap-")
    return pref + datetime.now().strftime("%Y-%m-%d")

def skapa_snapshot_om_saknas(df: pd.DataFrame) -> Tuple[bool, str]:
    sh, _ = _open_sheet()
    if not sh:
        return False, "Snapshot hoppades över (ingen Sheets-anslutning)."

    today_name = _snapshot_ws_name()
    try:
        for ws in sh.worksheets():
            if ws.title == today_name:
                return False, f"Snapshot '{today_name}' fanns redan."
        ws = _get_or_create_ws(sh, today_name, [today_name], rows=max(1000, len(df)+10), cols=max(20, len(df.columns)+5))
        if not ws:
            return False, "Kunde inte skapa snapshot-ark."
        ok = _write_df_to_ws(ws, df)
        return (True, f"Snapshot '{today_name}' skapad.") if ok else (False, "Misslyckades skriva snapshot.")
    except Exception:
        return False, "Snapshot misslyckades (okänt fel)."

# ---------- LOGG ----------

def spara_hamtlogg(logs: List[dict]) -> Tuple[bool, str]:
    if not logs:
        return False, "Ingen logg att spara."
    sh, _ = _open_sheet()
    df = pd.DataFrame(logs)
    if sh:
        pref = _sheet_conf().get("logs_ws", "FetchLog")
        ws = _get_or_create_ws(sh, pref, [pref], rows=max(1000, len(df)+10), cols=max(10, len(df.columns)+2))
        if ws:
            ok = _write_df_to_ws(ws, df)
            if ok:
                return True, "Hämtningslogg sparad till Sheets."
            return False, "Kunde inte skriva logg till Sheets."
    # Lokal fallback
    try:
        _LOCAL_LOGS.write_text(json.dumps(logs))
        return True, "Hämtningslogg sparad lokalt."
    except Exception:
        return False, "Kunde inte spara logg lokalt."
