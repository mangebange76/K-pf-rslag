# stockapp/sheets.py
from __future__ import annotations

import json
import math
import time
from typing import List, Optional

import pandas as pd
import streamlit as st

try:
    import gspread
except Exception:
    gspread = None  # type: ignore

try:
    from google.oauth2.service_account import Credentials
except Exception:
    Credentials = None  # type: ignore


# ------------------------------------------------------------
# Backoff helper
# ------------------------------------------------------------
def _with_backoff(func, *args, **kwargs):
    """
    Kör func(*args, **kwargs) med mjuk backoff.
    Användning:
        rows = _with_backoff(ws.get_all_values)
        ws   = _with_backoff(ss.worksheet, "Blad1")
    """
    delays = [0.0, 0.4, 0.8, 1.6, 3.0]
    err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = e
    # sista felet bubblas upp
    raise err  # type: ignore


# ------------------------------------------------------------
# Credential & client
# ------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _load_credentials():
    if Credentials is None:
        raise RuntimeError("google-auth saknas i miljön.")

    # Vanligaste: dict i TOML → AttrDict
    blob = st.secrets.get("GOOGLE_CREDENTIALS")
    if not blob:
        blob = st.secrets.get("GOOGLE_SERVICE_ACCOUNT")

    if not blob:
        # Förklara tydligt vilka nycklar vi spanar efter
        raise RuntimeError(
            "Hittade inga service account-uppgifter (minst client_email + private_key)."
        )

    # Konvertera till ren dict om AttrDict
    if not isinstance(blob, (dict,)):
        # Kan vara JSON-sträng
        try:
            blob = json.loads(str(blob))
        except Exception:
            # Sista försök: ge upp
            raise RuntimeError("GOOGLE_CREDENTIALS fanns men gick inte att tolka som JSON/dict.")

    # Säkerställ obligatoriska fält – fyll default där de saknas
    defaults = {
        "type": "service_account",
        "project_id": "",
        "private_key_id": "",
        "private_key": "",
        "client_email": "",
        "client_id": "",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "",
    }
    merged = {**defaults, **blob}

    if not merged.get("client_email") or not merged.get("private_key"):
        raise RuntimeError("Service account saknar client_email eller private_key.")

    try:
        creds = Credentials.from_service_account_info(merged, scopes=SCOPES)
    except Exception as e:
        raise RuntimeError(f"Kunde inte skapa service account Credentials: {e}")

    return creds


def _client():
    if gspread is None:
        raise RuntimeError("gspread saknas i miljön.")
    creds = _load_credentials()
    try:
        cli = gspread.authorize(creds)
    except Exception as e:
        raise RuntimeError(f"Kunde inte autorisera gspread: {e}")
    return cli


# ------------------------------------------------------------
# Spreadsheet helpers
# ------------------------------------------------------------
def _spreadsheet_id_or_url() -> str:
    """
    Hämta antingen full URL eller ID från secrets.
    Godtar:
      - SHEET_URL (full URL)
      - SPREADSHEET_ID (id)
      - GOOGLE_SHEETS.SPREADSHEET_URL / SPREADSHEET_ID
    """
    # Top-level
    url = st.secrets.get("SHEET_URL")
    if url:
        return str(url)

    gid = st.secrets.get("SPREADSHEET_ID")
    if gid:
        return str(gid)

    # Sektion GOOGLE_SHEETS
    gs = st.secrets.get("GOOGLE_SHEETS") or {}
    if isinstance(gs, dict):
        if gs.get("SPREADSHEET_URL"):
            return str(gs["SPREADSHEET_URL"])
        if gs.get("SPREADSHEET_ID"):
            return str(gs["SPREADSHEET_ID"])

    raise RuntimeError(
        "Hittade inget Spreadsheet-ID/URL i secrets. "
        "Stödjer t.ex. SHEET_URL, SPREADSHEET_ID eller GOOGLE_SHEETS.{SPREADSHEET_URL|SPREADSHEET_ID}."
    )


def get_spreadsheet():
    cli = _client()
    id_or_url = _spreadsheet_id_or_url()
    try:
        if str(id_or_url).startswith("http"):
            return _with_backoff(cli.open_by_url, id_or_url)
        else:
            return _with_backoff(cli.open_by_key, id_or_url)
    except Exception as e:
        raise RuntimeError(f"Kunde inte öppna kalkylbladet: {e}")


def list_worksheet_titles() -> List[str]:
    ss = get_spreadsheet()
    try:
        wss = _with_backoff(ss.worksheets)
    except Exception:
        return []
    titles = []
    for w in wss:
        try:
            titles.append(str(w.title))
        except Exception:
            pass
    return titles


def delete_worksheet(title: str) -> bool:
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, title)
    except Exception:
        return False
    try:
        _with_backoff(ss.del_worksheet, ws)
        return True
    except Exception:
        return False


def _get_or_create_ws(ss, title: str, rows: int = 200, cols: int = 26):
    """
    Hämta kalkylblad med titel, eller skapa om det saknas.
    """
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        pass

    # Skapa nytt
    try:
        return _with_backoff(ss.add_worksheet, title=title, rows=rows, cols=cols)
    except Exception as e:
        # Om det fallerar pga cellgräns etc – låt felet bubbla upp med bra text
        raise RuntimeError(f"Kunde inte skapa nytt blad '{title}': {e}")


# ------------------------------------------------------------
# Public API – read / write
# ------------------------------------------------------------
def ws_read_df(ws_title: str) -> pd.DataFrame:
    """
    Läs hela bladet till DataFrame.
    Använder get_all_values (snabbare än get_all_records vid breda blad).
    FIX: ingen extra () efter _with_backoff – annars blir det TypeError.
    """
    ss = get_spreadsheet()
    ws = _with_backoff(ss.worksheet, ws_title)

    rows = _with_backoff(ws.get_all_values)  # <— FIX: inget "()" här
    if not rows:
        return pd.DataFrame()

    header = rows[0] if rows else []
    data = rows[1:] if len(rows) > 1 else []

    # Fallback om headern är tom/konstig
    if not header or all((h is None) or (str(h).strip() == "") for h in header):
        try:
            records = _with_backoff(ws.get_all_records)
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()

    df = pd.DataFrame(data, columns=header)

    # Lätt typstäd – konvertera "siffriga" strängar där det går utan att paja text
    for c in df.columns:
        try:
            s = pd.Series(df[c])
            # Normalisera decimal-tecken och trimma
            s2 = s.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
            # Försök numeriskt – men behåll strängar där det inte går
            num = pd.to_numeric(s2, errors="coerce")
            # Om >70% är numeriskt → ta den konverteringen
            if num.notna().mean() >= 0.7:
                df[c] = num.fillna(0.0)
        except Exception:
            pass

    return df


def ws_write_df(ws_title: str, df: pd.DataFrame, include_index: bool = False) -> None:
    """
    Skriv DataFrame till bladet (ersätt hela innehållet).
    Skalar bladet försiktigt för att inte överskrida 10M-cellersgränsen.
    """
    if df is None:
        raise RuntimeError("ws_write_df: Inget DataFrame att skriva.")

    # Bygg 2D-array
    if include_index:
        df_to_write = df.reset_index()
    else:
        df_to_write = df.copy()

    # Konvertera alla värden till str för säker update
    values: List[List[str]] = []
    values.append([str(c) for c in df_to_write.columns])
    for _, row in df_to_write.iterrows():
        values.append([("" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)) for v in row.tolist()])

    ss = get_spreadsheet()

    # Rimlig storlek på nytt blad
    rows_need = max(10, len(values))
    cols_need = max(5, len(values[0]) if values else 0)

    ws = _get_or_create_ws(ss, ws_title, rows=rows_need, cols=cols_need)

    # Resiza (utan att blåsa upp extremt)
    try:
        _with_backoff(ws.resize, rows_need, cols_need)
    except Exception:
        # Ignorera resize-fel (t.ex. om cellgräns skulle passeras)
        pass

    # Töm och skriv
    _with_backoff(ws.clear)

    # Försök ett enda update-anrop; om det failar pga storlek, chunk:a
    try:
        _with_backoff(ws.update, values, value_input_option="USER_ENTERED")
        return
    except Exception:
        pass

    # Chunk fallback
    # Skriv header
    _with_backoff(ws.update, [values[0]], value_input_option="USER_ENTERED")
    # Skriv data i bitar
    chunk = 500
    start = 1
    while start < len(values):
        part = values[start : start + chunk]
        try:
            _with_backoff(ws.append_rows, part, value_input_option="USER_ENTERED")
        except Exception as e:
            raise RuntimeError(f"Misslyckades med chunkad skrivning: {e}")
        start += chunk
