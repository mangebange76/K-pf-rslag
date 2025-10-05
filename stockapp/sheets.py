# stockapp/sheets.py
from __future__ import annotations

import json
import time
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

# gspread + Google creds
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception as e:
    gspread = None
    Credentials = None

# ------------------------------------------------------------
# Backoff-hjälpare
# ------------------------------------------------------------
def with_backoff(func, *args, **kwargs):
    """Kör en gspread-funktion med enkel backoff för att mildra 429/kvotfel."""
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    # sista felet upp
    raise last_err


# ------------------------------------------------------------
# Credentials + klient
# ------------------------------------------------------------
def _load_credentials_dict() -> Tuple[dict, str]:
    """
    Försök läsa credentials ur st.secrets i följande ordning:
    - GOOGLE_CREDENTIALS (dict eller JSON-sträng)
    - GOOGLE_CREDENTIALS_JSON (JSON-sträng)
    - SERVICE_ACCOUNT_JSON (JSON-sträng)
    Returnerar (dict, källa_namn) eller raise RuntimeError.
    """
    srcs = ["GOOGLE_CREDENTIALS", "GOOGLE_CREDENTIALS_JSON", "SERVICE_ACCOUNT_JSON"]
    for key in srcs:
        try:
            raw = st.secrets[key]
        except Exception:
            continue
        # kan redan vara dict
        if isinstance(raw, dict):
            return raw, key
        # kan vara JSON-sträng
        try:
            data = json.loads(str(raw))
            if isinstance(data, dict):
                return data, key
        except Exception:
            pass
        raise RuntimeError(f"{key} fanns men hade okänt format (varken dict eller JSON-sträng).")
    raise RuntimeError("Hittade inga Google credentials i st.secrets (GOOGLE_CREDENTIALS / *_JSON).")


def _client():
    if gspread is None or Credentials is None:
        raise RuntimeError("gspread eller google-oauth saknas i miljön.")
    creds_dict, _ = _load_credentials_dict()
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)


# ------------------------------------------------------------
# Spreadsheet + worksheets
# ------------------------------------------------------------
def _open_spreadsheet(cli):
    # Försök läsa in URL först; annars ID
    url = None
    sid = None
    try:
        url = st.secrets["SHEET_URL"]
    except Exception:
        pass
    try:
        sid = st.secrets["SHEET_ID"]
    except Exception:
        pass

    if url:
        return with_backoff(cli.open_by_url, url)
    if sid:
        return with_backoff(cli.open_by_key, sid)
    raise RuntimeError("SHEET_URL/SHEET_ID saknas i st.secrets.")

def get_spreadsheet():
    cli = _client()
    return _open_spreadsheet(cli)

def _get_or_create_worksheet(title: str):
    ss = get_spreadsheet()
    try:
        return with_backoff(ss.worksheet, title)
    except Exception:
        # Skapa minimalt blad om saknas (enbart rubrikrad Ticker – resten fylls i appen)
        ws = with_backoff(ss.add_worksheet, title=title, rows=2, cols=1)
        with_backoff(ws.update, [["Ticker"]])
        return ws

def delete_worksheet(title: str):
    ss = get_spreadsheet()
    try:
        ws = with_backoff(ss.worksheet, title)
    except Exception:
        return
    with_backoff(ss.del_worksheet, ws)

def list_worksheet_titles() -> List[str]:
    try:
        ss = get_spreadsheet()
        sheets = with_backoff(ss.worksheets)
        return [s.title for s in sheets]
    except Exception:
        return []


# ------------------------------------------------------------
# Läs / skriv DataFrame
# ------------------------------------------------------------
def _sanitize_header(vals: List[str]) -> List[str]:
    """Trim + ta bort BOM och gör rubriker unika vid behov."""
    cleaned = []
    seen = {}
    for v in vals:
        s = str(v or "").strip().replace("\ufeff", "")
        if s in seen:
            seen[s] += 1
            s = f"{s} ({seen[s]})"
        else:
            seen[s] = 1
        cleaned.append(s)
    return cleaned

def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läs ett ark till DataFrame.
    - Om arket saknas skapas det med rubriken 'Ticker'.
    - Hanterar tomma ark → tom DF.
    - Första raden = header.
    """
    ws = _get_or_create_worksheet(title)
    rows = with_backoff(ws.get_all_values)()
    if not rows:
        return pd.DataFrame()
    header = _sanitize_header(rows[0]) if rows[0] else []
    data = rows[1:] if len(rows) > 1 else []
    if not header:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=header)
    # trimma whitespace i alla celler (strings)
    for c in df.columns:
        try:
            df[c] = df[c].map(lambda x: str(x).strip())
        except Exception:
            pass
    return df


def _to_gs_string(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (int, float)):
        # Håll punkt-notation; Streamlit/GS klarar båda
        return str(x)
    return str(x)

def ws_write_df(title: str, df: pd.DataFrame):
    """
    Skriv en DataFrame till arket `title`.
    Skapar arket om det saknas. Rensar innan uppdatering.
    """
    ws = _get_or_create_worksheet(title)
    # Bygg body
    cols = list(map(str, df.columns.tolist()))
    body = [cols]
    if not df.empty:
        values = df.astype(object).applymap(_to_gs_string).values.tolist()
        body.extend(values)
    # Clear + update
    with_backoff(ws.clear)
    with_backoff(ws.update, body)
