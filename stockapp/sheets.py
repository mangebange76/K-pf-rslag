# stockapp/sheets.py
from __future__ import annotations

import json
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception as e:
    raise RuntimeError("gspread/Google libs saknas. Lägg till 'gspread' och 'google-auth' i requirements.") from e


# ---------------- Backoff ----------------
def _with_backoff(func, *args, **kwargs):
    delays = [0.0, 0.6, 1.2, 2.4]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    if last:
        raise last
    raise RuntimeError("Okänt fel i _with_backoff")


# ---------------- Credentials ----------------
def _load_credentials_dict() -> dict:
    key = "GOOGLE_CREDENTIALS"
    if key not in st.secrets:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i st.secrets.")
    raw = st.secrets[key]
    # Tillåt både dict och JSON-sträng
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            raise RuntimeError("GOOGLE_CREDENTIALS är en str men inte giltig JSON.")
    raise RuntimeError("GOOGLE_CREDENTIALS fanns men hade okänt format (varken dict eller JSON-sträng).")


def _client() -> gspread.Client:
    creds_dict = _load_credentials_dict()
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)


def get_spreadsheet():
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("Saknar SHEET_URL i st.secrets.")
    cli = _client()
    return _with_backoff(cli.open_by_url, url)


# ---------------- Worksheet helpers ----------------
def list_worksheet_titles() -> List[str]:
    ss = get_spreadsheet()
    try:
        shs = _with_backoff(ss.worksheets)
        return [s.title for s in shs]
    except Exception:
        return []


def _get_or_create_worksheet(title: str, header: Optional[List[str]] = None):
    ss = get_spreadsheet()
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        ws = _with_backoff(ss.add_worksheet, title=title, rows=2000, cols=200)
        if header:
            _with_backoff(ws.update, [header])
        return ws


# ---------------- Public IO ----------------
def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läser ett blad där första raden är header.
    Returnerar tom DF om bladet är tomt (behåller kolumnnamn i appen).
    """
    ws = _get_or_create_worksheet(title)
    values = _with_backoff(ws.get_all_values)()
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    if not header:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header[:len(rows[0])] if rows else header)
    # Trunka extra kolumner om någon rad är kortare
    df = df.reindex(columns=header)
    return df


def ws_write_df(title: str, df: pd.DataFrame) -> None:
    """
    Skriv hela DF → blad. Allt som str (så Sheets tar emot).
    """
    ws = _get_or_create_worksheet(title, header=list(df.columns))
    # Clear + update (med backoff)
    _with_backoff(ws.clear)
    values = [list(df.columns)] + [[str(x) for x in row] for row in df.astype(object).values.tolist()]
    _with_backoff(ws.update, values)
