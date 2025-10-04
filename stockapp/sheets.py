# stockapp/sheets.py
from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
    from gspread_dataframe import set_with_dataframe
except Exception as e:
    gspread = None  # type: ignore


# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
SHEET_URL = st.secrets.get("SHEET_URL", "")
CRED_DICT = st.secrets.get("GOOGLE_CREDENTIALS", {})

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]


# ------------------------------------------------------------
# Backoff-hjälpare
# ------------------------------------------------------------
def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err


# ------------------------------------------------------------
# Autentisering / klient
# ------------------------------------------------------------
def _client() -> "gspread.Client":
    if gspread is None:
        raise RuntimeError("gspread saknas i miljön.")
    if not CRED_DICT or "client_email" not in CRED_DICT or "private_key" not in CRED_DICT:
        raise RuntimeError("Hittade inga service account-uppgifter (minst client_email + private_key).")
    creds = Credentials.from_service_account_info(CRED_DICT, scopes=SCOPE)
    return gspread.authorize(creds)


def get_spreadsheet() -> "gspread.Spreadsheet":
    cli = _client()
    if SHEET_URL:
        return _with_backoff(cli.open_by_url, SHEET_URL)
    raise RuntimeError("Hittade inget Spreadsheet-ID/URL i secrets (SHEET_URL).")


def _get_or_create_worksheet(title: str) -> "gspread.Worksheet":
    ss = get_spreadsheet()
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        # skapa nytt blad om det saknas
        ws = _with_backoff(ss.add_worksheet, title=title, rows=100, cols=26)
        return ws


# ------------------------------------------------------------
# Publika funktioner
# ------------------------------------------------------------
def list_worksheet_titles() -> List[str]:
    ss = get_spreadsheet()
    wss = _with_backoff(ss.worksheets)
    return [w.title for w in wss]


def ws_read_df(title: str) -> pd.DataFrame:
    ws = _get_or_create_worksheet(title)
    # Hämta alla värden
    vals = _with_backoff(ws.get_all_values)
    if not vals:
        return pd.DataFrame()
    # Första raden = header
    header = vals[0]
    rows = vals[1:]
    if not header:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    # Försök konvertera numeriskt där det går
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    return df


def ws_write_df(title: str, df: pd.DataFrame):
    ws = _get_or_create_worksheet(title)
    # Rensa och skriv
    _with_backoff(ws.clear)
    if df is None or df.empty:
        # skriv bara header om vi inte har data
        _with_backoff(ws.update, [list(df.columns)] if df is not None else [[""]])
        return
    # set_with_dataframe tar hand om storlek
    _with_backoff(set_with_dataframe, ws, df, include_index=False, include_column_header=True)


def delete_worksheet(title: str) -> bool:
    """
    Tar bort ett blad om det existerar. Returnerar True om borttaget, annars False.
    """
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, title)
    except Exception:
        return False
    _with_backoff(ss.del_worksheet, ws)
    return True
