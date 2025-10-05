from __future__ import annotations

import time
from typing import List
import pandas as pd
import streamlit as st

# --- gspread / auth ---
import gspread
from google.oauth2.service_account import Credentials

SHEET_URL = st.secrets["SHEET_URL"]

# service account + scopes
_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
_creds = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=_SCOPES
)
_client = gspread.authorize(_creds)

# ---------- helpers med backoff ----------
def _with_backoff(func, *args, **kwargs):
    """Prova om, med små fördröjningar – dämpar 429/kvotfel."""
    delays = [0.0, 0.5, 1.0, 2.0, 4.0]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last

def _spreadsheet():
    return _with_backoff(_client.open_by_url, SHEET_URL)

def _worksheet(title: str, create: bool = False):
    ss = _spreadsheet()
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        if not create:
            raise
        # skapa nytt blad med defaultstorlek
        _with_backoff(ss.add_worksheet, title=title, rows=1000, cols=50)
        ws = _with_backoff(ss.worksheet, title)
        # lägg tom rubrikrad om helt nytt
        _with_backoff(ws.update, [["Ticker"]])
        return ws

# ---------- publika API:n som appen använder ----------

def list_worksheet_titles() -> List[str]:
    ss = _spreadsheet()
    wss = _with_backoff(ss.worksheets)
    return [ws.title for ws in wss]

def delete_worksheet(title: str):
    ss = _spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, title)
    except Exception:
        return
    _with_backoff(ss.del_worksheet, ws)

def ws_read_df(title: str) -> pd.DataFrame:
    """Läs ett blad som dataframe (rubriker på rad 1). Robust mot 429."""
    ws = _worksheet(title, create=False)
    rows = _with_backoff(ws.get_all_values)()  # snabbare än get_all_records vid många kolumner
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    # Städa tomma kolumnnamn
    df = df.loc[:, ~df.columns.isna()]
    return df

def ws_write_df(title: str, df: pd.DataFrame):
    """Skriv över hela bladet med df (inkl header)."""
    ws = _worksheet(title, create=True)
    # töm först
    _with_backoff(ws.clear)
    # om df tom → skriv bara header
    if df is None or df.empty:
        _with_backoff(ws.update, [df.columns.tolist()] if df is not None else [["Ticker"]])
        return
    # konvertera allt till str (Google Sheets vill ha str vid bulk-update)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    _with_backoff(ws.update, values)
