# stockapp/sheets.py
from __future__ import annotations
import json, re, time
from typing import Optional, List
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# --------- Scopes & auth (samma som i din gamla app) ----------
_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

def _sa_from_secrets() -> dict:
    raw = st.secrets["GOOGLE_CREDENTIALS"]  # i din gamla app är detta en dict
    if isinstance(raw, dict):
        sa = dict(raw)
    else:
        # Om någon gång blir sträng – stöd JSON-sträng
        sa = json.loads(raw)
    if "private_key" in sa and isinstance(sa["private_key"], str):
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    return sa

def _client() -> gspread.Client:
    sa = _sa_from_secrets()
    creds = Credentials.from_service_account_info(sa, scopes=_SCOPES)
    return gspread.authorize(creds)

def _spreadsheet_url() -> str:
    return st.secrets["SHEET_URL"]  # exakt som i din gamla app

def _with_backoff(func, *args, **kwargs):
    for delay in (0, 0.5, 1.0, 2.0):
        if delay:
            time.sleep(delay)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last  # noqa: F821

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.strip().lower())

def _resolve_worksheet(sh: gspread.Spreadsheet, preferred: Optional[str]) -> gspread.Worksheet:
    names = [ws.title for ws in sh.worksheets()]
    if not names:
        return sh.add_worksheet(title="Blad1", rows="100", cols="26")

    # 1) exakt
    if preferred and preferred in names:
        return sh.worksheet(preferred)

    # 2) case/whitespace/tecken-oberoende
    if preferred:
        p = _norm(preferred)
        for n in names:
            if _norm(n) == p:
                return sh.worksheet(n)

    # 3) vanliga kandidater (din gamla default är "Blad1")
    for cand in ["Blad1", "Data", "Sheet1", "Ark1", "Blad", "Sheet", "Ark"]:
        for n in names:
            if _norm(n) == _norm(cand):
                return sh.worksheet(n)

    # 4) fallback: första bladet
    return sh.worksheets()[0]

# --------- Publikt API som appen använder ----------
def list_sheet_names(spreadsheet_url: Optional[str] = None) -> List[str]:
    cl = _client()
    sh = _with_backoff(cl.open_by_url, spreadsheet_url or _spreadsheet_url())
    return [ws.title for ws in sh.worksheets()]

def get_ws(spreadsheet_url: Optional[str] = None,
           worksheet_name: Optional[str] = None) -> gspread.Worksheet:
    """
    Öppnar kalkylblad via URL (inte ID) precis som din gamla app.
    Väljer blad enligt:
      1) worksheet_name (om angivet) eller st.secrets["WORKSHEET_NAME"]
      2) annars 'Blad1'
      3) annars första bladet
    """
    cl = _client()
    sh = _with_backoff(cl.open_by_url, spreadsheet_url or _spreadsheet_url())
    preferred = worksheet_name or st.secrets.get("WORKSHEET_NAME") or "Blad1"
    return _resolve_worksheet(sh, preferred)

def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    """
    Läser på samma sätt som din gamla app: get_all_records() (header första raden).
    Faller tillbaka till get_all_values() om arket saknar records.
    """
    try:
        records = _with_backoff(ws.get_all_records)
        df = pd.DataFrame(records)
        if not df.empty:
            return df
    except Exception:
        pass
    # fallback: råa values
    values = _with_backoff(ws.get_all_values)
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=header)

def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    """
    Samma mönster som din gamla app: clear + update([header] + strängvärden)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ws_write_df: df måste vara en pandas.DataFrame")
    header = list(map(str, df.columns.tolist()))
    body = [header] + df.astype(str).values.tolist()
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def save_dataframe(df: pd.DataFrame,
                   spreadsheet_url: Optional[str] = None,
                   worksheet_name: Optional[str] = None) -> None:
    ws = get_ws(spreadsheet_url=spreadsheet_url, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
