# stockapp/sheets.py
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Optional, List

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ------------------------------------------------------------
#  Scopes & auth – identiskt med din gamla app
# ------------------------------------------------------------
_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


def _sa_from_secrets() -> dict:
    """
    Läs servicekontot från st.secrets["GOOGLE_CREDENTIALS"].
    Stöder:
      - AttrDict (Streamlits inbyggda typ) / dict (nästlat) -> konverteras till vanlig dict
      - JSON-sträng (hela JSON-innehållet)
    Normaliserar private_key (\\n -> \n).
    """
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        raise RuntimeError("Saknar 'GOOGLE_CREDENTIALS' i secrets.")

    raw = st.secrets["GOOGLE_CREDENTIALS"]

    def _to_plain_dict(x):
        # Konvertera AttrDict/mapping (även nästlat) -> vanlig dict
        if isinstance(x, Mapping):
            return {k: _to_plain_dict(v) for k, v in x.items()}
        return x

    if isinstance(raw, str):
        # JSON-sträng
        sa = json.loads(raw)
    else:
        # AttrDict/dict/annan mapping
        sa = _to_plain_dict(raw)

    # Normalisera private_key (vanligt \\n-problem)
    pk = sa.get("private_key")
    if isinstance(pk, str) and "\\n" in pk:
        sa["private_key"] = pk.replace("\\n", "\n")

    if not sa.get("client_email") or not sa.get("private_key"):
        raise RuntimeError("GOOGLE_CREDENTIALS saknar 'client_email' eller 'private_key'.")

    return sa


def _client() -> gspread.Client:
    sa = _sa_from_secrets()
    creds = Credentials.from_service_account_info(sa, scopes=_SCOPES)
    return gspread.authorize(creds)


def _spreadsheet_url() -> str:
    """
    Läs kalkylbladets URL som i din gamla app: st.secrets["SHEET_URL"]
    """
    if "SHEET_URL" not in st.secrets:
        raise RuntimeError("Saknar 'SHEET_URL' i secrets.")
    return st.secrets["SHEET_URL"]


def _with_backoff(func, *args, **kwargs):
    """
    Liten backoff-hjälpare (mjukar upp 429/kvoter).
    """
    last_err = None
    for delay in (0, 0.5, 1.0, 2.0):
        if delay:
            time.sleep(delay)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.strip().lower())


def _resolve_worksheet(sh: gspread.Spreadsheet, preferred: Optional[str]) -> gspread.Worksheet:
    """
    Välj blad:
      1) exakt preferred/WORKSHEET_NAME
      2) case/whitespace-insensitiv match
      3) vanliga standarder: "Blad1", "Data", "Sheet1", ...
      4) första bladet
    """
    names = [ws.title for ws in sh.worksheets()]
    if not names:
        return sh.add_worksheet(title="Blad1", rows="100", cols="26")

    # 1) exakt
    if preferred and preferred in names:
        return sh.worksheet(preferred)

    # 2) normaliserad match
    if preferred:
        p = _norm(preferred)
        for n in names:
            if _norm(n) == p:
                return sh.worksheet(n)

    # 3) vanliga kandidater
    for cand in ["Blad1", "Data", "Sheet1", "Ark1", "Blad", "Sheet", "Ark"]:
        for n in names:
            if _norm(n) == _norm(cand):
                return sh.worksheet(n)

    # 4) fallback: första
    return sh.worksheets()[0]


# ------------------------------------------------------------
#  Publikt API – exakt vad appen anropar
# ------------------------------------------------------------
def list_sheet_names(spreadsheet_url: Optional[str] = None) -> List[str]:
    cl = _client()
    sh = _with_backoff(cl.open_by_url, spreadsheet_url or _spreadsheet_url())
    return [ws.title for ws in sh.worksheets()]


def get_ws(spreadsheet_url: Optional[str] = None,
           worksheet_name: Optional[str] = None) -> gspread.Worksheet:
    """
    Öppna via URL (inte ID), precis som i din gamla app.
    preferred blad:
      - 'worksheet_name' argument om angivet
      - annars secrets["WORKSHEET_NAME"] om finns
      - annars 'Blad1'
    """
    cl = _client()
    sh = _with_backoff(cl.open_by_url, spreadsheet_url or _spreadsheet_url())
    preferred = worksheet_name or st.secrets.get("WORKSHEET_NAME") or "Blad1"
    return _resolve_worksheet(sh, preferred)


def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    """
    Läs som i din gamla app:
     - försök get_all_records() (header = rad 1)
     - fallback till get_all_values() om records blir tomt
    """
    try:
        records = _with_backoff(ws.get_all_records)
        df = pd.DataFrame(records)
        if not df.empty:
            return df
    except Exception:
        pass

    values = _with_backoff(ws.get_all_values)
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=header)


def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    """
    Skriv som i din gamla app:
      clear() + update([header] + df.astype(str).values.tolist())
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
