# stockapp/sheets.py
from __future__ import annotations

import json
import time
from collections.abc import Mapping
import ast
from typing import Callable, List, Optional

import pandas as pd
import streamlit as st

# Tredjeparts
import gspread
from google.oauth2.service_account import Credentials


# ---------- Backoff-hjälpare ----------
def _with_backoff(func: Callable, *args, **kwargs):
    """Kör func med enkel exponential backoff (lindrar 429/kvotspärrar)."""
    delays = [0, 0.5, 1.0, 2.0, 4.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err


# Exportera ett publikt alias som vissa moduler använder
def with_backoff(func: Callable, *args, **kwargs):
    return _with_backoff(func, *args, **kwargs)


# ---------- Credentials ----------
def _load_credentials_dict() -> dict:
    """
    Läser GOOGLE_CREDENTIALS ur st.secrets och accepterar:
    - Streamlit SecretDict / valfri Mapping
    - Python dict
    - JSON-sträng
    - Python-literal str (t.ex. "{'type': 'service_account', ...}")
    """
    key = "GOOGLE_CREDENTIALS"
    if key not in st.secrets:
        raise RuntimeError("GOOGLE_CREDENTIALS saknas i secrets.")

    val = st.secrets[key]

    # 1) SecretDict / Mapping
    if isinstance(val, Mapping):
        return dict(val)

    # 2) Ren dict
    if isinstance(val, dict):
        return val

    # 3) Sträng → prova JSON, annars python-literal
    if isinstance(val, str):
        s = val.strip()
        # JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # Python-literal (enkla citationstecken etc)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, Mapping):
                return dict(obj)
        except Exception:
            pass
        raise RuntimeError(
            "GOOGLE_CREDENTIALS fanns men kunde inte tolkas – strängen var varken giltig JSON eller python-dict."
        )

    # 4) Sista chans: casta till dict om möjligt
    try:
        return dict(val)
    except Exception:
        pass

    raise RuntimeError(f"GOOGLE_CREDENTIALS hade oväntad typ: {type(val)}. Förväntar Mapping/dict eller JSON-sträng.")


# ---------- GSpread-klient ----------
def _client() -> gspread.Client:
    cred_dict = _load_credentials_dict()
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(cred_dict, scopes=scopes)
    return gspread.authorize(creds)


# ---------- Spreadsheet helpers ----------
def get_spreadsheet():
    """Returnerar öppnat Spreadsheet via URLn i secrets."""
    if "SHEET_URL" not in st.secrets:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    url = st.secrets["SHEET_URL"]
    cli = _client()
    return _with_backoff(cli.open_by_url, url)


def _get_or_create_worksheet(title: str):
    ss = get_spreadsheet()
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        # Skapa nytt om det saknas
        _with_backoff(ss.add_worksheet, title=title, rows=1000, cols=50)
        return _with_backoff(ss.worksheet, title)


# ---------- Publika IO-funktioner ----------
def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läser ett ark som DataFrame. Första raden tolkas som header.
    Tomt ark → tom DataFrame med 0 kolumner.
    """
    ws = _get_or_create_worksheet(title)
    rows = _with_backoff(ws.get_all_values)
    if not rows:
        return pd.DataFrame()

    header = rows[0] if rows else []
    # Ta bort helt tomma tail-kolumner i headern
    while header and header[-1] == "":
        header.pop()

    if not header:
        return pd.DataFrame()

    data_rows = [r[: len(header)] for r in rows[1:]]  # trimma rader till headerlängd
    df = pd.DataFrame(data_rows, columns=header)
    return df


def ws_write_df(title: str, df: pd.DataFrame):
    """
    Skriver DataFrame till ett ark (ersätter allt). Konverterar alla celler till str.
    """
    ws = _get_or_create_worksheet(title)
    if df is None or df.empty:
        # Skriv åtminstone header om tom
        if isinstance(df, pd.DataFrame) and list(df.columns):
            body = [list(map(str, df.columns.tolist()))]
        else:
            body = [[]]
        _with_backoff(ws.clear)
        _with_backoff(ws.update, body)
        return

    # Säkerställ strängar och header
    cols = list(map(str, df.columns.tolist()))
    values = df.astype(str).fillna("").values.tolist()
    body = [cols] + values

    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)


def list_worksheet_titles() -> List[str]:
    """Returnerar alla bladnamn i kalkylarket."""
    ss = get_spreadsheet()
    return [w.title for w in _with_backoff(ss.worksheets)]


def delete_worksheet(title: str) -> bool:
    """Tar bort ett blad. Returnerar True om det lyckades eller om bladet saknades."""
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, title)
    except Exception:
        return True
    _with_backoff(ss.del_worksheet, ws)
    return True
