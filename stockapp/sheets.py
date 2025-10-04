"""
Robust Google Sheets–IO utan externa beroenden (ingen gspread_dataframe).
Ger: ws_read_df, ws_write_df, list_worksheet_titles, get_spreadsheet.
Läser service account från st.secrets:
- GOOGLE_CREDENTIALS  (dict eller JSON-sträng)
- alt. GOOGLE_SHEETS: { SPREADSHEET_ID | SHEET_URL, GOOGLE_CREDENTIALS? }
Och Spreadsheet-ID/URL:
- SHEET_URL  eller  SPREADSHEET_ID  (även under GOOGLE_SHEETS)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# gspread + google auth
import gspread
from google.oauth2.service_account import Credentials


# ---------------- Backoff-hjälpare ----------------
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
    raise last_err


# ---------------- Secrets-hjälpare ----------------
def _from_secrets(*keys, default=None):
    cur: Any = st.secrets
    try:
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k)
        return cur if cur is not None else default
    except Exception:
        return default


def _load_credentials() -> Credentials:
    """
    Försöker läsa service account på flera sätt:
    - st.secrets["GOOGLE_CREDENTIALS"]  (dict eller JSON-sträng)
    - st.secrets["GOOGLE_SHEETS"]["GOOGLE_CREDENTIALS"]
    - st.secrets["GOOGLE_SHEETS"]["SERVICE_ACCOUNT_JSON"]
    - st.secrets["SERVICE_ACCOUNT_JSON"]
    - eller toppnivå-fält: client_email + private_key under GOOGLE_CREDENTIALS
    """
    raw = (
        _from_secrets("GOOGLE_CREDENTIALS")
        or _from_secrets("GOOGLE_SHEETS", "GOOGLE_CREDENTIALS")
        or _from_secrets("GOOGLE_SHEETS", "SERVICE_ACCOUNT_JSON")
        or _from_secrets("SERVICE_ACCOUNT_JSON")
    )
    data: Dict[str, Any] = {}

    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
        except Exception:
            # Kan vara env-liknande dump med \n i private_key – försök laga
            try:
                d2 = json.loads(raw.replace("\\n", "\n"))
                data = d2
            except Exception:
                pass

    # Fallback: leta toppnivå-nycklar
    if not data:
        ce = _from_secrets("client_email")
        pk = _from_secrets("private_key")
        if ce and pk:
            data = {"client_email": ce, "private_key": pk, "token_uri": "https://oauth2.googleapis.com/token"}

    if not data or "client_email" not in data or "private_key" not in data:
        raise RuntimeError("Hittade inga service account-uppgifter (minst client_email + private_key).")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    # Laga private_key om den är på en rad
    if "\\n" in data.get("private_key", ""):
        data["private_key"] = data["private_key"].replace("\\n", "\n")

    return Credentials.from_service_account_info(data, scopes=scopes)


def _get_spreadsheet_id_or_url() -> Dict[str, str]:
    """
    Returnerar {"type":"id"|"url", "value": "..."}.
    Stöd:
    - st.secrets["SHEET_URL"] eller ["SPREADSHEET_ID"]
    - st.secrets["GOOGLE_SHEETS"]["SHEET_URL"] | ["SPREADSHEET_ID"]
    """
    url = (
        _from_secrets("SHEET_URL")
        or _from_secrets("GOOGLE_SHEETS", "SHEET_URL")
    )
    sid = (
        _from_secrets("SPREADSHEET_ID")
        or _from_secrets("GOOGLE_SHEETS", "SPREADSHEET_ID")
    )
    if url:
        return {"type": "url", "value": url}
    if sid:
        return {"type": "id", "value": sid}
    raise RuntimeError("Hittade inget Spreadsheet-ID/URL i secrets.")


# ---------------- gspread-klient & Spreadsheet ----------------
def _client() -> gspread.Client:
    creds = _load_credentials()
    return gspread.authorize(creds)


def get_spreadsheet() -> gspread.Spreadsheet:
    cli = _client()
    ref = _get_spreadsheet_id_or_url()
    if ref["type"] == "url":
        return _with_backoff(cli.open_by_url, ref["value"])
    return _with_backoff(cli.open_by_key, ref["value"])


def list_worksheet_titles() -> List[str]:
    try:
        ss = get_spreadsheet()
        return [ws.title for ws in _with_backoff(ss.worksheets)]
    except Exception:
        return []


# ---------------- Läs/skriv DataFrame ----------------
def _ensure_ws(ss: gspread.Spreadsheet, title: str) -> gspread.Worksheet:
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        _with_backoff(ss.add_worksheet, title=title, rows=2000, cols=50)
        return _with_backoff(ss.worksheet, title)


def ws_read_df(title: str) -> pd.DataFrame:
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, title)
    except Exception:
        # Finns inte – returnera tom df
        return pd.DataFrame()

    values: List[List[str]] = _with_backoff(ws.get_all_values)
    if not values:
        return pd.DataFrame()
    # Förväntar första raden = headers
    headers = values[0]
    rows = values[1:] if len(values) > 1 else []
    df = pd.DataFrame(rows, columns=headers)
    return df


def ws_write_df(title: str, df: pd.DataFrame):
    ss = get_spreadsheet()
    ws = _ensure_ws(ss, title)

    # Bygg body = [headers] + rows (som strängar)
    headers = list(map(str, df.columns.tolist()))
    rows = [[str(x) if x is not None else "" for x in row] for row in df.astype(object).values.tolist()]
    body = [headers] + rows if headers else rows

    # Clear + update
    _with_backoff(ws.clear)
    if body:
        _with_backoff(ws.update, "A1", body)
