# stockapp/sheets.py
"""
Robust Google Sheets–IO (gspread) med tolerant secret-upplock.
Exponerar: ws_read_df, ws_write_df, list_worksheet_titles, get_spreadsheet.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# ---------------- Backoff ----------------
def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last


# ---------------- Utilities ----------------
def _secrets_as_dict() -> Dict[str, Any]:
    try:
        # Streamlit's AttrDict har .to_dict() i nya versioner
        if hasattr(st.secrets, "to_dict"):
            return st.secrets.to_dict()  # type: ignore[attr-defined]
        # fallback – gör en vanlig dict av items()
        return dict(st.secrets.items())
    except Exception:
        # sista fallback: försök använda som mapping
        try:
            return {k: st.secrets[k] for k in st.secrets.keys()}
        except Exception:
            return {}


def _maybe_json_to_dict(val: Any) -> Dict[str, Any] | None:
    """Om val är JSON-sträng som innehåller SA, returnera dict, annars None."""
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    # hantera escapeade radbrytningar i private_key
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = json.loads(s.replace("\\n", "\n"))
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _find_first_sa(obj: Any) -> Dict[str, Any] | None:
    """
    Gå igenom hela secrets (rekursivt) och hitta första dict (eller JSON-sträng)
    som innehåller 'client_email' och 'private_key'.
    """
    if isinstance(obj, Mapping):
        # direct dict?
        if "client_email" in obj and "private_key" in obj:
            return dict(obj)
        # JSON-sträng kapslad?
        if len(obj) == 1:
            # ibland ligger SA som enda värde under en nyckel
            only_val = next(iter(obj.values()))
            j = _maybe_json_to_dict(only_val)
            if isinstance(j, dict) and "client_email" in j and "private_key" in j:
                return j

        # annars gå rekursivt
        for v in obj.values():
            # JSON in value?
            if isinstance(v, str):
                j = _maybe_json_to_dict(v)
                if isinstance(j, dict) and "client_email" in j and "private_key" in j:
                    return j
            found = _find_first_sa(v)
            if found:
                return found
        return None

    # str? prova JSON
    if isinstance(obj, str):
        j = _maybe_json_to_dict(obj)
        if isinstance(j, dict) and "client_email" in j and "private_key" in j:
            return j
    return None


def _find_sheet_ref(obj: Any) -> Dict[str, str] | None:
    """
    Leta upp SHEET_URL eller SPREADSHEET_ID någonstans i secrets (rekursivt).
    Returnerar {"type":"url"|"id","value": "..."}.
    """
    if isinstance(obj, Mapping):
        # direkta träffar
        if "SHEET_URL" in obj and isinstance(obj["SHEET_URL"], str) and obj["SHEET_URL"].strip():
            return {"type": "url", "value": obj["SHEET_URL"].strip()}
        if "SPREADSHEET_ID" in obj and isinstance(obj["SPREADSHEET_ID"], str) and obj["SPREADSHEET_ID"].strip():
            return {"type": "id", "value": obj["SPREADSHEET_ID"].strip()}

        # rekursivt
        for v in obj.values():
            r = _find_sheet_ref(v)
            if r:
                return r
    return None


# ---------------- Auth / Spreadsheet ----------------
def _load_credentials() -> Credentials:
    secrets_dict = _secrets_as_dict()
    sa = _find_first_sa(secrets_dict)
    if not sa:
        # extra fallback: toppnivåfält client_email/private_key
        ce = secrets_dict.get("client_email")
        pk = secrets_dict.get("private_key")
        if ce and pk:
            sa = {"client_email": ce, "private_key": pk, "token_uri": "https://oauth2.googleapis.com/token"}

    if not sa or "client_email" not in sa or "private_key" not in sa:
        # diagnostic – visa vilka toppnycklar vi såg (inte värden)
        keys_preview = list(secrets_dict.keys())
        raise RuntimeError(
            f"Hittade inga service account-uppgifter (minst client_email + private_key). "
            f"Skannade toppnycklar: {keys_preview}"
        )

    # fixa privata nyckeln om den är på en rad
    if "\\n" in sa.get("private_key", ""):
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    return Credentials.from_service_account_info(sa, scopes=scopes)


def _client() -> gspread.Client:
    return gspread.authorize(_load_credentials())


def get_spreadsheet() -> gspread.Spreadsheet:
    cli = _client()
    secrets_dict = _secrets_as_dict()
    ref = _find_sheet_ref(secrets_dict)
    if not ref:
        raise RuntimeError("Hittade inget Spreadsheet-ID/URL i secrets (SHEET_URL eller SPREADSHEET_ID).")
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
        # Bladet finns inte – returnera tom df
        return pd.DataFrame()

    values: List[List[str]] = _with_backoff(ws.get_all_values)
    if not values:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:] if len(values) > 1 else []
    return pd.DataFrame(rows, columns=headers)


def ws_write_df(title: str, df: pd.DataFrame):
    ss = get_spreadsheet()
    ws = _ensure_ws(ss, title)

    headers = list(map(str, df.columns.tolist()))
    rows = [[str(x) if x is not None else "" for x in row] for row in df.astype(object).values.tolist()]
    body = [headers] + rows if headers else rows

    _with_backoff(ws.clear)
    if body:
        _with_backoff(ws.update, "A1", body)
