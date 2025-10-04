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


def _secrets_as_dict() -> Dict[str, Any]:
    try:
        if hasattr(st.secrets, "to_dict"):
            return st.secrets.to_dict()  # type: ignore[attr-defined]
        return dict(st.secrets.items())
    except Exception:
        try:
            return {k: st.secrets[k] for k in st.secrets.keys()}
        except Exception:
            return {}


def _maybe_json_to_dict(val: Any) -> Dict[str, Any] | None:
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = json.loads(s.replace("\\n", "\n"))
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _find_first_sa(obj: Any) -> Dict[str, Any] | None:
    if isinstance(obj, Mapping):
        if "client_email" in obj and "private_key" in obj:
            return dict(obj)
        if len(obj) == 1:
            only_val = next(iter(obj.values()))
            j = _maybe_json_to_dict(only_val)
            if isinstance(j, dict) and "client_email" in j and "private_key" in j:
                return j
        for v in obj.values():
            if isinstance(v, str):
                j = _maybe_json_to_dict(v)
                if isinstance(j, dict) and "client_email" in j and "private_key" in j:
                    return j
            found = _find_first_sa(v)
            if found:
                return found
        return None
    if isinstance(obj, str):
        j = _maybe_json_to_dict(obj)
        if isinstance(j, dict) and "client_email" in j and "private_key" in j:
            return j
    return None


def _find_sheet_ref(obj: Any) -> Dict[str, str] | None:
    if isinstance(obj, Mapping):
        if "SHEET_URL" in obj and isinstance(obj["SHEET_URL"], str) and obj["SHEET_URL"].strip():
            return {"type": "url", "value": obj["SHEET_URL"].strip()}
        if "SPREADSHEET_ID" in obj and isinstance(obj["SPREADSHEET_ID"], str) and obj["SPREADSHEET_ID"].strip():
            return {"type": "id", "value": obj["SPREADSHEET_ID"].strip()}
        for v in obj.values():
            r = _find_sheet_ref(v)
            if r:
                return r
    return None


def _load_credentials() -> Credentials:
    secrets_dict = _secrets_as_dict()
    sa = _find_first_sa(secrets_dict)
    if not sa:
        ce = secrets_dict.get("client_email")
        pk = secrets_dict.get("private_key")
        if ce and pk:
            sa = {"client_email": ce, "private_key": pk, "token_uri": "https://oauth2.googleapis.com/token"}
    if not sa or "client_email" not in sa or "private_key" not in sa:
        keys_preview = list(secrets_dict.keys())
        raise RuntimeError(
            f"Hittade inga service account-uppgifter (minst client_email + private_key). "
            f"Skannade toppnycklar: {keys_preview}"
        )
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
        # VIKTIGT: skriv som RAW så Sheets inte tolkar 9.37 som 9 370 000 000 i svensk lokal
        _with_backoff(ws.update, "A1", body, value_input_option="RAW")
