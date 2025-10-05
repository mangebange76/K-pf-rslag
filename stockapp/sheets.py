from __future__ import annotations

import json
import time
from typing import Any, Callable, List, Optional
import pandas as pd
import streamlit as st

# Googles klienter
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None  # type: ignore

# ---- Backoff ----
def with_backoff(func: Callable, *args, **kwargs):
    delays = [0, 0.6, 1.2, 2.5]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last  # type: ignore

# ---- Credentials ----
def _load_credentials_dict() -> dict:
    src = st.secrets.get("GOOGLE_CREDENTIALS")
    if isinstance(src, dict):
        return src
    if isinstance(src, str):
        s = src.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
    # alternativ: individuella fält
    ce = st.secrets.get("client_email")
    pk = st.secrets.get("private_key")
    if ce and pk:
        return {
            "type": "service_account",
            "client_email": ce,
            "private_key": pk.replace("\\n", "\n"),
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas.")

def _client():
    if gspread is None or Credentials is None:
        raise RuntimeError("gspread saknas i miljön.")
    cred_dict = _load_credentials_dict()
    creds = Credentials.from_service_account_info(
        cred_dict,
        scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"],
    )
    return gspread.authorize(creds)

def _spreadsheet():
    cli = _client()
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    return with_backoff(cli.open_by_url, url)

# ---- Worksheets ----
def list_worksheet_titles() -> List[str]:
    ss = _spreadsheet()
    return [ws.title for ws in ss.worksheets()]

def _get_or_create_worksheet(title: str):
    ss = _spreadsheet()
    try:
        return ss.worksheet(title)
    except Exception:
        return with_backoff(ss.add_worksheet, title=title, rows=1000, cols=80)

def delete_worksheet(title: str):
    ss = _spreadsheet()
    try:
        ws = ss.worksheet(title)
    except Exception:
        return
    with_backoff(ss.del_worksheet, ws)

# ---- Read/Write DF ----
def ws_read_df(title: str) -> pd.DataFrame:
    ws = _get_or_create_worksheet(title)
    rows = with_backoff(ws.get_all_values)
    if not rows:
        return pd.DataFrame()
    head = rows[0]
    data = rows[1:] if len(rows) > 1 else []
    # säkerställ unika rubriker
    seen = {}
    cols = []
    for h in head:
        h2 = h or ""
        if h2 in seen:
            seen[h2] += 1
            cols.append(f"{h2}__dup{seen[h2]}")
        else:
            seen[h2] = 0
            cols.append(h2)
    df = pd.DataFrame(data, columns=cols)
    # trim trailing tomma rader
    df = df[~(df.apply(lambda r: "".join(map(str, r.values))).str.strip() == "")]
    return df

def ws_write_df(title: str, df: pd.DataFrame):
    ws = _get_or_create_worksheet(title)
    # Rensa + skriv
    with_backoff(ws.clear)
    values = [list(df.columns)]
    if not df.empty:
        values += df.astype(str).replace({None: ""}).values.tolist()
    with_backoff(ws.update, values)
