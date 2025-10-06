# stockapp/sheets.py
import json, time
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ---------- Backoff ----------
def with_backoff(fn, *args, **kwargs):
    delays = [0, 0.4, 0.8, 1.6, 3.0]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
    raise last

# ---------- Credentials ----------
def _load_credentials_dict() -> dict:
    key = "GOOGLE_CREDENTIALS"
    if key not in st.secrets:
        raise RuntimeError("GOOGLE_CREDENTIALS saknas i secrets.")
    val = st.secrets[key]
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas (JSON).")
    raise RuntimeError("GOOGLE_CREDENTIALS hade okänt format (varken dict eller JSON-sträng).")

def _client():
    cred_dict = _load_credentials_dict()
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(cred_dict, scopes=scope)
    return gspread.authorize(creds)

def get_spreadsheet():
    if "SHEET_URL" not in st.secrets:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    cli = _client()
    return with_backoff(cli.open_by_url, st.secrets["SHEET_URL"])

# ---------- Worksheet helpers ----------
def _get_or_create_worksheet(title: str):
    ss = get_spreadsheet()
    try:
        return with_backoff(ss.worksheet, title)
    except Exception:
        # skapa litet blad
        with_backoff(ss.add_worksheet, title=title, rows=100, cols=50)
        return with_backoff(ss.worksheet, title)

def list_worksheet_titles():
    ss = get_spreadsheet()
    return [w.title for w in with_backoff(ss.worksheets)]

def delete_worksheet(title: str):
    ss = get_spreadsheet()
    try:
        ws = with_backoff(ss.worksheet, title)
        with_backoff(ss.del_worksheet, ws)
    except Exception:
        pass

# ---------- Read/Write ----------
def ws_read_df(title: str) -> pd.DataFrame:
    ws = _get_or_create_worksheet(title)
    rows = with_backoff(ws.get_all_values)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    # Skydda mot DUP-kolumner
    seen = {}
    uniq = []
    for h in header:
        h = str(h or "").strip()
        if h in seen:
            seen[h] += 1
            uniq.append(f"{h}__{seen[h]}")
        else:
            seen[h] = 0
            uniq.append(h)
    df = pd.DataFrame(data, columns=uniq)
    # Ta bort helt tomma rader
    if "Ticker" in df.columns:
        mask = (df["Ticker"].astype(str).str.strip() == "") & (df.fillna("").replace("","0").astype(str).eq("0")).all(axis=1)
        df = df[~mask].copy()
    return df

def ws_write_df(title: str, df: pd.DataFrame):
    ws = _get_or_create_worksheet(title)
    out = df.copy()
    out = out.fillna("")
    values = [list(out.columns)]
    values += out.astype(str).values.tolist()
    with_backoff(ws.clear)
    with_backoff(ws.update, values)
