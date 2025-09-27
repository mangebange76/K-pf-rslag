# -*- coding: utf-8 -*-
import time
import pandas as pd
import streamlit as st
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

from .config import FINAL_COLS
from .utils import ensure_columns

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

@st.cache_resource(show_spinner=False)
def _client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    cred = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(cred)

@st.cache_resource(show_spinner=False)
def _spreadsheet():
    return _client().open_by_url(SHEET_URL)

def _worksheet(name=SHEET_NAME):
    ss = _spreadsheet()
    delays = [0, 0.5, 1.0]
    last_err = None
    for d in delays:
        if d: time.sleep(d)
        try:
            return ss.worksheet(name)
        except APIError as e:
            last_err = e
            if "NOT_FOUND" in str(e).upper():
                try:
                    ss.add_worksheet(title=name, rows=1000, cols=50)
                    return ss.worksheet(name)
                except Exception as ee:
                    last_err = ee
            continue
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Kunde inte öppna arbetsbladet.")

def load_df() -> pd.DataFrame:
    try:
        ws = _worksheet()
        rows = ws.get_all_records()
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame({c: [] for c in FINAL_COLS})
        return ensure_columns(df, FINAL_COLS)
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        return pd.DataFrame({c: [] for c in FINAL_COLS})

def save_df(df: pd.DataFrame):
    try:
        ws = _worksheet()
        body = [df.columns.tolist()] + df.astype(str).values.tolist()
        ws.update(body, "A1")
    except APIError as e:
        st.sidebar.error(f"⛔ Skrivfel (Sheets): {e}")
        st.session_state["pending_save_df"] = df.copy()
    except Exception as e:
        st.sidebar.error(f"⛔ Okänt skrivfel: {e}")
        st.session_state["pending_save_df"] = df.copy()

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_columns(df, FINAL_COLS)

# ---- Rates-ark ----

def rates_ws():
    ws = None
    try:
        ws = _worksheet(RATES_SHEET_NAME)
    except Exception:
        ss = _spreadsheet()
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = _worksheet(RATES_SHEET_NAME)
        ws.update([["Valuta","Kurs"]], "A1")
    return ws

def read_saved_rates() -> dict:
    try:
        ws = rates_ws()
        rows = ws.get_all_records()
        out = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = str(r.get("Kurs", "")).replace(",", ".").strip()
            try:
                out[cur] = float(val)
            except Exception:
                pass
        return out
    except Exception:
        return {}

def save_rates(rates: dict):
    try:
        ws = rates_ws()
        body = [["Valuta","Kurs"]]
        for k in ["USD","NOK","CAD","EUR","SEK"]:
            body.append([k, str(rates.get(k, ""))])
        ws.update(body, "A1")
    except Exception as e:
        st.sidebar.error(f"Kunde inte spara valutakurser: {e}")
