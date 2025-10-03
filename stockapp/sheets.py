# stockapp/sheets.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe

def _get_sa_dict() -> dict:
    # Försök flera vanliga nyckelnamn i st.secrets
    for key in ("GOOGLE_SERVICE_ACCOUNT", "gcp_service_account", "service_account"):
        if key in st.secrets and isinstance(st.secrets[key], dict):
            return dict(st.secrets[key])
    raise RuntimeError("Service account-uppgifter saknas i st.secrets.")

def _get_sheet_cfg() -> tuple[str, str]:
    # Läs Spreadsheet-ID och bladnamn
    if "GOOGLE_SHEETS" in st.secrets:
        cfg = st.secrets["GOOGLE_SHEETS"]
        sid = cfg.get("SPREADSHEET_ID") or cfg.get("spreadsheet_id")
        wsn = cfg.get("WORKSHEET_NAME", "Data")
        if sid:
            return sid, wsn
    # fallback: platta nycklar
    sid = st.secrets.get("SPREADSHEET_ID") or st.secrets.get("spreadsheet_id")
    wsn = st.secrets.get("WORKSHEET_NAME", "Data")
    if not sid:
        raise RuntimeError("SPREADSHEET_ID saknas i st.secrets.")
    return str(sid), str(wsn)

def save_dataframe(df: pd.DataFrame) -> None:
    """Skriv hela DF till Google Sheets (ersätter bladet)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("save_dataframe: df måste vara en pandas.DataFrame")

    sa = _get_sa_dict()
    sid, wsn = _get_sheet_cfg()

    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(sid)
    try:
        ws = sh.worksheet(wsn)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=wsn, rows="100", cols="26")

    # Rensa och skriv in på nytt
    ws.clear()
    set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)
