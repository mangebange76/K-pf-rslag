# stockapp/sheets.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import gspread

# ---------- Interna helpers ----------
def _get_sa_dict() -> dict:
    # Leta efter vanligt förekommande nycklar i st.secrets
    for key in ("GOOGLE_SERVICE_ACCOUNT", "gcp_service_account", "service_account"):
        if key in st.secrets and isinstance(st.secrets[key], dict):
            return dict(st.secrets[key])
    raise RuntimeError("Service account-uppgifter saknas i st.secrets.")

def _get_sheet_cfg(spreadsheet_id: str | None = None,
                   worksheet_name: str | None = None) -> tuple[str, str]:
    if spreadsheet_id and worksheet_name:
        return spreadsheet_id, worksheet_name

    # Gruppnyckel
    if "GOOGLE_SHEETS" in st.secrets:
        cfg = st.secrets["GOOGLE_SHEETS"]
        sid = spreadsheet_id or cfg.get("SPREADSHEET_ID") or cfg.get("spreadsheet_id")
        wsn = worksheet_name or cfg.get("WORKSHEET_NAME") or "Data"
        if sid:
            return str(sid), str(wsn)

    # Platta nycklar
    sid = spreadsheet_id or st.secrets.get("SPREADSHEET_ID") or st.secrets.get("spreadsheet_id")
    wsn = worksheet_name or st.secrets.get("WORKSHEET_NAME") or "Data"
    if not sid:
        raise RuntimeError("SPREADSHEET_ID saknas i st.secrets.")
    return str(sid), str(wsn)

def _gc_client() -> gspread.Client:
    sa = _get_sa_dict()
    return gspread.service_account_from_dict(sa)

# ---------- Publika helpers (för kompatibilitet med storage.py) ----------
def get_ws(spreadsheet_id: str | None = None,
           worksheet_name: str | None = None) -> gspread.Worksheet:
    """Öppna/returnera Worksheet enligt secrets eller givna parametrar."""
    sid, wsn = _get_sheet_cfg(spreadsheet_id, worksheet_name)
    gc = _gc_client()
    sh = gc.open_by_key(sid)
    try:
        ws = sh.worksheet(wsn)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=wsn, rows="100", cols="26")
    return ws

def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    """Läs hela bladet till DataFrame. Första raden = header."""
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)
    # Försök numerifiera kolumner där det går
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
        # tomma strängar -> NaN
        if df[col].dtype == object:
            df.loc[df[col] == "", col] = pd.NA
    return df

def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    """Skriv DataFrame till bladet (ersätter allt)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ws_write_df: df måste vara pandas.DataFrame")

    header = list(map(str, df.columns.tolist()))
    # Byt NaN/None mot None så Google Sheets tar emot tomt
    data = df.astype(object).where(pd.notnull(df), None).values.tolist()
    values = [header] + data

    ws.clear()
    # Anpassa storlek ungefärligt
    ws.resize(rows=max(2, len(values)), cols=max(1, len(header)))
    # Skriv från A1
    ws.update("A1", values, value_input_option="USER_ENTERED")

# ---------- Bekväm wrapper ----------
def save_dataframe(df: pd.DataFrame,
                   spreadsheet_id: str | None = None,
                   worksheet_name: str | None = None) -> None:
    """Skriv hela df till Google Sheets utifrån secrets (eller parametrar)."""
    ws = get_ws(spreadsheet_id=spreadsheet_id, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
