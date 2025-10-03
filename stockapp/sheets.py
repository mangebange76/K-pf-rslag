# stockapp/sheets.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

# Försök importera gspread – om det saknas kör vi lokalläge
try:
    import gspread  # type: ignore
except Exception:
    gspread = None  # type: ignore


# ---------- Lokalt "worksheet" som fallback ----------
@dataclass
class _LocalWS:
    path: str
    title: str = "Local Data (fallback)"


_FALLBACK_PATH = "/mnt/data/local_sheet.csv"
_WARNED = {"fallback": False}


# ---------- Google Sheets helpers ----------
def _load_service_account_dict() -> Optional[dict]:
    """Försöker läsa SA från secrets. Returnerar None om inte möjligt."""
    try:
        creds = st.secrets.get("GOOGLE_CREDENTIALS")
        if creds is None:
            return None
        if isinstance(creds, dict):
            sa = dict(creds)
        elif isinstance(creds, str):
            sa = json.loads(creds)
        else:
            return None
        # Normalisera private_key
        if "private_key" in sa and isinstance(sa["private_key"], str):
            sa["private_key"] = sa["private_key"].replace("\\n", "\n")
        if not sa.get("client_email") or not sa.get("private_key"):
            return None
        return sa
    except Exception:
        return None


def _get_sheet_ids(spreadsheet_id: str | None = None, worksheet_name: str | None = None):
    sid = spreadsheet_id or st.secrets.get("SPREADSHEET_ID")
    if not sid:
        return None, None
    wsn = worksheet_name or st.secrets.get("WORKSHEET_NAME") or "Data"
    return sid, wsn


def _open_remote_ws(spreadsheet_id: Optional[str], worksheet_name: Optional[str]):
    """Öppna Google Sheet. Returnerar Worksheet eller None."""
    if gspread is None:
        return None
    sa = _load_service_account_dict()
    if not sa:
        return None
    if not spreadsheet_id:
        return None
    try:
        gc = gspread.service_account_from_dict(sa)
        sh = gc.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(worksheet_name or "Data")
        except gspread.WorksheetNotFound:  # type: ignore
            ws = sh.add_worksheet(title=(worksheet_name or "Data"), rows="100", cols="26")
        return ws
    except Exception:
        return None


# ---------- Publikt API ----------
def get_ws(spreadsheet_id: str | None = None, worksheet_name: str | None = None):
    sid, wsn = _get_sheet_ids(spreadsheet_id, worksheet_name)
    ws = _open_remote_ws(sid, wsn)
    if ws is not None:
        return ws
    # Fallback: lokalt "worksheet"
    if not _WARNED["fallback"]:
        st.warning("⚠️ Google Sheets otillgängligt – använder lokal CSV som fallback (/mnt/data/local_sheet.csv).")
        _WARNED["fallback"] = True
    return _LocalWS(_FALLBACK_PATH, "Local Data (fallback)")


def ws_read_df(ws) -> pd.DataFrame:
    # Remote
    if gspread is not None and not isinstance(ws, _LocalWS):
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        header, rows = values[0], values[1:]
        df = pd.DataFrame(rows, columns=header)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
            if df[c].dtype == object:
                df.loc[df[c] == "", c] = pd.NA
        return df
    # Lokal CSV
    if os.path.exists(ws.path):
        try:
            return pd.read_csv(ws.path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def ws_write_df(ws, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ws_write_df: df måste vara en pandas.DataFrame")

    # Remote
    if gspread is not None and not isinstance(ws, _LocalWS):
        header = list(map(str, df.columns.tolist()))
        data = df.astype(object).where(pd.notnull(df), None).values.tolist()
        ws.clear()
        ws.resize(rows=max(2, len(data) + 1), cols=max(1, len(header)))
        ws.update("A1", [header] + data, value_input_option="USER_ENTERED")
        return

    # Lokal CSV
    os.makedirs(os.path.dirname(ws.path), exist_ok=True)
    df.to_csv(ws.path, index=False)


def save_dataframe(df: pd.DataFrame, spreadsheet_id: str | None = None, worksheet_name: str | None = None) -> None:
    ws = get_ws(spreadsheet_id=spreadsheet_id, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
