# stockapp/sheets.py
from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import gspread


# ── Robust secrets-detektering ─────────────────────────────────────────────

_REQUIRED_SA_KEYS = {"type", "private_key", "client_email"}
_SHEET_URL_RE = re.compile(r"https?://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)")

def _looks_like_sa_dict(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    lower = set(map(str.lower, d.keys()))
    return _REQUIRED_SA_KEYS.issubset(lower)

def _maybe_json(s: Any) -> Optional[dict]:
    if isinstance(s, str) and s.strip().startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

def _flatten(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            out.append((p, v))
            out.extend(_flatten(v, p))
    return out

def _get_service_account_from_secrets() -> dict:
    """
    Hitta service account på valfri nivå:
    - Vanliga namn (case-insensitive)
    - Under t.ex. GOOGLE_SHEETS
    - JSON-strängar
    - Sista utväg: hitta första dict som ser ut som SA i hela secrets
    """
    guess_keys = [
        "GOOGLE_SERVICE_ACCOUNT", "google_service_account",
        "gcp_service_account", "GCP_SERVICE_ACCOUNT",
        "service_account", "SERVICE_ACCOUNT",
        "gsheets_service_account", "gspread_service_account",
        "google_credentials", "GOOGLE_CREDENTIALS",
        "google_sa", "GOOGLE_SA",
    ]
    for k in guess_keys:
        if k in st.secrets:
            v = st.secrets[k]
            if _looks_like_sa_dict(v):
                return dict(v)
            parsed = _maybe_json(v)
            if parsed and _looks_like_sa_dict(parsed):
                return parsed

    # Gruppnycklar (t.ex. GOOGLE_SHEETS)
    for group in ("GOOGLE_SHEETS", "google_sheets", "sheets", "gsheets", "google"):
        grp = st.secrets.get(group)
        if isinstance(grp, dict):
            if _looks_like_sa_dict(grp):
                return dict(grp)
            for subk, subv in grp.items():
                if _looks_like_sa_dict(subv):
                    return dict(subv)
                parsed = _maybe_json(subv)
                if parsed and _looks_like_sa_dict(parsed):
                    return parsed

    # Skanna allt
    for path, val in _flatten(st.secrets):
        if _looks_like_sa_dict(val):
            return dict(val)
        parsed = _maybe_json(val)
        if parsed and _looks_like_sa_dict(parsed):
            return parsed

    raise RuntimeError("Hittade inga service account-uppgifter i st.secrets.")

def _extract_sheet_id_from_any(value: Any) -> Optional[str]:
    """Plocka Spreadsheet-ID från sträng (direkt ID eller full URL)."""
    if not isinstance(value, str):
        return None
    m = _SHEET_URL_RE.search(value)
    if m:
        return m.group(1)
    # Om strängen ser ut som ett ID (lååång a-zA-Z0-9-_):
    if re.fullmatch(r"[A-Za-z0-9\-_]{25,}", value.strip()):
        return value.strip()
    return None

def _get_sheet_cfg(spreadsheet_id: Optional[str] = None,
                   worksheet_name: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returnerar (sheet_id, worksheet_name, sheet_url)
    worksheet_name kan vara None -> vi tar första bladet.
    """
    # Direkt parametrar vinner
    if spreadsheet_id:
        return spreadsheet_id, worksheet_name, None

    # Vanliga gruppnycklar
    for group in ("GOOGLE_SHEETS", "google_sheets", "sheets", "gsheets"):
        grp = st.secrets.get(group)
        if isinstance(grp, dict):
            # ID
            for key in ("SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id","SPREADSHEET","sheet"):
                if key in grp:
                    sid = _extract_sheet_id_from_any(grp[key])
                    if sid:
                        wsn = worksheet_name or grp.get("WORKSHEET_NAME") or grp.get("worksheet_name") \
                              or grp.get("SHEET_NAME") or grp.get("sheet_name") or grp.get("TAB") or grp.get("tab")
                        url = grp.get("SPREADSHEET_URL") or grp.get("spreadsheet_url") or grp.get("url")
                        return sid, (str(wsn) if wsn else None), (str(url) if url else None)
            # URL utan ID
            for key in ("SPREADSHEET_URL","spreadsheet_url","URL","url"):
                if key in grp:
                    sid = _extract_sheet_id_from_any(grp[key])
                    if sid:
                        wsn = worksheet_name or grp.get("WORKSHEET_NAME") or grp.get("worksheet_name") \
                              or grp.get("SHEET_NAME") or grp.get("sheet_name") or grp.get("TAB") or grp.get("tab")
                        return sid, (str(wsn) if wsn else None), str(grp[key])

    # Top-level ID/URL
    for key in ("SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id"):
        if key in st.secrets:
            sid = _extract_sheet_id_from_any(st.secrets[key])
            if sid:
                wsn = worksheet_name or st.secrets.get("WORKSHEET_NAME") or st.secrets.get("worksheet_name") \
                      or st.secrets.get("SHEET_NAME") or st.secrets.get("sheet_name") or st.secrets.get("tab")
                url = st.secrets.get("SPREADSHEET_URL") or st.secrets.get("spreadsheet_url") or st.secrets.get("url")
                return sid, (str(wsn) if wsn else None), (str(url) if url else None)

    # Sista utväg: leta URL/ID i ALLA strängar i secrets
    for path, val in _flatten(st.secrets):
        sid = _extract_sheet_id_from_any(val)
        if sid:
            return sid, worksheet_name, None

    raise RuntimeError("Hittade inget Spreadsheet-ID eller URL i secrets.")

def _gc_client() -> gspread.Client:
    sa = _get_service_account_from_secrets()
    return gspread.service_account_from_dict(sa)


# ── Publika funktioner (kompatibla med storage.py) ─────────────────────────

def get_ws(spreadsheet_id: Optional[str] = None,
           worksheet_name: Optional[str] = None) -> gspread.Worksheet:
    """Öppna/returnera Worksheet. Tar första bladet om namn saknas."""
    sid, wsn, _ = _get_sheet_cfg(spreadsheet_id, worksheet_name)
    gc = _gc_client()
    sh = gc.open_by_key(sid)
    if wsn:
        try:
            return sh.worksheet(wsn)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=wsn, rows="100", cols="26")
    # inget wsn → ta första bladet
    ws_list = sh.worksheets()
    if ws_list:
        return ws_list[0]
    # om dokumentet är tomt: skapa "Data"
    return sh.add_worksheet(title="Data", rows="100", cols="26")

def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    """Läs hela bladet till DataFrame. Första raden = header (om finns)."""
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
        if df[col].dtype == object:
            df.loc[df[col] == "", col] = pd.NA
    return df

def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    """Skriv DataFrame till bladet (ersätter allt)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ws_write_df: df måste vara pandas.DataFrame")
    header = list(map(str, df.columns.tolist()))
    data = df.astype(object).where(pd.notnull(df), None).values.tolist()
    values = [header] + data
    ws.clear()
    ws.resize(rows=max(2, len(values)), cols=max(1, len(header)))
    ws.update("A1", values, value_input_option="USER_ENTERED")

def save_dataframe(df: pd.DataFrame,
                   spreadsheet_id: Optional[str] = None,
                   worksheet_name: Optional[str] = None) -> None:
    """Bekväm wrapper som öppnar bladet via secrets (eller parametrar) och skriver hela df."""
    ws = get_ws(spreadsheet_id=spreadsheet_id, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
