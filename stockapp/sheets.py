# stockapp/sheets.py
from __future__ import annotations
import json, re, os
from typing import Any, Optional, Tuple
import pandas as pd
import streamlit as st
import gspread

# ── Mönster för att hitta Sheet-ID i URL eller “rått” ID ───────────────────
_SHEET_URL_RE = re.compile(r"https?://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9\-_]+)")

# Nycklar som ett Google service account typiskt har
_SA_KEYS = {"type", "private_key", "client_email"}

def _looks_like_sa(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    lower = {k.lower() for k in d.keys()}
    return _SA_KEYS.issubset(lower)

def _try_parse_json(val: Any) -> Optional[dict]:
    if isinstance(val, str) and val.strip().startswith("{"):
        try:
            return json.loads(val)
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

# ── Hämta SERVICE ACCOUNT (robust: secrets → grupper → allt → env) ─────────
def _get_service_account() -> dict:
    # 1) Vanliga nycklar (case-insensitive)
    candidates = [
        "GOOGLE_SERVICE_ACCOUNT","google_service_account",
        "gcp_service_account","GCP_SERVICE_ACCOUNT",
        "service_account","SERVICE_ACCOUNT",
        "google_credentials","GOOGLE_CREDENTIALS",
        "gspread_service_account","gsheets_service_account",
        "google_sa","GOOGLE_SA",
    ]
    for k in candidates:
        if k in st.secrets:
            v = st.secrets[k]
            if _looks_like_sa(v):  # dict
                return dict(v)
            parsed = _try_parse_json(v)  # JSON-sträng?
            if parsed and _looks_like_sa(parsed):
                return parsed

    # 2) Gruppnycklar (t.ex. GOOGLE_SHEETS: { service_account: {...} })
    groups = ["GOOGLE_SHEETS","google_sheets","sheets","gsheets","google"]
    for g in groups:
        grp = st.secrets.get(g)
        if isinstance(grp, dict):
            if _looks_like_sa(grp):
                return dict(grp)
            for _, subv in grp.items():
                if _looks_like_sa(subv):
                    return dict(subv)
                parsed = _try_parse_json(subv)
                if parsed and _looks_like_sa(parsed):
                    return parsed

    # 3) Skanna hela secrets-trädet
    for _, val in _flatten(st.secrets):
        if _looks_like_sa(val):
            return dict(val)
        parsed = _try_parse_json(val)
        if parsed and _looks_like_sa(parsed):
            return parsed

    # 4) Sista utväg: miljövariabler (om någon CI satt dem där)
    for k, v in os.environ.items():
        parsed = _try_parse_json(v)
        if parsed and _looks_like_sa(parsed):
            return parsed

    raise RuntimeError("Hittade inga service account-uppgifter (client_email/private_key) i secrets eller miljön.")

# ── Hämta SPREADSHEET-ID & BLADNAMN (tolerant) ─────────────────────────────
def _extract_sheet_id(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    m = _SHEET_URL_RE.search(x)
    if m:
        return m.group(1)
    # ser ut som ett ID?
    s = x.strip()
    if re.fullmatch(r"[A-Za-z0-9\-_]{25,}", s):
        return s
    return None

def _get_sheet_cfg(spreadsheet_id: Optional[str] = None, worksheet_name: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if spreadsheet_id:
        return spreadsheet_id, worksheet_name

    # Gruppnycklar
    for g in ["GOOGLE_SHEETS","google_sheets","sheets","gsheets"]:
        grp = st.secrets.get(g)
        if isinstance(grp, dict):
            # ID/URL
            for key in ["SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id","SPREADSHEET_URL","spreadsheet_url","URL","url"]:
                if key in grp:
                    sid = _extract_sheet_id(grp[key])
                    if sid:
                        wsn = worksheet_name or grp.get("WORKSHEET_NAME") or grp.get("worksheet_name") \
                              or grp.get("SHEET_NAME") or grp.get("sheet_name") or grp.get("TAB") or grp.get("tab")
                        return sid, (str(wsn) if wsn else None)

    # Top-level ID/URL
    for key in ["SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id","SPREADSHEET_URL","spreadsheet_url","URL","url"]:
        if key in st.secrets:
            sid = _extract_sheet_id(st.secrets[key])
            if sid:
                wsn = worksheet_name or st.secrets.get("WORKSHEET_NAME") or st.secrets.get("worksheet_name") \
                      or st.secrets.get("SHEET_NAME") or st.secrets.get("sheet_name") or st.secrets.get("tab")
                return sid, (str(wsn) if wsn else None)

    # Sista utväg: leta ID/URL i alla strängar i secrets
    for _, val in _flatten(st.secrets):
        sid = _extract_sheet_id(val)
        if sid:
            return sid, worksheet_name  # tar ev. första bladet senare

    raise RuntimeError("Hittade inget Spreadsheet-ID/URL i secrets.")

# ── gspread klient ─────────────────────────────────────────────────────────
def _gc() -> gspread.Client:
    sa = _get_service_account()
    return gspread.service_account_from_dict(sa)

# ── Publikt API (kompatibelt med storage.py) ───────────────────────────────
def get_ws(spreadsheet_id: Optional[str] = None, worksheet_name: Optional[str] = None) -> gspread.Worksheet:
    sid, wsn = _get_sheet_cfg(spreadsheet_id, worksheet_name)
    gc = _gc()
    sh = gc.open_by_key(sid)
    if wsn:
        try:
            return sh.worksheet(wsn)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=wsn, rows="100", cols="26")
    # inget namn – använd första bladet eller skapa “Data”
    wss = sh.worksheets()
    return wss[0] if wss else sh.add_worksheet(title="Data", rows="100", cols="26")

def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ws_write_df: df måste vara en pandas.DataFrame")
    header = list(map(str, df.columns.tolist()))
    data = df.astype(object).where(pd.notnull(df), None).values.tolist()
    values = [header] + data
    ws.clear()
    ws.resize(rows=max(2, len(values)), cols=max(1, len(header)))
    ws.update("A1", values, value_input_option="USER_ENTERED")

def save_dataframe(df: pd.DataFrame, spreadsheet_id: Optional[str] = None, worksheet_name: Optional[str] = None) -> None:
    ws = get_ws(spreadsheet_id=spreadsheet_id, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
