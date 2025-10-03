# stockapp/sheets.py
from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import gspread


# ───────────────────────── Helpers (robust secrets-detektering) ─────────────────────────

_REQUIRED_SA_KEYS = {"type", "private_key", "client_email"}

def _looks_like_sa_dict(d: Any) -> bool:
    return isinstance(d, dict) and _REQUIRED_SA_KEYS.issubset(set(map(str.lower, d.keys())))

def _try_parse_json(s: Any) -> Optional[dict]:
    if isinstance(s, str) and s.strip().startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

def _flatten_secrets(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    """Plattar ut hela st.secrets för att kunna leta efter SA på godtycklig nivå."""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            out.append((p, v))
            out.extend(_flatten_secrets(v, p))
    return out

def _get_service_account_from_secrets() -> dict:
    """
    Hitta service account på ett tolerant sätt:
    - Vanliga nycklar: GOOGLE_SERVICE_ACCOUNT / gcp_service_account / service_account / google_service_account ...
    - Vilken som helst dict i secrets som innehåller {type, private_key, client_email}
    - JSON-strängar som innehåller SA
    """
    # 1) Snabba vägar via vanliga nycklar (case-insensitive)
    candidate_keys = [
        "GOOGLE_SERVICE_ACCOUNT", "google_service_account",
        "gcp_service_account", "GCP_SERVICE_ACCOUNT",
        "service_account", "SERVICE_ACCOUNT",
        "gsheets_service_account", "gspread_service_account",
        "google_credentials", "GOOGLE_CREDENTIALS",
        "google_sa", "GOOGLE_SA",
    ]
    for k in candidate_keys:
        if k in st.secrets:
            v = st.secrets[k]
            if _looks_like_sa_dict(v):
                return dict(v)
            parsed = _try_parse_json(v)
            if parsed and _looks_like_sa_dict(parsed):
                return parsed

    # 2) Gruppnycklar (t.ex. GOOGLE_SHEETS: { service_account: {...} })
    for group_key in ("GOOGLE_SHEETS", "google_sheets", "sheets", "gsheets", "google"):
        grp = st.secrets.get(group_key, None)
        if isinstance(grp, dict):
            # direkt SA?
            if _looks_like_sa_dict(grp):
                return dict(grp)
            # barn med SA?
            for subk, subv in grp.items():
                if _looks_like_sa_dict(subv):
                    return dict(subv)
                parsed = _try_parse_json(subv)
                if parsed and _looks_like_sa_dict(parsed):
                    return parsed

    # 3) Skanna hela secrets-trädet efter en dict som ser ut som SA
    for path, val in _flatten_secrets(st.secrets):
        if _looks_like_sa_dict(val):
            return dict(val)
        parsed = _try_parse_json(val)
        if parsed and _looks_like_sa_dict(parsed):
            return parsed

    # Om vi hamnar här hittade vi inget SA i secrets
    raise RuntimeError(
        "Service account-uppgifter kunde inte hittas i st.secrets. "
        "Jag letade efter en dict (eller JSON-sträng) som innehåller nycklarna "
        f"{sorted(_REQUIRED_SA_KEYS)} på valfri nivå. "
        "Exempelnycklar som stöds: GOOGLE_SERVICE_ACCOUNT, gcp_service_account, service_account, "
        "GOOGLE_SHEETS.service_account, m.fl."
    )

def _get_sheet_cfg(spreadsheet_id: Optional[str] = None,
                   worksheet_name: Optional[str] = None) -> Tuple[str, str]:
    """Tolerant läsning av Spreadsheet-ID och bladnamn."""
    if spreadsheet_id and worksheet_name:
        return spreadsheet_id, worksheet_name

    # Först: gruppnyckel
    for group_key in ("GOOGLE_SHEETS", "google_sheets", "sheets", "gsheets"):
        grp = st.secrets.get(group_key, None)
        if isinstance(grp, dict):
            # leta efter något som liknar id
            sid = spreadsheet_id or grp.get("SPREADSHEET_ID") or grp.get("spreadsheet_id") \
                  or grp.get("SHEET_ID") or grp.get("sheet_id") or grp.get("id")
            wsn = worksheet_name or grp.get("WORKSHEET_NAME") or grp.get("worksheet_name") \
                  or grp.get("SHEET_NAME") or grp.get("sheet_name") or grp.get("tab") or "Data"
            if sid:
                return str(sid), str(wsn)

    # Top-level varianter
    sid_candidates = [
        "SPREADSHEET_ID", "spreadsheet_id",
        "GOOGLE_SHEET_ID", "google_sheet_id",
        "SHEET_ID", "sheet_id", "id",
    ]
    for key in sid_candidates:
        sid = st.secrets.get(key, None)
        if sid:
            wsn = worksheet_name or st.secrets.get("WORKSHEET_NAME") or st.secrets.get("worksheet_name") \
                  or st.secrets.get("SHEET_NAME") or st.secrets.get("sheet_name") or "Data"
            return str(sid), str(wsn)

    raise RuntimeError(
        "Hittade inget Spreadsheet-ID i secrets. "
        "Stödjer t.ex. GOOGLE_SHEETS.SPREADSHEET_ID eller top-level SPREADSHEET_ID."
    )

def _gc_client() -> gspread.Client:
    sa = _get_service_account_from_secrets()
    return gspread.service_account_from_dict(sa)


# ───────────────────────── Offentliga funktioner (kompatibla) ─────────────────────────

def get_ws(spreadsheet_id: Optional[str] = None,
           worksheet_name: Optional[str] = None) -> gspread.Worksheet:
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
    """Läs hela bladet till DataFrame. Första raden antas vara header."""
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)
    # försök konvertera numeriska strängar → tal
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
    """Bekväm wrapper som öppnar bladet via secrets och skriver hela df."""
    ws = get_ws(spreadsheet_id=spreadsheet_id, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
