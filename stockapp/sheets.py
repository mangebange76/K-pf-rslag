# stockapp/sheets.py
from __future__ import annotations
import json, re, os, base64, ast
from typing import Any, Optional, Tuple, List
import pandas as pd
import streamlit as st
import gspread

_SHEET_URL_RE = re.compile(r"https?://docs\.google\.com/spreadsheets/d/([A-Za-z0-9\-_]+)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.iam\.gserviceaccount\.com")
_SA_MIN_KEYS = {"client_email", "private_key"}

# ── Utilities ──────────────────────────────────────────────────────────────
def _has_min_keys(d: Any) -> bool:
    return isinstance(d, dict) and _SA_MIN_KEYS.issubset({k.lower() for k in d.keys()})

def _flatten(obj: Any, path: str = "") -> List[tuple[str, Any]]:
    out: List[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            out.append((p, v)); out.extend(_flatten(v, p))
    return out

def _extract_sheet_id(x: Any) -> Optional[str]:
    if not isinstance(x, str): return None
    m = _SHEET_URL_RE.search(x)
    if m: return m.group(1)
    s = x.strip()
    return s if re.fullmatch(r"[A-Za-z0-9\-_]{25,}", s) else None

def _try_json(s: str) -> Optional[dict]:
    try: return json.loads(s)
    except Exception: return None

def _try_base64_json(s: str) -> Optional[dict]:
    try:
        dec = base64.b64decode(s.strip()).decode("utf-8", "ignore")
        return _try_json(dec) or _parse_kv(dec) or _try_ast_dict(dec)
    except Exception:
        return None

def _parse_kv(s: str) -> Optional[dict]:
    # key:value eller key=value per rad
    d: dict[str, Any] = {}
    for raw in s.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if ":" in line: k, v = line.split(":", 1)
        elif "=" in line: k, v = line.split("=", 1)
        else: continue
        k = k.strip().strip('"').strip("'")
        v = v.strip().strip('"').strip("'")
        d[k] = v
    if "private_key" in d: d["private_key"] = str(d["private_key"]).replace("\\n", "\n")
    return d if _has_min_keys(d) else None

def _try_ast_dict(s: str) -> Optional[dict]:
    # Hanterar Python-liknande dict-sträng (med 'enkla citat')
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict) and _has_min_keys(obj):
            obj["private_key"] = str(obj["private_key"]).replace("\\n", "\n")
            return obj
    except Exception:
        pass
    return None

def _parse_sa_from_any(val: Any) -> Optional[dict]:
    if _has_min_keys(val):
        sa = dict(val); sa["private_key"] = str(sa["private_key"]).replace("\\n", "\n"); return sa
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{"):
            j = _try_json(s) or _try_ast_dict(s)  # JSON eller Python-dict
            if _has_min_keys(j):
                j["private_key"] = str(j["private_key"]).replace("\\n", "\n"); return j
        j = _try_base64_json(s)
        if _has_min_keys(j): return j
        j = _parse_kv(s)
        if _has_min_keys(j): return j
        # sista chans: plocka bara ut mönster direkt i strängen
        email = None; m = _EMAIL_RE.search(s)
        if m: email = m.group(0)
        if email and ("BEGIN PRIVATE KEY" in s or "PRIVATE KEY-----" in s):
            return {"type":"service_account","client_email":email,"private_key":s}
    return None

# ── Service Account discovery ──────────────────────────────────────────────
def _get_service_account() -> dict:
    checked: List[str] = []

    # 1) Direktkandidater inkl. din GOOGLE_CREDENTIALS
    for k in [
        "GOOGLE_CREDENTIALS","google_credentials",
        "GOOGLE_SERVICE_ACCOUNT","google_service_account",
        "GCP_SERVICE_ACCOUNT","gcp_service_account",
        "SERVICE_ACCOUNT","service_account",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON","google_application_credentials_json",
        "gspread_service_account","gsheets_service_account",
        "GOOGLE_SA","google_sa",
    ]:
        if k in st.secrets:
            checked.append(k)
            sa = _parse_sa_from_any(st.secrets[k])
            if sa: return sa

    # 2) Gruppnycklar
    for g in ["GOOGLE_SHEETS","google_sheets","sheets","gsheets","google","connections"]:
        grp = st.secrets.get(g)
        if isinstance(grp, dict):
            checked.append(g)
            sa = _parse_sa_from_any(grp)
            if sa: return sa
            for subk, subv in grp.items():
                checked.append(f"{g}.{subk}")
                sa = _parse_sa_from_any(subv)
                if sa: return sa

    # 3) Full skanning
    for path, val in _flatten(st.secrets):
        checked.append(path)
        sa = _parse_sa_from_any(val)
        if sa: return sa

    # 4) Miljövariabler (om host lägger dem där)
    for envk in ["GOOGLE_APPLICATION_CREDENTIALS_JSON","GOOGLE_CREDENTIALS","GCP_SERVICE_ACCOUNT","SERVICE_ACCOUNT"]:
        v = os.environ.get(envk)
        if not v: continue
        sa = _parse_sa_from_any(v)
        if sa: return sa

    raise RuntimeError(f"Hittade inga service account-uppgifter (minst client_email + private_key). Skannade nycklar: {', '.join(sorted(set(checked)))}")

# ── Spreadsheet-ID / bladnamn (tolerant) ───────────────────────────────────
def _get_sheet_cfg(spreadsheet_id: Optional[str] = None,
                   worksheet_name: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if spreadsheet_id: return spreadsheet_id, worksheet_name

    for g in ["GOOGLE_SHEETS","google_sheets","sheets","gsheets","google","connections"]:
        grp = st.secrets.get(g)
        if isinstance(grp, dict):
            for key in ["SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id","SPREADSHEET_URL","spreadsheet_url","URL","url"]:
                if key in grp:
                    sid = _extract_sheet_id(str(grp[key]))
                    if sid:
                        wsn = (worksheet_name or grp.get("WORKSHEET_NAME") or grp.get("worksheet_name")
                               or grp.get("SHEET_NAME") or grp.get("sheet_name") or grp.get("TAB") or grp.get("tab"))
                        return sid, (str(wsn) if wsn else None)

    for key in ["SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id","SPREADSHEET_URL","spreadsheet_url","URL","url"]:
        if key in st.secrets:
            sid = _extract_sheet_id(str(st.secrets[key]))
            if sid:
                wsn = (worksheet_name or st.secrets.get("WORKSHEET_NAME") or st.secrets.get("worksheet_name")
                       or st.secrets.get("SHEET_NAME") or st.secrets.get("sheet_name") or st.secrets.get("tab"))
                return sid, (str(wsn) if wsn else None)

    for _, val in _flatten(st.secrets):
        sid = _extract_sheet_id(val if isinstance(val, str) else "")
        if sid: return sid, worksheet_name

    raise RuntimeError("Hittade inget Spreadsheet-ID/URL i secrets.")

# ── gspread client & publikt API ───────────────────────────────────────────
def _gc() -> gspread.Client:
    sa = _get_service_account()
    return gspread.service_account_from_dict(sa)

def get_ws(spreadsheet_id: Optional[str] = None, worksheet_name: Optional[str] = None) -> gspread.Worksheet:
    sid, wsn = _get_sheet_cfg(spreadsheet_id, worksheet_name)
    gc = _gc()
    sh = gc.open_by_key(sid)
    if wsn:
        try: return sh.worksheet(wsn)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=wsn, rows="100", cols="26")
    ws_list = sh.worksheets()
    return ws_list[0] if ws_list else sh.add_worksheet(title="Data", rows="100", cols="26")

def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values: return pd.DataFrame()
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
        if df[col].dtype == object: df.loc[df[col] == "", col] = pd.NA
    return df

def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ws_write_df: df måste vara pandas.DataFrame")
    header = list(map(str, df.columns.tolist()))
    data = df.astype(object).where(pd.notnull(df), None).values.tolist()
    ws.clear(); ws.resize(rows=max(2, len(data)+1), cols=max(1, len(header)))
    ws.update("A1", [header] + data, value_input_option="USER_ENTERED")

def save_dataframe(df: pd.DataFrame, spreadsheet_id: Optional[str] = None, worksheet_name: Optional[str] = None) -> None:
    ws = get_ws(spreadsheet_id=spreadsheet_id, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
