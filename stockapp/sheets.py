# stockapp/sheets.py
from __future__ import annotations
import json, re, os, base64, ast
from typing import Any, Optional, Tuple, List, Dict
import pandas as pd
import streamlit as st
import gspread

# ---- ID/URL mönster --------------------------------------------------------
_SHEET_URL_RE = re.compile(r"https?://docs\.google\.com/spreadsheets/d/([A-Za-z0-9\-_]+)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.iam\.gserviceaccount\.com")

# runtime-override (om användaren klistrar in JSON i appen)
_RUNTIME_SA_OVERRIDE: Optional[dict] = None

# ---- Små utils --------------------------------------------------------------
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

def _norm_pk(v: str) -> str:
    return v.replace("\\n", "\n")

def _try_json(s: str) -> Optional[dict]:
    try: return json.loads(s)
    except Exception: return None

def _try_ast_dict(s: str) -> Optional[dict]:
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _try_base64_json(s: str) -> Optional[dict]:
    try:
        dec = base64.b64decode(s.strip()).decode("utf-8", "ignore")
        return _try_json(dec) or _try_ast_dict(dec)
    except Exception:
        return None

def _parse_kv(s: str) -> Optional[dict]:
    d: Dict[str, Any] = {}
    for raw in s.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if ":" in line: k, v = line.split(":", 1)
        elif "=" in line: k, v = line.split("=", 1)
        else: continue
        d[k.strip().strip('"').strip("'")] = v.strip().strip('"').strip("'")
    return d or None

# ---- Hitta client_email / private_key även med konstiga nyckelnamn ----------
def _find_sa_in_dict(d: dict) -> Optional[dict]:
    # platta nycklar (case/tecken-oberoende)
    def find_key(candidates: List[str]) -> Optional[str]:
        norm_map = {re.sub(r"[^a-z]", "", k.lower()): k for k in d.keys()}
        for want in candidates:
            w = re.sub(r"[^a-z]", "", want.lower())
            if w in norm_map:
                return norm_map[w]
        # fuzzy: hitta första nyckel som innehåller alla ord
        for k in d.keys():
            nk = re.sub(r"[^a-z]", "", k.lower())
            if all(re.sub(r"[^a-z]", "", w) in nk for w in candidates):
                return k
        return None

    email_key = find_key(["clientemail"]) or find_key(["email","client"]) or find_key(["service","account","email"])
    pk_key    = find_key(["privatekey"]) or find_key(["private","key"])

    email = d.get(email_key) if email_key else None
    pk    = d.get(pk_key)    if pk_key else None

    # fallback: leta i strängar
    if not email:
        for v in d.values():
            if isinstance(v, str):
                m = _EMAIL_RE.search(v)
                if m: email = m.group(0); break
    if not pk and "private_key" in d:
        pk = d["private_key"]
    if not pk:
        for v in d.values():
            if isinstance(v, str) and ("BEGIN PRIVATE KEY" in v or "PRIVATE KEY-----" in v):
                pk = v; break

    if isinstance(email, str) and isinstance(pk, str):
        return {"type": "service_account", "client_email": email, "private_key": _norm_pk(pk)}
    return None

def _parse_sa_from_any(val: Any) -> Optional[dict]:
    # 1) direkt dict
    if isinstance(val, dict):
        # rekursivt sök i dict + barn
        hit = _find_sa_in_dict(val)
        if hit: return hit
        for _, v in _flatten(val):
            if isinstance(v, dict):
                hit = _find_sa_in_dict(v)
                if hit: return hit
        return None
    # 2) sträng
    if isinstance(val, str):
        s = val.strip()
        # JSON / Python-dict
        for parser in (_try_json, _try_ast_dict, _try_base64_json):
            j = parser(s)
            if isinstance(j, dict):
                hit = _parse_sa_from_any(j)
                if hit: return hit
        # key=value / key: value
        j = _parse_kv(s)
        if isinstance(j, dict):
            hit = _parse_sa_from_any(j)
            if hit: return hit
        # rena mönster direkt i strängen
        email = None; m = _EMAIL_RE.search(s)
        if m: email = m.group(0)
        if email and ("BEGIN PRIVATE KEY" in s or "PRIVATE KEY-----" in s):
            return {"type":"service_account","client_email":email,"private_key": s}
    return None

# ---- Publika hjälpare för runtime-override & diagnos -----------------------
def set_runtime_service_account(sa_json_or_dict: Any) -> None:
    """Anropa från appen om användaren klistrar in sin SA JSON. Tar dict eller str."""
    global _RUNTIME_SA_OVERRIDE
    if isinstance(sa_json_or_dict, dict):
        hit = _parse_sa_from_any(sa_json_or_dict)
    elif isinstance(sa_json_or_dict, str):
        hit = _parse_sa_from_any(sa_json_or_dict)
    else:
        hit = None
    if not hit:
        raise ValueError("Ogiltigt service account-format (kunde inte hitta client_email/private_key).")
    _RUNTIME_SA_OVERRIDE = hit

def secrets_diagnose() -> dict:
    """Returnerar *endast* nyckelnamn/typer för felsökning – inga hemligheter."""
    tree = {}
    def rec(obj: Any, name: str):
        if isinstance(obj, dict):
            tree[name] = sorted(list(obj.keys()))
            for k, v in obj.items():
                rec(v, f"{name}.{k}" if name else k)
        else:
            tree[name] = type(obj).__name__
    rec(dict(st.secrets), "")
    return tree

# ---- Core: hämta SA och Sheet-konfig --------------------------------------
def _get_service_account() -> dict:
    if _RUNTIME_SA_OVERRIDE:
        return dict(_RUNTIME_SA_OVERRIDE)

    checked: List[str] = []
    # testa kända toppnycklar (inkl. GOOGLE_CREDENTIALS)
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

    # gruppnycklar
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

    # full skanning
    for path, val in _flatten(st.secrets):
        checked.append(path)
        sa = _parse_sa_from_any(val)
        if sa: return sa

    # miljövariabler
    for envk in ["GOOGLE_APPLICATION_CREDENTIALS_JSON","GOOGLE_CREDENTIALS","GCP_SERVICE_ACCOUNT","SERVICE_ACCOUNT"]:
        v = os.environ.get(envk); 
        if not v: continue
        sa = _parse_sa_from_any(v)
        if sa: return sa

    raise RuntimeError(f"Hittade inga service account-uppgifter (minst client_email + private_key). Skannade nycklar: {', '.join(sorted(set(checked)))}")

def _get_sheet_cfg(spreadsheet_id: Optional[str] = None,
                   worksheet_name: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if spreadsheet_id:
        return spreadsheet_id, worksheet_name
    # grupper
    for g in ["GOOGLE_SHEETS","google_sheets","sheets","gsheets","google","connections"]:
        grp = st.secrets.get(g)
        if isinstance(grp, dict):
            for key in ["SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id",
                        "SPREADSHEET_URL","spreadsheet_url","URL","url"]:
                if key in grp:
                    sid = _extract_sheet_id(str(grp[key]))
                    if sid:
                        wsn = (worksheet_name or grp.get("WORKSHEET_NAME") or grp.get("worksheet_name")
                               or grp.get("SHEET_NAME") or grp.get("sheet_name") or grp.get("TAB") or grp.get("tab"))
                        return sid, (str(wsn) if wsn else None)
    # toppnivå
    for key in ["SPREADSHEET_ID","spreadsheet_id","SHEET_ID","sheet_id","ID","id",
                "SPREADSHEET_URL","spreadsheet_url","URL","url"]:
        if key in st.secrets:
            sid = _extract_sheet_id(str(st.secrets[key]))
            if sid:
                wsn = (worksheet_name or st.secrets.get("WORKSHEET_NAME") or st.secrets.get("worksheet_name")
                       or st.secrets.get("SHEET_NAME") or st.secrets.get("sheet_name") or st.secrets.get("tab"))
                return sid, (str(wsn) if wsn else None)
    # fri sökning
    for _, val in _flatten(st.secrets):
        sid = _extract_sheet_id(val if isinstance(val, str) else "")
        if sid:
            return sid, worksheet_name
    raise RuntimeError("Hittade inget Spreadsheet-ID/URL i secrets.")

# ---- gspread-klient & publikt API -----------------------------------------
def _gc() -> gspread.Client:
    sa = _get_service_account()
    return gspread.service_account_from_dict(sa)

def get_ws(spreadsheet_id: Optional[str] = None,
           worksheet_name: Optional[str] = None) -> gspread.Worksheet:
    sid, wsn = _get_sheet_cfg(spreadsheet_id, worksheet_name)
    gc = _gc()
    sh = gc.open_by_key(sid)
    if wsn:
        try: return sh.worksheet(wsn)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=wsn, rows="100", cols="26")
    wss = sh.worksheets()
    return wss[0] if wss else sh.add_worksheet(title="Data", rows="100", cols="26")

def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    vals = ws.get_all_values()
    if not vals: return pd.DataFrame()
    header, rows = vals[0], vals[1:]
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
        if df[c].dtype == object: df.loc[df[c] == "", c] = pd.NA
    return df

def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ws_write_df: df måste vara pandas.DataFrame")
    header = list(map(str, df.columns.tolist()))
    data = df.astype(object).where(pd.notnull(df), None).values.tolist()
    ws.clear(); ws.resize(rows=max(2, len(data)+1), cols=max(1, len(header)))
    ws.update("A1", [header] + data, value_input_option="USER_ENTERED")

def save_dataframe(df: pd.DataFrame,
                   spreadsheet_id: Optional[str] = None,
                   worksheet_name: Optional[str] = None) -> None:
    ws = get_ws(spreadsheet_id=spreadsheet_id, worksheet_name=worksheet_name)
    ws_write_df(ws, df)
