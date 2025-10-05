# stockapp/sheets.py
from __future__ import annotations

import base64
import json
import time
from typing import List, Tuple

import pandas as pd
import streamlit as st

# Ny: acceptera dict-lika objekt (AttrDict/Mapping)
from collections.abc import Mapping

# 3p
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None  # type: ignore
    Credentials = None  # type: ignore


# ---------------- Backoff ----------------
def _with_backoff(func, *args, **kwargs):
    delays = [0.0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Okänt fel vid Google Sheets-anrop.")


# ---------------- Credential helpers ----------------
def _json_from_str(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception as e:
        raise RuntimeError(f"Kunde inte JSON-tolka credentials-sträng ({e}).")


def _json_from_b64(s: str) -> dict:
    try:
        raw = base64.b64decode(s)
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Kunde inte BASE64-avkoda credentials ({e}).")


def _fix_private_key_newlines(data: dict) -> dict:
    pk = data.get("private_key")
    if isinstance(pk, str):
        data["private_key"] = pk.replace("\\n", "\n")
    return data


def _build_minimal_dict_from_separate_keys() -> dict | None:
    email = st.secrets.get("SHEETS_CLIENT_EMAIL")
    pkey  = st.secrets.get("SHEETS_PRIVATE_KEY")
    if not email or not pkey:
        return None
    return _fix_private_key_newlines({
        "type": "service_account",
        "project_id": st.secrets.get("SHEETS_PROJECT_ID", ""),
        "private_key_id": st.secrets.get("SHEETS_PRIVATE_KEY_ID", ""),
        "private_key": pkey,
        "client_email": email,
        "client_id": st.secrets.get("SHEETS_CLIENT_ID", ""),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": st.secrets.get("SHEETS_CLIENT_X509_CERT_URL", ""),
    })


def _load_credentials_dict() -> Tuple[dict, str]:
    """
    Försöker läsa service account-JSON från flera källor (i ordning):
      1) GOOGLE_CREDENTIALS / google_credentials / gcp_service_account
         - accepterar Mapping (AttrDict), dict, str (JSON) eller bytes
      2) GOOGLE_CREDENTIALS_B64 (base64 av JSON)
      3) GOOGLE_CREDENTIALS_PATH (fil på disk)
      4) separata nycklar SHEETS_CLIENT_EMAIL + SHEETS_PRIVATE_KEY
    """
    # 1) Direkt i secrets
    raw = (
        st.secrets.get("GOOGLE_CREDENTIALS")
        or st.secrets.get("google_credentials")
        or st.secrets.get("gcp_service_account")
    )
    if raw:
        # Ny: acceptera alla Mapping-varianter (inkl. Streamlits AttrDict)
        if isinstance(raw, Mapping):
            return _fix_private_key_newlines(dict(raw)), "secrets:mapping"
        if isinstance(raw, dict):
            return _fix_private_key_newlines(dict(raw)), "secrets:dict"
        if isinstance(raw, (bytes, bytearray)):
            return _fix_private_key_newlines(_json_from_str(raw.decode("utf-8"))), "secrets:json_bytes"
        if isinstance(raw, str):
            return _fix_private_key_newlines(_json_from_str(raw)), "secrets:json_str"
        # Om något annat ovanligt dyker upp – försök tolkning via str()
        try:
            return _fix_private_key_newlines(_json_from_str(str(raw))), "secrets:coerced_str"
        except Exception:
            raise RuntimeError("GOOGLE_CREDENTIALS fanns men hade okänt format (varken mapping/str/bytes).")

    # 2) BASE64
    b64 = st.secrets.get("GOOGLE_CREDENTIALS_B64")
    if b64:
        return _fix_private_key_newlines(_json_from_b64(b64)), "secrets:b64"

    # 3) PATH
    path = st.secrets.get("GOOGLE_CREDENTIALS_PATH")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _fix_private_key_newlines(data), "file:path"
        except Exception as e:
            raise RuntimeError(f"Kunde inte läsa credentials-fil ({e}).")

    # 4) Separerade nycklar
    sep = _build_minimal_dict_from_separate_keys()
    if sep:
        return sep, "separate_keys"

    raise RuntimeError(
        "Hittade inga service account-uppgifter. "
        "Använd någon av: GOOGLE_CREDENTIALS (dict/str/mapping), GOOGLE_CREDENTIALS_B64, "
        "GOOGLE_CREDENTIALS_PATH eller SHEETS_CLIENT_EMAIL + SHEETS_PRIVATE_KEY."
    )


def _validate_creds(data: dict) -> None:
    missing = [k for k in ("client_email", "private_key") if not data.get(k)]
    if missing:
        raise RuntimeError(f"Service account saknar fält: {', '.join(missing)}.")


def _load_credentials():
    if Credentials is None:
        raise RuntimeError("google-auth saknas i miljön.")
    data, _ = _load_credentials_dict()
    _validate_creds(data)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        return Credentials.from_service_account_info(data, scopes=scope)
    except Exception as e:
        raise RuntimeError(f"Kunde inte skapa Google-credentials: {e}")


def _client():
    if gspread is None:
        raise RuntimeError("gspread saknas i miljön.")
    creds = _load_credentials()
    return gspread.authorize(creds)


# ---------------- Spreadsheet open ----------------
def _spreadsheet():
    """
    Öppnar spreadsheet via URL (SHEET_URL) eller key (SHEET_ID) från secrets.
    """
    url = st.secrets.get("SHEET_URL")
    key = st.secrets.get("SHEET_ID")
    if not url and not key:
        raise RuntimeError("SHEET_URL eller SHEET_ID saknas i secrets.")
    cli = _client()
    if url:
        return _with_backoff(cli.open_by_url, url)
    return _with_backoff(cli.open_by_key, key)


def get_spreadsheet():
    return _spreadsheet()


# ---------------- Diagnostics ----------------
def creds_debug_summary() -> str:
    """
    Kort diagnos om hur credentials hittades och ser ut (inga hemligheter läcks).
    """
    try:
        data, source = _load_credentials_dict()
        email = bool(data.get("client_email"))
        pk = str(data.get("private_key", ""))
        has_pk = bool(pk)
        has_nl = ("\\n" in pk) or ("\n" in pk)
        return f"Källa: {source} | client_email: {email} | private_key: {has_pk} | radbrytningar: {has_nl}"
    except Exception as e:
        return f"Creds: FEL – {e}"


def test_connection() -> Tuple[bool, str]:
    try:
        ss = get_spreadsheet()
        _ = [ws.title for ws in ss.worksheets()]
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ---------------- Headerhjälp ----------------
def _trim_trailing_empty(seq: List[str]) -> List[str]:
    out = list(seq)
    while out and (out[-1] is None or str(out[-1]).strip() == ""):
        out.pop()
    return out


def _sheet_headers(ws) -> List[str]:
    row1 = _with_backoff(ws.row_values, 1) or []
    headers = [str(x).strip() for x in row1]
    headers = _trim_trailing_empty(headers)
    if not headers:
        raise RuntimeError("Arket saknar header-rad (rad 1).")
    return headers


def list_worksheet_titles() -> List[str]:
    try:
        ss = get_spreadsheet()
        return [sh.title for sh in ss.worksheets()]
    except Exception:
        return []


def delete_worksheet(title: str) -> None:
    try:
        ss = get_spreadsheet()
        ws = ss.worksheet(title)
        _with_backoff(ss.del_worksheet, ws)
    except Exception:
        pass


# ---------------- Läs / skriv DataFrame ----------------
def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läser ett blad till DataFrame i exakt arket’s kolumnordning.
    Slänger helt tomma rader; fyller saknade celler med "".
    """
    ss = get_spreadsheet()
    ws = _with_backoff(ss.worksheet, title)

    vals: List[List[str]] = _with_backoff(ws.get_all_values)
    if not vals:
        return pd.DataFrame()

    headers = _sheet_headers(ws)
    data_rows = vals[1:] if len(vals) > 1 else []

    norm_rows: List[List[str]] = []
    for r in data_rows:
        r = r[: len(headers)] + [""] * max(0, len(headers) - len(r))
        if all((str(x).strip() == "") for x in r):
            continue
        norm_rows.append(r)

    return pd.DataFrame(norm_rows, columns=headers)


def ws_write_df(title: str, df: pd.DataFrame) -> None:
    """
    Skriver DF till bladet i arket’s headerordning.
    Saknade kolumner fylls med tomt, extra DF-kolumner ignoreras (så inget förskjuts).
    """
    ss = get_spreadsheet()
    ws = _with_backoff(ss.worksheet, title)
    headers = _sheet_headers(ws)

    body: List[List[str]] = [headers]
    for _, row in df.iterrows():
        out_row: List[str] = []
        for h in headers:
            v = row[h] if h in df.columns else ""
            if pd.isna(v):
                v = ""
            out_row.append(str(v))
        body.append(out_row)

    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)
