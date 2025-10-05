from __future__ import annotations

import json
import time
from typing import List

import pandas as pd
import streamlit as st

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


# ---------------- Credentials / klient ----------------
def _load_credentials():
    """
    Läser service account-uppgifter från secrets, accepterar:
      - st.secrets["GOOGLE_CREDENTIALS"] som dict ELLER JSON-sträng
      - fallback-nycklar: "google_credentials", "gcp_service_account"
    Fixar privata nyckelns radbrytningar och validerar nödvändiga fält.
    """
    if Credentials is None:
        raise RuntimeError("google-auth saknas i miljön.")

    raw = (
        st.secrets.get("GOOGLE_CREDENTIALS")
        or st.secrets.get("google_credentials")
        or st.secrets.get("gcp_service_account")
    )
    if not raw:
        raise RuntimeError(
            "GOOGLE_CREDENTIALS saknas i secrets. Lägg in hela service account JSON:en "
            "(antingen som dict i .streamlit/secrets.toml eller som JSON-sträng)."
        )

    # Sträng → försök JSON-läsa
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception as e:
            raise RuntimeError(
                f"GOOGLE_CREDENTIALS kunde inte tolkas som JSON-sträng ({e})."
            )
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas.")

    # Fixa private_key radbrytningar
    pk = data.get("private_key")
    if isinstance(pk, str):
        data["private_key"] = pk.replace("\\n", "\n")

    # Validera
    missing = [k for k in ("client_email", "private_key") if not data.get(k)]
    if missing:
        raise RuntimeError(
            f"Service account saknar fält: {', '.join(missing)}. "
            "Kontrollera att du klistrat in HELA JSON-nyckeln."
        )

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
