from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import streamlit as st

# gspread och google creds
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None  # type: ignore
    Credentials = None  # type: ignore


# ------------------------------------------------------------
# Backoff-hjälpare
# ------------------------------------------------------------
def _with_backoff(func, *args, **kwargs):
    """
    Kör en funktion med mild backoff för att hantera sporadiska 429/5xx från Sheets API.
    """
    delays = [0.0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Okänt fel i _with_backoff")


# ------------------------------------------------------------
# Autentisering / Spreadsheet-hantering
# ------------------------------------------------------------
def _load_credentials():
    """
    Läser service-account creds från st.secrets["GOOGLE_CREDENTIALS"] (dict/JSON) och
    returnerar en Credentials-instans.
    """
    if Credentials is None:
        raise RuntimeError("google-auth saknas i miljön.")
    raw = st.secrets.get("GOOGLE_CREDENTIALS")
    if not raw:
        raise RuntimeError("Hittade inga service account-uppgifter i secrets (GOOGLE_CREDENTIALS).")
    # Kan vara JSON-sträng eller dict
    if isinstance(raw, str):
        import json
        try:
            data = json.loads(raw)
        except Exception:
            raise RuntimeError("GOOGLE_CREDENTIALS fanns men kunde inte tolkas.")
    elif isinstance(raw, dict):
        data = raw
    else:
        raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas.")

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    return Credentials.from_service_account_info(data, scopes=scope)


def _client():
    if gspread is None:
        raise RuntimeError("gspread saknas i miljön.")
    creds = _load_credentials()
    return gspread.authorize(creds)


def _spreadsheet():
    """
    Returnerar gspread Spreadsheet (öppnad via URL i st.secrets['SHEET_URL']).
    Exponeras för andra moduler.
    """
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    cli = _client()
    return _with_backoff(cli.open_by_url, url)


def get_spreadsheet():
    """Alias som används av appen."""
    return _spreadsheet()


# ------------------------------------------------------------
# Hjälpare för headers / trimming
# ------------------------------------------------------------
def _trim_trailing_empty(seq: List[str]) -> List[str]:
    out = list(seq)
    while out and (out[-1] is None or str(out[-1]).strip() == ""):
        out.pop()
    return out


def _sheet_headers(ws) -> List[str]:
    """
    Läser första raden (rad 1) som headers. Trimmar tomma släpande celler.
    """
    row1 = _with_backoff(ws.row_values, 1) or []
    headers = [str(x).strip() for x in row1]
    headers = _trim_trailing_empty(headers)
    if not headers:
        raise RuntimeError("Arket saknar header-rad (rad 1 är tom).")
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


# ------------------------------------------------------------
# Läs / Skriv DataFrame
# ------------------------------------------------------------
def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läser ett blad till DataFrame, med kolumnordning exakt som arket har.
    Tar bort helt tomma rader; fyller saknade celler med "".
    """
    ss = get_spreadsheet()
    ws = _with_backoff(ss.worksheet, title)

    # Läs hela matrisen en gång (snabbare vid breda ark)
    vals: List[List[str]] = _with_backoff(ws.get_all_values)
    if not vals:
        return pd.DataFrame()

    # Hämta header exakt som arket använder
    headers = _sheet_headers(ws)

    # Skapa DataFrame med header-längd
    data_rows = vals[1:] if len(vals) > 1 else []
    norm_rows: List[List[str]] = []
    for r in data_rows:
        r = r[:len(headers)] + [""] * max(0, len(headers) - len(r))
        # är hela raden tom?
        if all((str(x).strip() == "") for x in r):
            continue
        norm_rows.append(r)

    df = pd.DataFrame(norm_rows, columns=headers)

    # Trimma helt tomma kolumner längst till höger (ifall man råkat skriva extra)
    # (men bara om de inte ingår i header – vilket de inte gör här)
    return df


def ws_write_df(title: str, df: pd.DataFrame) -> None:
    """
    Skriver en DataFrame till ett blad men *alltid* i arket’s kolumnordning.
    Kolumner som finns i arket men saknas i DF fylls med "", så inget förskjuts.
    DF-kolumner som inte finns i arket ignoreras (för att undvika misspass).
    """
    ss = get_spreadsheet()
    ws = _with_backoff(ss.worksheet, title)

    headers = _sheet_headers(ws)

    # Bygg body med exakt headerordning
    body: List[List[str]] = []
    body.append(headers)

    # Om DF har färre kolumner än headern: fyll saknade med ""
    # Om DF har fler: ignorera de extra – skriv *inte* dem (undviker förskjutning)
    for _, row in df.iterrows():
        out_row: List[str] = []
        for h in headers:
            v = row[h] if h in df.columns else ""
            # Gör om till str så vi inte får mixed types-problem
            if pd.isna(v):
                v = ""
            out_row.append(str(v))
        body.append(out_row)

    # Rensa och skriv
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)


# Bekväm wrapper som appen använder
def list_worksheet_titles_safe() -> List[str]:
    try:
        return list_worksheet_titles()
    except Exception:
        return []
