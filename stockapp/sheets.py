# stockapp/sheets.py
from __future__ import annotations

from typing import List, Optional, Mapping, Any
import time
import json

import pandas as pd
import streamlit as st

# Google Sheets
try:
    import gspread
except Exception:
    gspread = None  # type: ignore

try:
    from google.oauth2.service_account import Credentials as SACredentials
except Exception:
    SACredentials = None  # type: ignore


# ------------------------------------------------------------
# Backoff-hjälpare
# ------------------------------------------------------------
def _with_backoff(func, *args, **kwargs):
    """
    Kör func med enkel backoff. Används för gspread-anrop.
    """
    delays = [0, 0.4, 0.8, 1.6, 3.2]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    # Om allt misslyckas → höj sista felet
    raise last_err  # type: ignore


# ------------------------------------------------------------
# Credentials + klient
# ------------------------------------------------------------
def _load_credentials():
    """
    Läser st.secrets["GOOGLE_CREDENTIALS"] som kan vara:
      - dict/AttrDict (rekommenderat i secrets.toml som [GOOGLE_CREDENTIALS])
      - JSON-sträng (hela service-account JSON som str)
    Fixar även private_key med '\\n' -> '\n'.
    """
    raw = st.secrets.get("GOOGLE_CREDENTIALS", None)
    if not raw:
        raise RuntimeError("Hittade inga service account-uppgifter (GOOGLE_CREDENTIALS saknas).")

    data: Optional[dict] = None

    # 1) Mapping (dict/AttrDict)
    if isinstance(raw, Mapping):
        data = dict(raw)  # AttrDict -> vanlig dict

    # 2) Sträng (JSON)
    elif isinstance(raw, str):
        # Prova ladda som JSON
        try:
            data = json.loads(raw)
        except Exception:
            # Om strängen inte är JSON, avbryt med tydligt fel
            raise RuntimeError("GOOGLE_CREDENTIALS fanns men gick inte att tolka som JSON/dict.")

    else:
        # 3) Fallback: försök kasta till dict
        try:
            data = dict(raw)  # type: ignore
        except Exception:
            data = None

    if not isinstance(data, dict):
        raise RuntimeError("GOOGLE_CREDENTIALS fanns men gick inte att tolka som JSON/dict.")

    # Normalisera private_key (vanligt med '\\n' i secrets)
    pk = data.get("private_key")
    if isinstance(pk, str) and "\\n" in pk and "BEGIN PRIVATE KEY" in pk:
        data["private_key"] = pk.replace("\\n", "\n")

    if not data.get("client_email") or not data.get("private_key"):
        raise RuntimeError("Hittade inga service account-uppgifter (minst client_email + private_key).")

    if SACredentials is None:
        raise RuntimeError("google-auth (google.oauth2.service_account) saknas i miljön.")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    return SACredentials.from_service_account_info(data, scopes=scopes)


def _client():
    if gspread is None:
        raise RuntimeError("gspread saknas i miljön.")
    creds = _load_credentials()
    return gspread.authorize(creds)


# ------------------------------------------------------------
# Spreadsheet helpers
# ------------------------------------------------------------
def get_spreadsheet():
    """
    Öppna kalkylbladet via URL i secrets.
    Kräver st.secrets['SHEET_URL'].
    """
    url = st.secrets.get("SHEET_URL", "")
    if not url:
        raise RuntimeError("Hittade inget Spreadsheet-ID/URL i secrets.")
    cli = _client()
    return _with_backoff(cli.open_by_url, url)


def _get_or_create_ws(title: str):
    """
    Hämta worksheet med given titel, eller skapa nytt om det saknas.
    Skapar en minimal header-rad om arket är tomt.
    """
    ss = get_spreadsheet()
    # Försök hitta befintligt
    try:
        ws = _with_backoff(ss.worksheet, title)
        return ws
    except Exception:
        pass

    # Skapa nytt
    ws = _with_backoff(ss.add_worksheet, title=title, rows=100, cols=20)
    _with_backoff(ws.update, [["Ticker"]], value_input_option="USER_ENTERED")
    return ws


# ------------------------------------------------------------
# Publika IO-funktioner för DataFrame
# ------------------------------------------------------------
def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läs ett helt blad till DataFrame.
    Antag: första raden = header.
    """
    ss = get_spreadsheet()
    ws = _with_backoff(ss.worksheet, title)

    # get_all_values är snabbare än get_all_records vid många kolumner
    rows: List[List[str]] = _with_backoff(ws.get_all_values)
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []
    # Säkerställ att varje rad har lika många kolumner som headern
    norm: List[List[str]] = []
    for r in data:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[: len(header)]
        norm.append(r)

    df = pd.DataFrame(norm, columns=header)
    return df


def ws_write_df(title: str, df: pd.DataFrame) -> None:
    """
    Skriv en DataFrame till ett blad (clear + update).
    """
    ws = _get_or_create_ws(title)
    # Konvertera till strängar (Google Sheets är textbaserat vid update)
    body = [list(df.columns)]
    if not df.empty:
        body += df.astype(str).values.tolist()

    _with_backoff(ws.clear)
    if body:
        _with_backoff(ws.update, body, value_input_option="USER_ENTERED")


def list_worksheet_titles() -> List[str]:
    """
    Lista alla blad (worksheet titles) i spreadsheetet.
    """
    ss = get_spreadsheet()
    try:
        return [w.title for w in _with_backoff(ss.worksheets)]
    except Exception:
        return []


def delete_worksheet(title: str) -> None:
    """
    Ta bort ett blad med exakt titel.
    """
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, title)
    except Exception:
        return
    _with_backoff(ss.del_worksheet, ws)
