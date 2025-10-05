# stockapp/sheets.py
from __future__ import annotations

import base64
import json
import re
import time
from typing import List, Optional

import pandas as pd
import streamlit as st

# 3p
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:  # pragma: no cover
    gspread = None
    Credentials = None


# --------- Konfiguration ----------
SHEET_URL = (
    st.secrets.get("SHEET_URL")
    or st.secrets.get("GSHEET_URL")
    or st.secrets.get("GOOGLE_SHEET_URL")
    or ""
)

# Mindre default-storlek när vi skapar nya blad (spar cell-kvoten)
NEW_SHEET_ROWS = int(st.secrets.get("NEW_SHEET_ROWS", 200))
NEW_SHEET_COLS = int(st.secrets.get("NEW_SHEET_COLS", 60))


# --------- Hjälpare ----------
def _with_backoff(fn, *args, **kwargs):
    """Mild backoff för Sheets-kvoter/429."""
    delays = [0, 0.6, 1.2, 2.4]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover
            last = e
    raise last


def _normalize_private_key(k: str) -> str:
    """Hantera '\\n' i hemligheten och se till att nyckeln har rätt header/footer."""
    if not k:
        return k
    k = k.replace("\\n", "\n").strip()
    if "BEGIN PRIVATE KEY" not in k:
        # ibland saknas header/footer i miljöer där key lagts in "rå"
        k = "-----BEGIN PRIVATE KEY-----\n" + k + "\n-----END PRIVATE KEY-----\n"
    return k


def _load_credentials() -> dict:
    """
    Läser GOOGLE_CREDENTIALS från st.secrets.
    Accepterar:
      - dict (redan laddad)
      - JSON-sträng
      - Base64-kodat JSON
    Kräver minst client_email + private_key.
    """
    raw = st.secrets.get("GOOGLE_CREDENTIALS")
    if raw is None:
        raise RuntimeError(
            "Saknar GOOGLE_CREDENTIALS i .streamlit/secrets. "
            "Lägg in hela service account-objektet."
        )

    data = None

    # 1) Redan dict
    if isinstance(raw, dict):
        data = dict(raw)

    # 2) Base64?
    if data is None and isinstance(raw, str):
        s = raw.strip()
        try:
            maybe = base64.b64decode(s).decode("utf-8")
            data = json.loads(maybe)
        except Exception:
            pass

    # 3) JSON-sträng
    if data is None and isinstance(raw, str):
        s = raw.strip()
        try:
            data = json.loads(s)
        except Exception:
            # ibland har man rå-dump där quotes/specialtecken strulat;
            # gör ett sista försök att ersätta enkelquotes
            try:
                s2 = s.replace("'", '"')
                data = json.loads(s2)
            except Exception as e:
                raise RuntimeError(
                    "GOOGLE_CREDENTIALS fanns men gick inte att tolka som JSON/dict."
                ) from e

    if not isinstance(data, dict):
        raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas.")

    # Säkerställ privata nyckeln
    if not data.get("client_email") or not data.get("private_key"):
        raise RuntimeError(
            "Hittade inga service account-uppgifter (minst client_email + private_key)."
        )
    data["private_key"] = _normalize_private_key(str(data["private_key"]))
    return data


def _client():
    if gspread is None or Credentials is None:  # pragma: no cover
        raise RuntimeError("gspread eller google.oauth2 saknas i miljön.")
    creds_dict = _load_credentials()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


def get_spreadsheet():
    if not SHEET_URL:
        raise RuntimeError(
            "SHEET_URL saknas i secrets. Lägg till URL till ditt kalkylark."
        )
    cli = _client()
    if SHEET_URL.startswith("http"):
        return _with_backoff(cli.open_by_url, SHEET_URL)
    # annars tolka som key
    return _with_backoff(cli.open_by_key, SHEET_URL)


def list_worksheet_titles() -> List[str]:
    try:
        ss = get_spreadsheet()
        return [ws.title for ws in _with_backoff(ss.worksheets)]
    except Exception:
        return []


def _get_or_create_ws(ss, title: str):
    """Hämta bladet. Skapa lättviktsblad om det saknas."""
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        # Skapa med liten dimension för att spara cell-kvota
        _with_backoff(ss.add_worksheet, title=title, rows=NEW_SHEET_ROWS, cols=NEW_SHEET_COLS)
        return _with_backoff(ss.worksheet, title)


def ws_read_df(ws_title: str) -> pd.DataFrame:
    """
    Läser hela bladet.
    Returnerar alltid en DataFrame (tom om bladet är tomt).
    """
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, ws_title)
    except Exception:
        # skapa litet blad med en header-rad som räddar första körningen
        ws = _get_or_create_ws(ss, ws_title)
        _with_backoff(ws.update, [["Ticker"]])
        return pd.DataFrame(columns=["Ticker"])

    # Läs värden (snabbare än get_all_records för breda ark)
    rows = _with_backoff(ws.get_all_values)
    if not rows:
        return pd.DataFrame(columns=["Ticker"])

    headers = rows[0]
    data = rows[1:] if len(rows) > 1 else []
    # trimma/utjämna rader
    width = len(headers)
    fixed = []
    for r in data:
        r = list(r)
        if len(r) < width:
            r += [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        fixed.append(r)
    df = pd.DataFrame(fixed, columns=headers)
    return df


def ws_write_df(ws_title: str, df: pd.DataFrame):
    """Skriv DataFrame till bladet (överskriv)."""
    ss = get_spreadsheet()
    ws = _get_or_create_ws(ss, ws_title)

    values = [list(df.columns)]
    if not df.empty:
        values += df.astype(str).fillna("").values.tolist()

    _with_backoff(ws.clear)
    if values:
        _with_backoff(ws.update, values)


def delete_worksheet(title: str) -> bool:
    """Ta bort ett blad; returnerar True/False."""
    try:
        ss = get_spreadsheet()
        ws = _with_backoff(ss.worksheet, title)
        _with_backoff(ss.del_worksheet, ws)
        return True
    except Exception:
        return False
