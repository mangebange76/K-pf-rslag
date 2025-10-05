# stockapp/sheets.py
from __future__ import annotations

import json
import time
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

# 3:e parts – förväntas finnas i miljön
try:
    import gspread
except Exception:
    gspread = None  # type: ignore

try:
    from google.oauth2.service_account import Credentials
except Exception:
    Credentials = None  # type: ignore


# ----------------------------
#     Backoff helper
# ----------------------------
def _with_backoff(fn, *args, **kwargs):
    """
    Kör fn med mild backoff för att hantera tillfälliga 429/kvotfel.
    """
    delays = [0.0, 0.5, 1.0, 2.0]
    last_exc = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc


# ----------------------------
#     Secrets → Credentials
# ----------------------------
def _load_credentials() -> dict:
    """
    Läser service account ur st.secrets["GOOGLE_CREDENTIALS"].
    Stöd för:
      - MagicDict/AttrDict/dict (direkt objekt)
      - JSON-sträng
    Fixar även private_key med '\\n' → '\n'.
    Kräver åtminstone client_email + private_key.
    """
    raw = st.secrets.get("GOOGLE_CREDENTIALS", None)
    if raw is None:
        raise RuntimeError("Hittade inga service account-uppgifter (GOOGLE_CREDENTIALS saknas i secrets).")

    # 1) Mapping-lika (MagicDict/AttrDict/dict)
    try:
        from collections.abc import Mapping
        if isinstance(raw, Mapping):
            creds_dict = dict(raw)
        else:
            creds_dict = None
    except Exception:
        creds_dict = None

    # 2) JSON-sträng
    if creds_dict is None:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        if not isinstance(raw, str):
            raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas.")
        try:
            creds_dict = json.loads(raw)
        except Exception:
            raise RuntimeError("GOOGLE_CREDENTIALS fanns men gick inte att tolka som JSON/dict.")

    # 3) Private key '\n' fix
    pk = str(creds_dict.get("private_key", "") or "")
    if pk and ("\\n" in pk) and ("\n" not in pk):
        creds_dict["private_key"] = pk.replace("\\n", "\n")

    # 4) Minimala nycklar
    if not creds_dict.get("client_email") or not creds_dict.get("private_key"):
        raise RuntimeError("Hittade inga service account-uppgifter (minst client_email + private_key).")

    return creds_dict


# ----------------------------
#     GSpread client & Sheet
# ----------------------------
_CLIENT: Optional["gspread.Client"] = None  # cache

def _client() -> "gspread.Client":
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if gspread is None:
        raise RuntimeError("gspread saknas i miljön.")
    if Credentials is None:
        raise RuntimeError("google.oauth2 saknas i miljön.")

    creds_dict = _load_credentials()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    _CLIENT = gspread.authorize(creds)
    return _CLIENT


def get_spreadsheet():
    """
    Öppnar kalkylbladet från st.secrets['SHEET_URL'].
    """
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    cli = _client()
    return _with_backoff(cli.open_by_url, url)


# Alias som vissa moduler importerar
def _spreadsheet():
    return get_spreadsheet()


def list_worksheet_titles() -> List[str]:
    """
    Returnerar alla blad-titlar i kalkylbladet.
    """
    ss = get_spreadsheet()
    wss = _with_backoff(ss.worksheets)
    return [w.title for w in wss]


def _get_ws_by_title(ss, title: str):
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        return None


def get_or_create_worksheet(title: str, rows: int = 100, cols: int = 26):
    """
    Hämtar ett blad med titel 'title' – skapar vid behov.
    """
    ss = get_spreadsheet()
    ws = _get_ws_by_title(ss, title)
    if ws is not None:
        return ws
    # skapa litet blad – växer automatiskt vid update
    return _with_backoff(ss.add_worksheet, title=title, rows=rows, cols=cols)


def delete_worksheet(title: str) -> None:
    """
    Tar bort blad med angiven titel om det finns.
    """
    ss = get_spreadsheet()
    ws = _get_ws_by_title(ss, title)
    if ws is None:
        return
    _with_backoff(ss.del_worksheet, ws)


# ----------------------------
#     Read / Write helpers
# ----------------------------
def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läser hela bladet till DataFrame.
    Använder get_all_values (snabbare än get_all_records vid många kolumner).
    För tomt blad → tom DF.
    """
    ss = get_spreadsheet()
    ws = _get_ws_by_title(ss, title)
    if ws is None:
        # tom df om bladet saknas (appen skapar ofta kolumner senare)
        return pd.DataFrame()

    rows = _with_backoff(ws.get_all_values)
    if not rows or len(rows) == 0:
        return pd.DataFrame()
    if len(rows) == 1:
        # Bara header
        return pd.DataFrame(columns=rows[0])

    header = rows[0]
    data = rows[1:]
    # Trimma trailing tomma kolumner (vanligt i Sheets)
    max_len = max(len(r) for r in data) if data else len(header)
    header = header[:max_len]
    norm = [r + [""] * (max_len - len(r)) for r in data]

    df = pd.DataFrame(norm, columns=header)

    # Försök konvertera numeriska kolumner – men låt appen göra typ-säkring själv.
    # (Vi lämnar som strängar; appens ensure_columns/konvertering tar vid.)
    return df


def ws_write_df(title: str, df: pd.DataFrame) -> None:
    """
    Skriver en DataFrame till bladet (överskriver).
    NaN → "".
    """
    ss = get_spreadsheet()
    ws = get_or_create_worksheet(title)

    # Till str + NaN→""
    out = df.copy()
    if not out.columns.size:
        # säkra att det finns minst en kolumn om df är tom
        out = pd.DataFrame(columns=["_"])
    out = out.astype(object).where(pd.notnull(out), "")

    # Header + data
    body = [list(map(str, out.columns.tolist()))] + [
        ["" if (x is None) else str(x) for x in row] for row in out.values.tolist()
    ]

    # Rensa + skriv
    _with_backoff(ws.clear)
    # Dela upp i chunkar om mycket data (minskar 413/413: Request Entity Too Large risk)
    # men oftast räcker en update.
    _with_backoff(ws.update, body)


# ----------------------------
#     Rates helpers (valfritt)
# ----------------------------
def ensure_rates_sheet(title: str = "Valutakurser"):
    """
    Säkerställer att ett litet 'Valutakurser'-blad finns med rubriker.
    """
    ws = get_or_create_worksheet(title, rows=10, cols=5)
    rows = _with_backoff(ws.get_all_values)
    if not rows:
        _with_backoff(ws.update, [["Valuta", "Kurs"]])
    elif rows and rows[0][:2] != ["Valuta", "Kurs"]:
        # skriv rubriker överst och behåll ev. befintligt
        vals = [["Valuta", "Kurs"]] + rows
        _with_backoff(ws.clear)
        _with_backoff(ws.update, vals)
