# -*- coding: utf-8 -*-
"""
Robusta Google Sheets-hjälpare (fristående):
- get_ws(name: str|None) -> gspread.Worksheet    (smart fallback till första fliken)
- ws_read_df(ws) -> pd.DataFrame                 (läser flexibelt & bygger DataFrame)
- ws_write_df(ws, df) -> None                    (skriver rubrik + data)
- ensure_headers(ws, headers) -> None            (säkrar rubrikrad)
"""

from __future__ import annotations
from typing import List, Optional

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME


def _client() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"],
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def _open_spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas i secrets/config.")
    cli = _client()
    return cli.open_by_url(SHEET_URL)


def get_ws(name: Optional[str] = None) -> gspread.Worksheet:
    """
    Hämta worksheet. Försök i ordning:
      1) Namnet som skickas in (om angivet)
      2) config.SHEET_NAME
      3) Första fliken i dokumentet
    """
    ss = _open_spreadsheet()

    # 1) explicit name
    if name:
        try:
            return ss.worksheet(name)
        except Exception:
            st.warning(f"⚠️ Hittade inte fliken '{name}', försöker med '{SHEET_NAME}'.")

    # 2) config.SHEET_NAME
    try:
        return ss.worksheet(SHEET_NAME)
    except Exception:
        st.warning(f"⚠️ Hittade inte fliken '{SHEET_NAME}', använder första fliken i arket.")

    # 3) första bladet
    ws = ss.get_worksheet(0)
    if ws is None:
        raise RuntimeError("Ingen flik kunde öppnas i Google Sheet.")
    return ws


def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    """
    Läser hela bladet. Tar första raden som rubriker och bygger DataFrame, även
    om raderna har olika längd (pad:ar med tomma strängar).
    Tomma rader (alla celler tomma) filtreras bort.
    """
    values = ws.get_all_values() or []
    if not values:
        return pd.DataFrame()

    # rubriker
    headers = [h.strip() for h in (values[0] or [])]
    max_len = max(len(r) for r in values)
    rows = []
    for r in values[1:]:
        rr = list(r)
        if len(rr) < max_len:
            rr += [""] * (max_len - len(rr))
        rows.append(rr)

    df = pd.DataFrame(rows, columns=headers)

    # släng helt tomma rader
    mask_nonempty = (df.astype(str).apply(lambda s: s.str.strip()) != "").any(axis=1)
    df = df[mask_nonempty].copy()

    return df


def ensure_headers(ws: gspread.Worksheet, headers: List[str]) -> None:
    """
    Säkerställ att rubrikraden exakt matchar headers. Om inte, skriv om rad 1.
    """
    try:
        row1 = ws.row_values(1) or []
    except Exception:
        row1 = []
    want = [str(h) for h in headers]
    if [c.strip() for c in row1] != want:
        ws.clear()
        ws.update([want])


def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    """
    Skriver DataFrame (rubrik + data). Tom df -> bara rubriker.
    """
    df = df.copy()
    df = df.fillna("")
    body = [list(df.columns)]
    body += df.astype(str).values.tolist()
    ws.clear()
    ws.update(body)
