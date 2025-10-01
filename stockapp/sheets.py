# -*- coding: utf-8 -*-
"""
stockapp.sheets
----------------
Säker, återanvändbar koppling till Google Sheets via gspread.
- get_ws(title=None): öppnar rätt kalkylblad och worksheet
- list_worksheets(): listar blad
- safe_get_all_values(ws): robust hämtning av alla värden
"""

from __future__ import annotations

from typing import List, Optional
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME
from .utils import with_backoff


def _client() -> gspread.Client:
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_info = st.secrets.get("GOOGLE_CREDENTIALS")
    if not creds_info:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i st.secrets.")
    credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
    return gspread.authorize(credentials)


def _spreadsheet() -> gspread.Spreadsheet:
    url = st.secrets.get("SHEET_URL", SHEET_URL)
    if not url:
        raise RuntimeError("SHEET_URL saknas i st.secrets och config.")
    cli = _client()
    return cli.open_by_url(url)


def list_worksheets() -> List[str]:
    try:
        ss = _spreadsheet()
        return [ws.title for ws in ss.worksheets()]
    except Exception as e:
        st.error(f"⚠️ Kunde inte lista worksheet: {e}")
        return []


def get_ws(title: Optional[str] = None) -> gspread.Worksheet:
    """
    Öppnar worksheet. Om 'title' saknas används SHEET_NAME.
    Faller tillbaka till första arkfliken om den angivna inte hittas.
    Skapar INTE nya blad här (skrivmodul gör det), för att undvika
    att råka skapa fel blad vid stavfel i namn.
    """
    ss = _spreadsheet()
    want = (title or SHEET_NAME or "").strip()

    if want:
        try:
            return ss.worksheet(want)
        except Exception:
            st.warning(f"Angivet worksheet '{want}' hittades inte. Faller tillbaka till första bladet.")
    # fallback: första bladet
    wss = ss.worksheets()
    if not wss:
        raise RuntimeError("Kalkylbladet saknar helt worksheets.")
    return wss[0]


def safe_get_all_values(ws: gspread.Worksheet) -> List[List[str]]:
    """
    Hämtar alla cellvärden (inklusive ev. tomma). Returnerar alltid list (kan vara []).
    """
    try:
        vals = with_backoff(ws.get_all_values)
        return vals or []
    except Exception as e:
        st.error(f"⚠️ Kunde inte läsa data från Google Sheet: {e}")
        return []
