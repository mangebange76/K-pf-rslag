# -*- coding: utf-8 -*-
"""
stockapp.sheets
----------------
Central gspread-koppling:
- get_gspread_client()
- get_spreadsheet()
- get_ws(name: str | None = None) -> gspread.Worksheet
"""

from __future__ import annotations
from typing import Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME
from .utils import with_backoff


def _service_account_credentials() -> Credentials:
    """
    Skapar Credentials från st.secrets["GOOGLE_CREDENTIALS"].
    Kräver att credentials JSON finns i secrets.
    """
    try:
        info = st.secrets["GOOGLE_CREDENTIALS"]
    except KeyError as e:
        raise RuntimeError("GOOGLE_CREDENTIALS saknas i st.secrets") from e

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    return Credentials.from_service_account_info(info, scopes=scope)


def get_gspread_client() -> gspread.Client:
    """
    Returnerar en auktoriserad gspread-klient.
    """
    creds = _service_account_credentials()
    try:
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        raise RuntimeError(f"Kunde inte auktorisera gspread-klient: {e}") from e


def get_spreadsheet() -> gspread.Spreadsheet:
    """
    Öppnar Google Sheet via URL.
    """
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas (ange i st.secrets eller config).")
    client = get_gspread_client()
    try:
        # använd with_backoff runt http-anropet
        return with_backoff(client.open_by_url, SHEET_URL)
    except Exception as e:
        raise RuntimeError(f"Kunde inte öppna Google Sheet: {e}") from e


def get_ws(name: Optional[str] = None) -> gspread.Worksheet:
    """
    Returnerar ett worksheet-objekt.
    - name=None => använd standardfliken SHEET_NAME
    - Skapar inte nya blad; kastar fel om bladet inte finns.
    """
    ss = get_spreadsheet()
    ws_name = (name or SHEET_NAME or "").strip()
    if not ws_name:
        raise RuntimeError("Ogiltigt bladnamn: tomt.")
    try:
        return with_backoff(ss.worksheet, ws_name)
    except Exception as e:
        raise RuntimeError(f"Kunde inte hitta blad '{ws_name}': {e}") from e
