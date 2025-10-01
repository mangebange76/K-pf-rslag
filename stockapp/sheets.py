# stockapp/sheets.py
# -*- coding: utf-8 -*-
"""
Minimal och självständig Google Sheets-koppling för appen.
- get_ws(title)  -> gspread.Worksheet (skapar bladet om det saknas)
- get_spreadsheet() -> gspread.Spreadsheet

Den här modulen är medvetet frikopplad från övriga moduler (ingen import
från stockapp.utils) för att undvika cirkulära beroenden.
"""

from __future__ import annotations

import time
from typing import Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# ---------------------------------------------------------------------
# Intern backoff (så vi slipper bero på utils)
# ---------------------------------------------------------------------
def _with_backoff(fn, *args, **kwargs):
    delay = 0.5
    last_err = None
    for _ in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 – vi vill bubbla upp sista felet
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    # ge upp
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------
# GSpread-klient
# ---------------------------------------------------------------------
def _gspread_client() -> gspread.Client:
    """Auktorisera gspread på service-account-info i st.secrets."""
    info = st.secrets.get("GOOGLE_CREDENTIALS")
    if not info:
        raise RuntimeError("Saknar 'GOOGLE_CREDENTIALS' i st.secrets.")
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scope)
    return gspread.authorize(creds)


def get_spreadsheet() -> gspread.Spreadsheet:
    """Öppna Spreadsheet via URL i st.secrets['SHEET_URL']."""
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("Saknar 'SHEET_URL' i st.secrets.")
    client = _gspread_client()
    return _with_backoff(client.open_by_url, url)


def get_ws(title: Optional[str] = None) -> gspread.Worksheet:
    """
    Hämta (eller skapa) ett worksheet med angivet namn.
    Om title är None används st.secrets['SHEET_NAME'] eller 'Blad1'.
    """
    if title is None:
        title = st.secrets.get("SHEET_NAME", "Blad1")

    ss = get_spreadsheet()

    # Försök hämta befintligt blad
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        # Skapa nytt blad om det saknas
        _with_backoff(ss.add_worksheet, title=title, rows=2000, cols=100)
        return _with_backoff(ss.worksheet, title)
