# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME
from .utils import with_backoff


def _gspread_client() -> gspread.client.Client:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_info = st.secrets.get("GOOGLE_CREDENTIALS", None)
    if not creds_info:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i st.secrets.")
    credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
    return gspread.authorize(credentials)


def get_spreadsheet() -> gspread.Spreadsheet:
    client = _gspread_client()
    url = st.secrets.get("SHEET_URL", SHEET_URL)
    if not url:
        raise RuntimeError("SHEET_URL saknas i st.secrets och config.")
    return with_backoff(client.open_by_url, url)


def get_ws(name: Optional[str] = None) -> gspread.Worksheet:
    """
    Returnerar önskat arbetsblad. Provar:
      1) Angivet name (eller config.SHEET_NAME)
      2) Första arbetsbladet (index 0)
    """
    ss = get_spreadsheet()
    # 1) Försök explicit namn
    target = name or SHEET_NAME
    if target:
        try:
            return with_backoff(ss.worksheet, target)
        except Exception:
            pass
    # 2) Fallback: första bladet
    try:
        return with_backoff(ss.get_worksheet, 0)
    except Exception as e:
        raise RuntimeError(f"Kunde inte öppna något arbetsblad: {e}")
