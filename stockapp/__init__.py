# -*- coding: utf-8 -*-
"""
Stockapp package initializer.

This file exposes the most commonly used constants and helper namespaces so
app.py and other modules can simply import from `stockapp` without deep paths.

It deliberately avoids importing heavy modules at import-time to reduce the
risk of circular imports (e.g. nothing from `views` is imported here).
"""

from __future__ import annotations

# Re-export central configuration constants
from .config import (
    SHEET_URL,
    SHEET_NAME,
    RATES_SHEET_NAME,
    FINAL_COLS,
    TS_FIELDS,
    STANDARD_VALUTAKURSER,
    SECTOR_WEIGHTS,
)

# Lightweight namespaces that are safe to expose at package import
from . import sheets as _sheets_mod
from . import storage as _storage_mod
from . import utils as _utils_mod
from . import calc as _calc_mod
from . import scoring as _scoring_mod
from . import portfolio as _portfolio_mod
from . import investment as _investment_mod
from . import editor as _editor_mod
from . import views as _views_mod  # if you have a single consolidated views.py

# Fetchers namespace (FMP, SEC, Yahoo, Orchestrator)
from .fetchers import (
    fmp as fmp,
    sec as sec,
    yahoo as yahoo,
    orchestrator as orchestrator,
)

# Public convenience aliases (optional)
sheets = _sheets_mod
storage = _storage_mod
utils = _utils_mod
calc = _calc_mod
scoring = _scoring_mod
portfolio = _portfolio_mod
investment = _investment_mod
views = _views_mod

__all__ = [
    # Config
    "SHEET_URL",
    "SHEET_NAME",
    "RATES_SHEET_NAME",
    "FINAL_COLS",
    "TS_FIELDS",
    "STANDARD_VALUTAKURSER",
    "SECTOR_WEIGHTS",
    # Namespaces
    "sheets",
    "storage",
    "utils",
    "calc",
    "scoring",
    "portfolio",
    "investment",
    "views",
    # Fetchers
    "fmp",
    "sec",
    "yahoo",
    "orchestrator",
]

__version__ = "0.1.0"
