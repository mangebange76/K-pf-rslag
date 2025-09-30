# -*- coding: utf-8 -*-
"""
Fetcher namespace initializer.

Keeps imports lightweight and re-exports the main functions so callers can do:
    from stockapp.fetchers import yahoo, fmp, sec, orchestrator
"""

from __future__ import annotations

from . import yahoo as yahoo
from . import fmp as fmp
from . import sec as sec
from .orchestrator import (
    run_update_price_only,
    run_update_full,
    merge_fetch_payloads,
)

__all__ = [
    "yahoo",
    "fmp",
    "sec",
    "run_update_price_only",
    "run_update_full",
    "merge_fetch_payloads",
]
