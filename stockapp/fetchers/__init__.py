# -*- coding: utf-8 -*-
"""Tunt init så att import av `stockapp.fetchers` inte laddar hela världen."""
from __future__ import annotations

__all__ = []  # importera under-moduler explicit: stockapp.fetchers.yahoo, fmp, sec, orchestrator
