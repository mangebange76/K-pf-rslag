# -*- coding: utf-8 -*-
"""
Init för datakällor (fetchers).

Detta gör två saker:
1) Exporterar bekväma alias till de vanligaste funktionerna:
   - get_live_price (yahoo)
   - run_update_full (orchestrator)
2) Anger __all__ så appen tydligt kan använda dessa.

OBS: Själva modulerna (yahoo, fmp, sec) kan vara tunga eller kräva nätverk,
men att importera deras toppnivå är okej. Funktionerna anropas av appen/batchen.
"""

from __future__ import annotations

__all__ = [
    "yahoo",
    "fmp",
    "sec",
    "orchestrator",
    "get_live_price",
    "run_update_full",
]

from . import yahoo       # noqa: E402
from . import fmp         # noqa: E402
from . import sec         # noqa: E402
from . import orchestrator  # noqa: E402

# Bekväma alias som appen kan importera direkt:
get_live_price = yahoo.get_live_price
run_update_full = orchestrator.run_update_full
