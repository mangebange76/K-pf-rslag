# -*- coding: utf-8 -*-
"""
Minimal package initializer för att undvika cirkulära/import-biverkningar.
Importera alltid delmoduler direkt, t.ex.:
    from stockapp.config import FINAL_COLS
    from stockapp.fetchers.yahoo import fetch_all
    from stockapp.scoring import sektorviktad_score
osv.
"""

from __future__ import annotations

__all__ = []        # vi re-exporterar inget från rotpaketet
__version__ = "0.1.0"
