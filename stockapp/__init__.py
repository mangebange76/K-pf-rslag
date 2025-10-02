# -*- coding: utf-8 -*-
"""
stockapp package
----------------
Medvetet minimal __init__ för att undvika cirkulära imports.
Importera moduler explicit där de används:
    from stockapp.sheets import get_ws, get_spreadsheet
    from stockapp.storage import hamta_data, spara_data
    ...
"""

from __future__ import annotations

__all__ = [
    # håll denna tom eller lista enbart rena konstanter/namn utan sid-effekter
]
