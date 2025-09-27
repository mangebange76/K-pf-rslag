# stockapp/views/__init__.py
"""
Exponerar vy-funktionerna så att app.py kan göra:
from stockapp.views import analysvy, kontrollvy, lagg_till_eller_uppdatera, visa_investeringsforslag, visa_portfolj
"""
from .analysis import analysvy
from .control import kontrollvy
from .edit import lagg_till_eller_uppdatera
from .portfolio import visa_portfolj
from .proposals import visa_investeringsforslag

__all__ = [
    "analysvy",
    "kontrollvy",
    "lagg_till_eller_uppdatera",
    "visa_portfolj",
    "visa_investeringsforslag",
]
