# stockapp/__init__.py
# Gör 'stockapp' till ett riktigt paket och (valfritt) re-exportera vyerna.
from .views import (
    analysvy,
    kontrollvy,
    lagg_till_eller_uppdatera,
    visa_investeringsforslag,
    visa_portfolj,
)

__all__ = [
    "analysvy",
    "kontrollvy",
    "lagg_till_eller_uppdatera",
    "visa_investeringsforslag",
    "visa_portfolj",
]
