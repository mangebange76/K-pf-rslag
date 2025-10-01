# -*- coding: utf-8 -*-
"""
Lightweight package init för stockapp.

Viktigt:
- Importera inte Streamlit-vyer här (för att undvika cirkulära imports och onödiga sid-effekter).
- Håll detta modulens imports lätta. Tunga moduler importeras där de används.
"""

from __future__ import annotations

__all__ = [
    "config",
    "utils",
    "sheets",
    "storage",
    "rates",
    "scoring",
    "invest",
    "editor",
    "portfolio",
    "batch",
    "fetchers",
    "__version__",
]

__version__ = "0.9.0"

# Exponera delpaket (lättviktiga). Notera: inga Streamlit-anrop här.
from . import config  # noqa: E402
from . import utils   # noqa: E402
from . import sheets  # noqa: E402
from . import storage # noqa: E402
from . import rates   # noqa: E402
from . import scoring # noqa: E402

# De vy-modulerna (invest/editor/portfolio/batch) importeras inte här för att
# undvika att Streamlit-komponenter laddas innan app.py kör.
# De importeras direkt i app.py när de behövs.

# Exponera fetchers som subpaket (själva logiken importeras i runtime)
from . import fetchers  # noqa: E402
