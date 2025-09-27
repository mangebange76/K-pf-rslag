# stockapp/__init__.py
"""
Håll paketets __init__ helt minimal för att undvika cirkulära importer.
Importera INTE views här. Låt app.py importera från stockapp.views.
"""
__all__ = []
