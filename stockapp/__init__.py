# stockapp/__init__.py
"""
Håll __init__ minimal för att undvika cirkulära importer.
Importera inte views här – app.py importerar från stockapp.views.
"""
__all__ = []
