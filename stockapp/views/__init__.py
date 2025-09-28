# -*- coding: utf-8 -*-
from .proposals import visa_investeringsforslag

# Om du redan har separata filer för andra vyer (analysis.py, control.py, edit.py, portfolio.py),
# och de exponerar funktionerna nedan, kan du exportera dem här också:
try:
    from .analysis import analysvy
except Exception:
    def analysvy(*args, **kwargs):
        import streamlit as st
        st.info("Analys-vy saknas i denna build.")

try:
    from .control import kontrollvy
except Exception:
    def kontrollvy(*args, **kwargs):
        import streamlit as st
        st.info("Kontroll-vy saknas i denna build.")

try:
    from .edit import lagg_till_eller_uppdatera
except Exception:
    def lagg_till_eller_uppdatera(*args, **kwargs):
        import streamlit as st
        st.info("Redigerings-vy saknas i denna build.")

try:
    from .portfolio import visa_portfolj
except Exception:
    def visa_portfolj(*args, **kwargs):
        import streamlit as st
        st.info("Portfölj-vy saknas i denna build.")
