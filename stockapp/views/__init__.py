# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def _fallback(name):
    def _view(df, *args, **kwargs):
        st.warning(f"Vyn '{name}' saknas. Visar enkel fallback.")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df.head(30), use_container_width=True)
        else:
            st.info("Ingen data att visa.")
        return df
    return _view

# Försök importera riktiga vyer
try:
    from .analysis import analysvy  # noqa: F401
except Exception:
    analysvy = _fallback("analysvy")

try:
    from .edit import lagg_till_eller_uppdatera  # noqa: F401
except Exception:
    def lagg_till_eller_uppdatera(df, *args, **kwargs):
        st.warning("Vyn 'Lägg till / uppdatera bolag' saknas. Fallback visar data.")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True)
        return df

try:
    from .portfolio import visa_portfolj  # noqa: F401
except Exception:
    def visa_portfolj(df, *args, **kwargs):
        st.warning("Vyn 'Portfölj' saknas. Fallback visar innehav (Antal aktier > 0) om möjligt.")
        if isinstance(df, pd.DataFrame) and "Antal aktier" in df.columns:
            port = df[df["Antal aktier"] > 0]
            if not port.empty:
                st.dataframe(port, use_container_width=True)
            else:
                st.info("Inget innehav hittat.")
        else:
            st.info("Data saknar kolumnen 'Antal aktier'.")

try:
    from .proposal import visa_investeringsforslag  # noqa: F401
except Exception:
    def visa_investeringsforslag(df, *args, **kwargs):
        st.warning("Vyn 'Investeringsförslag' saknas. Fallback visar top-20 på P/S-snitt om tillgängligt.")
        if isinstance(df, pd.DataFrame) and "P/S-snitt" in df.columns:
            v = df[df["P/S-snitt"] > 0].sort_values(by="P/S-snitt").head(20)
            if not v.empty:
                st.dataframe(v[["Ticker","Bolagsnamn","P/S-snitt"]], use_container_width=True)
            else:
                st.info("Inga bolag med P/S-snitt > 0.")
        else:
            st.info("Data saknar 'P/S-snitt'.")
        return df

try:
    from .control import kontrollvy  # noqa: F401
except Exception:
    def kontrollvy(df, *args, **kwargs):
        st.warning("Vyn 'Kontroll' saknas. Fallback visar de 20 äldsta TS (om finns TS_*-kolumner).")
        if isinstance(df, pd.DataFrame) and not df.empty:
            ts_cols = [c for c in df.columns if str(c).startswith("TS_")]
            if ts_cols:
                work = df.copy()
                def _pick_oldest(row):
                    for c in ts_cols:
                        v = row.get(c, "")
                        return v
                st.dataframe(work.head(20), use_container_width=True)
            else:
                st.dataframe(df.head(20), use_container_width=True)
        else:
            st.info("Ingen data att visa.")
        return df
