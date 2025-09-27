# stockapp/views/proposals.py
import streamlit as st
import pandas as pd

def _rate(cur: str, user_rates: dict) -> float:
    if not user_rates: return 1.0
    return float(user_rates.get(str(cur or "SEK").upper(), 1.0))

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    if df is None or df.empty:
        st.info("Inga bolag.")
        return

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        [c for c in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"] if c in df.columns],
        index=1 if "Riktkurs om 1 år" in df.columns else 0
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    base = df.copy()
    if subset == "Endast portfölj" and "Antal aktier" in base.columns:
        base = base[base["Antal aktier"] > 0].copy()

    need_cols = [riktkurs_val, "Aktuell kurs"]
    if any(c not in base.columns for c in need_cols):
        st.warning("Saknar kolumnerna som krävs för att räkna potential.")
        return

    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Pager i session_state (robust mot krascher)
    key = "forslags_index"
    if key not in st.session_state: st.session_state[key] = 0
    st.session_state[key] = min(st.session_state[key], len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state[key] = max(0, st.session_state[key] - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state[key]+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state[key] = min(len(base)-1, st.session_state[key] + 1)

    rad = base.iloc[st.session_state[key]]

    # Portfölj-värden för andelsberäkning
    port = df[df.get("Antal aktier", 0) > 0].copy() if "Antal aktier" in df.columns else pd.DataFrame()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: _rate(v, user_rates)) if not port.empty else 1.0
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"] if not port.empty else 0.0
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = _rate(rad.get("Valuta","SEK"), user_rates)
    kurs_sek = float(rad["Aktuell kurs"]) * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty and "Ticker" in port.columns:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad.get('Ticker','')})")
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad.get('Valuta','')}",
        f"- **Riktkurs idag:** {round(rad.get('Riktkurs idag',0.0),2)} {rad.get('Valuta','')}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad.get('Riktkurs om 1 år',0.0),2)} {rad.get('Valuta','')}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad.get('Riktkurs om 2 år',0.0),2)} {rad.get('Valuta','')}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad.get('Riktkurs om 3 år',0.0),2)} {rad.get('Valuta','')}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
    ]
    st.markdown("\n".join(lines))
