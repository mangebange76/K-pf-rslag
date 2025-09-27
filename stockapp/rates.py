# -*- coding: utf-8 -*-
import requests
import streamlit as st
from .sheets import read_saved_rates, save_rates
from .config import STANDARD_VALUTAKURSER

def _fetch_auto_rates():
    misses = []
    rates = {}
    provider = None

    # 1) FMP (om key finns)
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    if fmp_key:
        try:
            base = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
            def _pair(pair):
                url = f"{base}/api/v3/fx/{pair}"
                r = requests.get(url, params={"apikey": fmp_key}, timeout=15)
                if r.status_code != 200:
                    return None, r.status_code
                j = r.json() or {}
                px = j.get("price")
                return float(px) if px is not None else None, 200
            provider = "FMP"
            for pair in ("USDSEK","NOKSEK","CADSEK","EURSEK"):
                v, sc = _pair(pair)
                if v and v > 0: rates[pair[:3]] = float(v)
                else: misses.append(f"{pair} (HTTP {sc if sc else '??'})")
        except Exception:
            pass

    # 2) Frankfurter
    if len(rates) < 4:
        provider = "Frankfurter"
        for base_ccy in ("USD","EUR","CAD","NOK"):
            try:
                r2 = requests.get("https://api.frankfurter.app/latest",
                                  params={"from": base_ccy, "to": "SEK"}, timeout=10)
                if r2.status_code == 200:
                    v = (r2.json() or {}).get("rates", {}).get("SEK")
                    if v: rates[base_ccy] = float(v)
            except Exception:
                pass

    # 3) exchangerate.host
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base_ccy in ("USD","EUR","CAD","NOK"):
            try:
                r = requests.get("https://api.exchangerate.host/latest",
                                 params={"base": base_ccy, "symbols": "SEK"}, timeout=10)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v: rates[base_ccy] = float(v)
            except Exception:
                pass

    # fyll luckor
    saved = read_saved_rates()
    for base_ccy in ("USD","EUR","CAD","NOK"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))

    return rates, misses, (provider or "okänd")

def sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")
    saved = read_saved_rates()

    # permanenta state
    for k in ("USD","NOK","CAD","EUR"):
        key = f"rate_{k.lower()}"
        if key not in st.session_state:
            st.session_state[key] = float(saved.get(k, STANDARD_VALUTAKURSER[k]))
        key_in = f"{key}_input"
        if key_in not in st.session_state:
            st.session_state[key_in] = st.session_state[key]

    info_msg = None; warn_list = None

    # knappar FÖRE widgets
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = _fetch_auto_rates()
        for k in ("USD","NOK","CAD","EUR"):
            st.session_state[f"rate_{k.lower()}_input"] = float(auto_rates.get(k, st.session_state[f"rate_{k.lower()}_input"]))
        info_msg = f"Valutakurser uppdaterade (källa: {provider})."
        warn_list = misses

    if st.sidebar.button("↻ Läs sparade kurser"):
        sr = read_saved_rates()
        for k in ("USD","NOK","CAD","EUR"):
            st.session_state[f"rate_{k.lower()}_input"] = float(sr.get(k, st.session_state[f"rate_{k.lower()}_input"]))
        info_msg = "Läste sparade kurser."

    if info_msg: st.sidebar.success(info_msg)
    if warn_list: st.sidebar.warning("Kunde inte hämta:\n- " + "\n- ".join(warn_list))

    # widgets (skapas efter ev. state-uppdateringar)
    st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd_input), step=0.01, format="%.4f", key="rate_usd_input")
    st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok_input), step=0.01, format="%.4f", key="rate_nok_input")
    st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad_input), step=0.01, format="%.4f", key="rate_cad_input")
    st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur_input), step=0.01, format="%.4f", key="rate_eur_input")

    if st.sidebar.button("💾 Spara kurser"):
        for k in ("usd","nok","cad","eur"):
            st.session_state[f"rate_{k}"] = float(st.session_state[f"rate_{k}_input"])
        to_save = {
            "USD": st.session_state.rate_usd,
            "NOK": st.session_state.rate_nok,
            "CAD": st.session_state.rate_cad,
            "EUR": st.session_state.rate_eur,
            "SEK": 1.0,
        }
        save_rates(to_save)
        st.sidebar.success("Valutakurser sparade.")

    return {
        "USD": float(st.session_state.rate_usd_input),
        "NOK": float(st.session_state.rate_nok_input),
        "CAD": float(st.session_state.rate_cad_input),
        "EUR": float(st.session_state.rate_eur_input),
        "SEK": 1.0,
    }
