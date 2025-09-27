# -*- coding: utf-8 -*-
import streamlit as st
from .yahoo import fetch_yahoo_basics, yahoo_quarterly_revenues, yahoo_prices_for_dates, implied_shares_from_yahoo
from .sec import sec_cik_for, sec_companyfacts, sec_latest_shares_robust, sec_quarterly_revenues_dated_with_unit
from .fmp import fmp_ratios_ttm, fmp_ratios_quarterly, fx_rate_cached

def auto_fetch_for_ticker(ticker: str) -> tuple[dict, dict]:
    """
    Huvudpipeline för ett bolag:
      1) Yahoo-basics (namn, valuta, pris, utdelning, sector/industry, EV/EBITDA, marketCap, sharesOutstanding)
      2) SEC (om US/CIK): shares (instant), kvartalsintäkter (3-mån), TTM → P/S nu & Q1–Q4
         - prisvaluta vs rapportvaluta → FX-konvertering
         - historiska priser för P/S Q1–Q4 & MCAP Q1–Q4
      3) Yahoo global fallback: quarterly_financials → TTM → P/S
      4) FMP (om API-nyckel): ratios-ttm & quarterly ratios (backup för P/S)
    Returnerar (vals, debug)
    """
    tkr = str(ticker).strip().upper()
    vals, debug = {}, {"ticker": tkr}

    # --- 1) Yahoo basics ---
    try:
        y = fetch_yahoo_basics(tkr)
        debug["yahoo_basics"] = y.copy()
        for k in ("Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","Sector","Industry","EV","EBITDA","EV/EBITDA","Market Cap (valuta)","Debt/Equity","Gross Margin (%)","Net Margin (%)","Cash & Equivalents","Free Cash Flow"):
            v = y.get(k)
            if v not in (None, ""):
                vals[k] = v
        # shares (implied) — i styck
        if y.get("_implied_shares") and float(y["_implied_shares"]) > 0:
            vals["Utestående aktier"] = float(y["_implied_shares"]) / 1e6
    except Exception as e:
        debug["yahoo_err"] = str(e)

    # --- 2) SEC ---
    try:
        cik = sec_cik_for(tkr)
        debug["sec_cik"] = cik
        if cik:
            facts, sc = sec_companyfacts(cik)
            debug["sec_sc"] = sc
            if sc == 200 and isinstance(facts, dict):
                # shares instant (robust summering) — fallback endast om yahoo-implied saknas
                sec_sh = sec_latest_shares_robust(facts)
                if (not vals.get("Utestående aktier")) and sec_sh and sec_sh > 0:
                    vals["Utestående aktier"] = float(sec_sh) / 1e6
                    debug["_shares_source"] = "SEC instant (robust)"
                # kvartalsintäkter (daterade) + unit
                rows, unit = sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
                debug["sec_quarters"] = {"n": len(rows), "unit": unit}
                if rows and unit:
                    # bygg TTM-fönster (nyast→äldst), 4 st
                    ttm = []
                    if len(rows) >= 4:
                        for i in range(0, min(5, len(rows) - 3)):  # ta upp till 5 TTM så vi kan välja 4 senaste okej
                            end_i = rows[i][0]
                            ttm_i = sum(v for (_, v) in rows[i:i+4])
                            ttm.append((end_i, float(ttm_i)))
                    px_ccy = (vals.get("Valuta") or "USD").upper()
                    conv = 1.0
                    if unit and px_ccy and unit.upper() != px_ccy:
                        conv = fx_rate_cached(unit.upper(), px_ccy) or 1.0
                    ttm_px = [(d, v*conv) for (d, v) in ttm]
                    # mcap nu
                    mcap_now = float(vals.get("Market Cap (valuta)", 0.0))
                    if mcap_now <= 0 and vals.get("Aktuell kurs", 0) and vals.get("Utestående aktier", 0):
                        mcap_now = float(vals["Aktuell kurs"]) * float(vals["Utestående aktier"]) * 1e6
                    # sätt P/S nu
                    if mcap_now > 0 and ttm_px:
                        ltm_now = float(ttm_px[0][1])
                        if ltm_now > 0:
                            vals["P/S"] = mcap_now / ltm_now
                            debug["_ps_now_source"] = "SEC TTM via FX"
                    # P/S Q1–Q4 + MCAP Q1–Q4 via prisdatan
                    if ttm_px and vals.get("Utestående aktier", 0) > 0:
                        q_dates = [d for (d, _) in ttm_px]
                        px_map = yahoo_prices_for_dates(tkr, q_dates)
                        shares = float(vals["Utestående aktier"]) * 1e6
                        ps_quads = {}
                        mcap_quads = {}
                        idx = 1
                        for (d_end, ttm_rev_px) in ttm_px:
                            if idx > 4: break
                            if ttm_rev_px and ttm_rev_px > 0 and px_map.get(d_end):
                                mcap_hist = shares * float(px_map[d_end])
                                ps_hist = mcap_hist / float(ttm_rev_px)
                                ps_quads[f"P/S Q{idx}"] = ps_hist
                                mcap_quads[f"MCAP Q{idx}"] = mcap_hist
                                idx += 1
                        vals.update(ps_quads)
                        vals.update(mcap_quads)
    except Exception as e:
        debug["sec_err"] = str(e)

    # --- 3) Yahoo quarterly fallback (om P/S saknas) ---
    try:
        if "P/S" not in vals:
            q_rows = yahoo_quarterly_revenues(tkr)
            if len(q_rows) >= 4:
                # TTM windows
                ttm = []
                for i in range(0, min(5, len(q_rows) - 3)):
                    end_i = q_rows[i][0]
                    ttm_i = sum(v for (_, v) in q_rows[i:i+4])
                    ttm.append((end_i, float(ttm_i)))
                mcap_now = float(vals.get("Market Cap (valuta)", 0.0))
                if mcap_now <= 0 and vals.get("Aktuell kurs", 0) and vals.get("Utestående aktier", 0):
                    mcap_now = float(vals["Aktuell kurs"]) * float(vals["Utestående aktier"]) * 1e6
                if mcap_now > 0 and ttm:
                    ltm_now = float(ttm[0][1])
                    if ltm_now > 0:
                        vals["P/S"] = mcap_now / ltm_now
                        debug["_ps_now_source"] = "Yahoo TTM"
                # P/S Q1–Q4 via pris
                if ttm and vals.get("Utestående aktier", 0) > 0:
                    q_dates = [d for (d, _) in ttm]
                    px_map = yahoo_prices_for_dates(tkr, q_dates)
                    shares = float(vals["Utestående aktier"]) * 1e6
                    idx = 1
                    for (d_end, ttm_rev) in ttm:
                        if idx > 4: break
                        if ttm_rev and ttm_rev > 0 and px_map.get(d_end):
                            mcap_hist = shares * float(px_map[d_end])
                            vals[f"P/S Q{idx}"] = mcap_hist / float(ttm_rev)
                            vals[f"MCAP Q{idx}"] = mcap_hist
                            idx += 1
    except Exception as e:
        debug["yahoo_quarters_err"] = str(e)

    # --- 4) FMP backup för P/S (om saknas) + quarterly ratios ---
    try:
        if "P/S" not in vals:
            ps = fmp_ratios_ttm(tkr)
            if ps and ps > 0:
                vals["P/S"] = float(ps)
                debug["_ps_now_source"] = "FMP ratios-ttm"
        rqs = fmp_ratios_quarterly(tkr)
        # skriv Q1..Q4 bara där vi saknar
        for i in range(1, 5):
            key = f"P/S Q{i}"
            if key not in vals and rqs.get(i):
                vals[key] = float(rqs[i])
    except Exception as e:
        debug["fmp_err"] = str(e)

    return vals, debug
