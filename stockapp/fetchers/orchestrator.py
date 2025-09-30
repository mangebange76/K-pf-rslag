# -*- coding: utf-8 -*-
"""
Orchestrator: sammanfogar Yahoo + SEC (+ valfritt FMP) till en enhetlig dict.

Publik:
    fetch_all(ticker: str) -> (vals: dict, meta: dict)

vals innehåller fält anpassade för din databas/app:
  - "Bolagsnamn", "Valuta", "Aktuell kurs"
  - "Utestående aktier" (miljoner)
  - "Market Cap"
  - "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"
  - "_rev_quarterly": lista på kvartalsintäkter [{end, value, unit}] (nyast→äldst)
  - "_ttm": lista [(end_date_str, ttm_value_px_ccy)] (nyast→äldst) i prisvaluta
  - ev. pass-through från Yahoo: "Årlig utdelning", "CAGR 5 år (%)", "Sektor", "Bransch"

meta innehåller:
  - "sources": vilka källor som användes
  - "shares_source": vilken källa som valdes för shares
  - "notes": fria kommentarer
  - "errors": ev fel/undantag som inträffat

OBS:
  - Orchestrator hämtar INTE "Omsättning i år/nästa år" (ska matas in manuellt enligt ditt önskemål).
  - SEC används främst för robusta kvartalsintäkter (inkl dec/jan) och aktier (fallback).
  - P/S-historik Q1–Q4 räknas med *samma* antal aktier (senaste kända) och historiska priser (Yahoo) “på eller strax före” periodens slutdatum.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
from datetime import date, datetime, timedelta
import math

import requests
import yfinance as yf

# --- Importera käll-fetchers ---
from .yahoo import fetch_yahoo  # måste finnas
from .sec import fetch_sec      # måste finnas

# FMP är valfri. Om modulen inte finns hoppar vi den.
try:
    from .fmp import fetch_fmp  # optional
    _HAS_FMP = True
except Exception:
    _HAS_FMP = False


# ============================== Hjälpare ==============================

def _parse_iso(d: str) -> Optional[date]:
    if not d:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dZ"):
        try:
            return datetime.strptime(d.replace("Z", ""), "%Y-%m-%d").date()
        except Exception:
            pass
    try:
        # sista utväg
        return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _ttm_windows(rows: List[Dict[str, Any]], need: int = 5) -> List[Tuple[str, float]]:
    """
    rows = [{"end": "YYYY-MM-DD", "value": float, "unit": "USD"}, ...] nyast→äldst
    returnerar upp till 'need' TTM-summor: [(end_date_str, ttm_sum), ...]
    """
    if not rows or len(rows) < 4:
        return []
    out: List[Tuple[str, float]] = []
    for i in range(0, min(need, len(rows) - 3)):
        ttm = 0.0
        for j in range(i, i+4):
            ttm += float(rows[j]["value"] or 0.0)
        out.append((rows[i]["end"], float(ttm)))
    return out


def _fx_rate(base: str, quote: str, timeout: int = 12) -> float:
    """
    Enkel FX via Frankfurter -> exchangerate.host som fallback.
    """
    base = (base or "").upper()
    quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    # Frankfurter
    try:
        r = requests.get("https://api.frankfurter.app/latest",
                         params={"from": base, "to": quote}, timeout=timeout)
        if r.status_code == 200:
            j = r.json() or {}
            v = (j.get("rates") or {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    # exchangerate.host
    try:
        r = requests.get("https://api.exchangerate.host/latest",
                         params={"base": base, "symbols": quote}, timeout=timeout)
        if r.status_code == 200:
            j = r.json() or {}
            v = (j.get("rates") or {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0


def _historical_prices_yf(ticker: str, dates: List[date]) -> Dict[date, float]:
    """
    Hämtar dagliga 'Close' från yfinance för ett intervall som täcker alla datum,
    och väljer priset på eller närmast FÖRE respektive datum (stänger luckor vid helg/helgdag).
    """
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx = [i.date() for i in hist.index]
        closes = list(hist["Close"].values)
        out: Dict[date, float] = {}
        for d in dates:
            px = None
            # gå baklänges för att hitta närmast före/likamed
            for j in range(len(idx)-1, -1, -1):
                if idx[j] <= d:
                    try:
                        px = float(closes[j])
                    except Exception:
                        px = None
                    break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


# ============================== Orchestrator ==============================

def fetch_all(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {"sources": [], "errors": [], "notes": []}
    out: Dict[str, Any] = {}

    # ---------- Yahoo (för kurs/valuta/MCAP/namn/sektor m.m.) ----------
    try:
        ydat, ysc, ysrc = fetch_yahoo(ticker)
        meta["sources"].append(f"{ysrc}:{ysc}")
    except Exception as e:
        ydat, ysc, ysrc = {}, 599, "Yahoo"
        meta["errors"].append(f"Yahoo error: {e}")

    # ---------- SEC (kvartalsintäkter + shares fallback) ----------
    try:
        sdat, ssc, ssrc = fetch_sec(ticker)
        meta["sources"].append(f"{ssrc}:{ssc}")
    except Exception as e:
        sdat, ssc, ssrc = {}, 599, "SEC"
        meta["errors"].append(f"SEC error: {e}")

    # ---------- FMP (valfritt) ----------
    if _HAS_FMP:
        try:
            fdat, fsc, fsrc = fetch_fmp(ticker)
            meta["sources"].append(f"{fsrc}:{fsc}")
        except Exception as e:
            fdat, fsc, fsrc = {}, 599, "FMP"
            meta["errors"].append(f"FMP error: {e}")
    else:
        fdat, fsc, fsrc = {}, 0, "FMP(N/A)"

    # ================== Grundfält från Yahoo ==================
    out["Bolagsnamn"] = ydat.get("Namn") or ydat.get("Bolagsnamn") or ""
    out["Valuta"] = (ydat.get("Valuta") or "USD").upper()
    out["Aktuell kurs"] = _safe_float(ydat.get("Senaste kurs") or ydat.get("Aktuell kurs"))
    out["Market Cap"] = _safe_float(ydat.get("Market Cap"))
    out["Årlig utdelning"] = _safe_float(ydat.get("Dividend (ttm)") or ydat.get("Årlig utdelning"))
    out["CAGR 5 år (%)"] = _safe_float(ydat.get("CAGR 5 år (%)"))
    out["Sektor"] = ydat.get("Sektor") or ""
    out["Bransch"] = ydat.get("Bransch") or ""

    px_ccy = out["Valuta"]

    # ================== Utestående aktier (välja bästa) ==================
    shares_src = "unknown"
    implied = 0.0
    if out["Market Cap"] > 0 and out["Aktuell kurs"] > 0:
        implied = out["Market Cap"] / out["Aktuell kurs"]

    if implied > 0:
        shares = implied
        shares_src = "Yahoo implied (mcap/price)"
    else:
        # yahoo explicit shares?
        y_sh = _safe_float(ydat.get("Utestående aktier (milj.)")) * 1e6 or _safe_float(ydat.get("Utestående aktier"))
        if y_sh > 0:
            shares = y_sh
            shares_src = "Yahoo shares"
        else:
            # SEC robust
            s_sh = _safe_float(sdat.get("Utestående aktier (milj.)")) * 1e6
            if s_sh > 0:
                shares = s_sh
                shares_src = "SEC robust shares"
            else:
                # FMP fallback om finns
                f_sh = _safe_float(fdat.get("Utestående aktier (milj.)")) * 1e6 or _safe_float(fdat.get("shares"))
                if f_sh > 0:
                    shares = f_sh
                    shares_src = "FMP shares"
                else:
                    shares = 0.0

    meta["shares_source"] = shares_src
    out["Utestående aktier"] = round(shares / 1e6, 6) if shares > 0 else 0.0  # i miljoner

    # ================== Kvartalsintäkter (SEC prioriteras) ==================
    rev_rows = sdat.get("_SEC_rev_quarterly") or []
    rev_unit = (sdat.get("_SEC_rev_unit") or "").upper()
    if not rev_rows:
        # Fallback: Yahoo quarterly revenues om din yahoo-fetcher exponerar dem
        y_rev = ydat.get("_YF_rev_quarterly") or []
        # för kompatibilitet, mappa om
        rev_rows = [{"end": r.get("end"), "value": _safe_float(r.get("value")), "unit": (ydat.get("financialCurrency") or px_ccy)} for r in y_rev]
        rev_unit = (ydat.get("financialCurrency") or px_ccy).upper()

    # städa + sortera rev_rows nyast→äldst och deduplicera per end
    tmp = {}
    for r in rev_rows:
        d = _parse_iso(str(r.get("end", "")))
        v = _safe_float(r.get("value"))
        if d and v > 0:
            tmp[d] = v
    rev_pairs = sorted(tmp.items(), key=lambda t: t[0], reverse=True)
    rev_rows = [{"end": d.strftime("%Y-%m-%d"), "value": v, "unit": rev_unit} for (d, v) in rev_pairs]
    out["_rev_quarterly"] = rev_rows  # spara rådata för debug

    # ================== Bygg TTM i prisvaluta ==================
    ttm_list = _ttm_windows(rev_rows, need=5)  # [(end_str, ttm_value_native)]
    if ttm_list:
        # valuta-konvertering om behövs
        if rev_unit and rev_unit != px_ccy:
            fx = _fx_rate(rev_unit, px_ccy) or 1.0
        else:
            fx = 1.0
        ttm_px = [(d, float(ttm) * fx) for (d, ttm) in ttm_list]
    else:
        ttm_px = []
    out["_ttm"] = ttm_px

    # ================== P/S (nu) ==================
    ps_now = 0.0
    if out["Market Cap"] > 0 and ttm_px:
        ltm = _safe_float(ttm_px[0][1])
        if ltm > 0:
            ps_now = out["Market Cap"] / ltm
    out["P/S"] = round(ps_now, 4) if ps_now > 0 else 0.0

    # ================== P/S Q1–Q4 (historik) ==================
    out["P/S Q1"] = 0.0
    out["P/S Q2"] = 0.0
    out["P/S Q3"] = 0.0
    out["P/S Q4"] = 0.0

    if shares > 0 and ttm_px:
        # vilka kvartalsslut?
        q_dates: List[date] = []
        for (ds, _) in ttm_px[:4]:
            dd = _parse_iso(ds)
            if dd:
                q_dates.append(dd)
        # hämta historiska priser
        px_map = _historical_prices_yf(ticker, q_dates)
        for i, (ds, ttm_v) in enumerate(ttm_px[:4], start=1):
            dd = _parse_iso(ds)
            if not dd:
                continue
            p = _safe_float(px_map.get(dd))
            if p > 0 and ttm_v > 0:
                mcap_hist = shares * p
                ps_hist = mcap_hist / float(ttm_v)
                out[f"P/S Q{i}"] = round(ps_hist, 4)

    # ================== Avrunda vissa fält snyggt ==================
    if out["Aktuell kurs"] > 0:
        out["Aktuell kurs"] = float(f"{out['Aktuell kurs']:.6g}")
    if out["Market Cap"] > 0:
        out["Market Cap"] = float(f"{out['Market Cap']:.6g}")

    # ================== Meta-kommentarer ==================
    if not ttm_px:
        meta["notes"].append("TTM kunde inte räknas (saknar 4 kvartal).")
    if out["P/S"] == 0.0:
        meta["notes"].append("P/S nu kunde inte räknas (saknar mcap/ttm).")

    return out, meta
