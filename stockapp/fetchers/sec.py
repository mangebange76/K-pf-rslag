# stockapp/fetchers/sec.py
# -*- coding: utf-8 -*-
"""
SEC/EDGAR-hämtare för amerikanska bolag.

Kräver *ingen* API-nyckel, men det är *viktigt* med en User-Agent.
Sätt gärna i st.secrets:
  SEC_USER_AGENT="kpf-app/1.0 (mailto:din@email)"

Publikt API:
    get_live_price(ticker) -> None (SEC saknar pris – vi lämnar None)
    fetch_ticker(ticker)   -> Dict[str, Any]

Returnerade nycklar (om tillgängliga):
    currency                -> "USD"
    shares_outstanding      -> float
    revenue_quarters        -> List[{"end": "YYYY-MM-DD", "value": float}]  # senaste 4 kvartal (M USD)
    cash_m                  -> float (miljoner USD)
    fcf_m                   -> float (miljoner USD, approx CFO - CapEx)
    dividends_paid_m        -> float (miljoner USD, >0 betyder utbetalningar)
    payout_ratio_cf_pct     -> float (%), approx (utdelningar / FCF) om FCF>0
    runway_quarters         -> float (hur många kvartal kassan räcker om FCF<0)

OBS:
 - SEC-data finns bara för US-rapporterande bolag.
 - Vi räknar om större belopp till miljoner USD (M).
 - P/S per kvartal beräknas *inte här*, eftersom SEC inte ger pris.
   Orchestratorn kan kombinera revenue_quarters med pris/market cap från Yahoo/FMP.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import math
import time
import requests
import streamlit as st
from datetime import datetime


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
_SEC_HEADERS = lambda: {
    "User-Agent": st.secrets.get("SEC_USER_AGENT", "kpf-app/1.0 (mailto:example@example.com)"),
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        # pröva str->float
        try:
            v = float(str(x).replace(",", "."))
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

def _parse_date(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

def _take_last_n_quarters(facts: List[Dict[str, Any]], n: int = 4) -> List[Dict[str, Any]]:
    """
    Tar de senaste n kvartalsvärdena (duration ~ kvartal) sorterat på 'end' (slutdatum).
    Filtrerar bort dubletter på slutdatum och icke-positiva värden.
    Returnerar lista med dict {"end": "YYYY-MM-DD", "value": float}
    """
    # sortera på end
    recs = []
    for f in facts:
        end = f.get("end")
        val = _to_float(f.get("val"))
        if not end or val is None:
            continue
        # SEC kan ha olika skalor – antar USD baserat på enheten senare
        recs.append({"end": end, "value": float(val)})

    # sortera nyast först på slutdatum
    recs = [r for r in recs if _parse_date(r["end"]) is not None]
    recs.sort(key=lambda r: _parse_date(r["end"]) or datetime.min, reverse=True)

    # dedupe på end-datum
    seen = set()
    uniq = []
    for r in recs:
        if r["end"] in seen:
            continue
        seen.add(r["end"])
        # bara vettiga värden
        if r["value"] is not None:
            uniq.append(r)

    return uniq[:n]


# ------------------------------------------------------------
# CIK-lookup (cache i session)
# ------------------------------------------------------------
def _sec_lookup_cik(ticker: str) -> Optional[str]:
    """
    Hämta CIK för ett US-ticker via SEC filen company_tickers.json.
    Cache:ar i sessionen för denna körning.
    """
    if "_sec_ticker2cik" not in st.session_state:
        st.session_state["_sec_ticker2cik"] = {}

    t = str(ticker).upper().strip()

    if t in st.session_state["_sec_ticker2cik"]:
        return st.session_state["_sec_ticker2cik"][t]

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(url, headers=_SEC_HEADERS(), timeout=30)
        if r.status_code != 200:
            return None
        js = r.json()  # dict med nycklar "0","1",...
        # Bygg upp karta
        for _, row in js.items():
            tk = str(row.get("ticker", "")).upper().strip()
            cik = str(row.get("cik_str", "")).strip()
            if tk:
                st.session_state["_sec_ticker2cik"][tk] = cik.zfill(10) if cik else None
    except Exception:
        return None

    return st.session_state["_sec_ticker2cik"].get(t)


# ------------------------------------------------------------
# CompanyFacts
# ------------------------------------------------------------
def _sec_companyfacts(cik: str) -> Optional[Dict[str, Any]]:
    """
    Hämta companyfacts som JSON.
    """
    if not cik:
        return None
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    try:
        # vänligt tempo
        time.sleep(0.25)
        r = requests.get(url, headers=_SEC_HEADERS(), timeout=45)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _pick_units(facts: Dict[str, Any], tag: str, units_order: List[str]) -> List[Dict[str, Any]]:
    """
    Plocka ut listelementen för en viss 'tag' (ex. 'us-gaap:Revenues'), i första förekommande enhet från units_order.
    """
    if not facts:
        return []
    parts = tag.split(":")
    if len(parts) == 2:
        ns, nm = parts
    else:
        ns, nm = "us-gaap", tag
    root = facts.get("facts", {}).get(ns, {}).get(nm, {})
    units = root.get("units", {})
    for u in units_order:
        arr = units.get(u)
        if isinstance(arr, list) and arr:
            return arr
    return []


# ------------------------------------------------------------
# Publika API
# ------------------------------------------------------------
def get_live_price(ticker: str) -> None:
    """
    SEC har ingen prisfeed. Håll signaturen kompatibel med andra fetchers.
    """
    return None


def fetch_ticker(ticker: str) -> Dict[str, Any]:
    """
    Hämtar normaliserade nycklar från SEC:
      - currency="USD"
      - shares_outstanding
      - revenue_quarters (senaste 4 kvartalen, M USD)
      - cash_m
      - fcf_m (≈ CFO - CapEx, M USD)
      - dividends_paid_m (M USD, positivt tal)
      - payout_ratio_cf_pct (om FCF>0 och utdelningar>0)
      - runway_quarters (om FCF<0)
    """
    out: Dict[str, Any] = {}
    tkr = str(ticker).upper().strip()
    cik = _sec_lookup_cik(tkr)
    if not cik:
        return out  # ingen CIK => kan inte hämta

    facts = _sec_companyfacts(cik)
    if not facts:
        return out

    out["currency"] = "USD"  # SEC rapporterar i USD i dessa endpoints

    # ---------------------------
    # Shares Outstanding (aktier)
    # ---------------------------
    # Vanliga taggar: "EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"
    shares_units = _pick_units(facts, "us-gaap:EntityCommonStockSharesOutstanding", ["shares"])
    if not shares_units:
        shares_units = _pick_units(facts, "us-gaap:CommonStockSharesOutstanding", ["shares"])

    shares_val = None
    if shares_units:
        # ta senaste "instant" (de har 'fy'/'fp' etc — vi väljer nyaste end)
        shares_units = sorted(
            [u for u in shares_units if "end" in u and _to_float(u.get("val"))],
            key=lambda x: _parse_date(x["end"]) or datetime.min,
            reverse=True,
        )
        if shares_units:
            shares_val = _to_float(shares_units[0].get("val"))
            if shares_val and shares_val > 0:
                out["shares_outstanding"] = float(shares_val)

    # ---------------------------
    # Revenue (kvartal)
    # ---------------------------
    # Vanliga taggar: SalesRevenueNet, Revenues, RevenueFromContractWithCustomerExcludingAssessedTax
    rev_units = []
    for tag in ("us-gaap:SalesRevenueNet",
                "us-gaap:Revenues",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"):
        rev_units = _pick_units(facts, tag, ["USD"])
        if rev_units:
            break

    rev_q = []
    if rev_units:
        # filtrera till duration runt kvartal (SEC har ibland 'frame' typ "CY2024Q4")
        q_candidates = []
        for u in rev_units:
            frame = u.get("frame", "")
            # acceptera Qn-frame eller duration ~ < 100 dagar / ~90-95
            dur = None
            if "start" in u and "end" in u:
                try:
                    s = _parse_date(u["start"])
                    e = _parse_date(u["end"])
                    if s and e:
                        dur = (e - s).days
                except Exception:
                    pass

            if (isinstance(frame, str) and "Q" in frame) or (dur is not None and 60 <= dur <= 110):
                # ta med
                q_candidates.append(u)

        last4 = _take_last_n_quarters(q_candidates, n=4)
        # konvertera till miljoner
        for r in last4:
            rev_q.append({"end": r["end"], "value": float(r["value"]) / 1e6})

        if rev_q:
            out["revenue_quarters"] = rev_q

    # ---------------------------
    # Cash (år)
    # ---------------------------
    # CashAndCashEquivalentsAtCarryingValue eller CashAndShortTermInvestments
    cash_units = _pick_units(facts, "us-gaap:CashAndCashEquivalentsAtCarryingValue", ["USD"])
    if not cash_units:
        cash_units = _pick_units(facts, "us-gaap:CashAndShortTermInvestments", ["USD"])
    if cash_units:
        cash_units = sorted(
            [u for u in cash_units if "end" in u and _to_float(u.get("val"))],
            key=lambda x: _parse_date(x["end"]) or datetime.min,
            reverse=True,
        )
        if cash_units:
            cash_val = _to_float(cash_units[0].get("val"))
            if cash_val is not None:
                out["cash_m"] = float(cash_val) / 1e6

    # ---------------------------
    # FCF (≈ CFO - CapEx) + Dividends (år)
    # ---------------------------
    cfo_units = _pick_units(facts, "us-gaap:NetCashProvidedByUsedInOperatingActivities", ["USD"])
    capex_units = _pick_units(facts, "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", ["USD"])
    div_units = _pick_units(facts, "us-gaap:PaymentsOfDividends", ["USD"])

    def _last_annual(units: List[Dict[str, Any]]) -> Optional[float]:
        if not units:
            return None
        # välj senaste som *inte* är quarterly frame (saknar "Q" i frame & duration ~ 300-400 dagar)
        candidates = []
        for u in units:
            end = u.get("end")
            s = u.get("start")
            val = _to_float(u.get("val"))
            if not end or val is None:
                continue
            # rensa bort Q-frames
            frame = u.get("frame", "")
            if isinstance(frame, str) and ("Q" in frame or "H" in frame):
                continue
            # kontrollera duration ~ årsperiod
            dur_ok = False
            if s:
                sd = _parse_date(s)
                ed = _parse_date(end)
                if sd and ed:
                    days = (ed - sd).days
                    # 300..400 dagar
                    if 300 <= days <= 400:
                        dur_ok = True
            # om vi inte har duration, låt ändå candidate passera (bättre något än inget)
            if dur_ok or not s:
                candidates.append(u)

        if not candidates:
            candidates = units

        candidates = sorted(
            [u for u in candidates if "end" in u and _to_float(u.get("val")) is not None],
            key=lambda x: _parse_date(x["end"]) or datetime.min,
            reverse=True,
        )
        if not candidates:
            return None
        return _to_float(candidates[0].get("val"))

    cfo = _last_annual(cfo_units)
    capex = _last_annual(capex_units)
    if cfo is not None and capex is not None:
        fcf = float(cfo) - float(capex)  # CFO - CapEx
        out["fcf_m"] = float(fcf) / 1e6

    if div_units:
        dv = _last_annual(div_units)
        if dv is not None:
            out["dividends_paid_m"] = abs(float(dv)) / 1e6  # positivt tal

    # payout-ratio på CF-basis (om FCF>0)
    if out.get("fcf_m") and out["fcf_m"] > 0 and out.get("dividends_paid_m") and out["dividends_paid_m"] > 0:
        out["payout_ratio_cf_pct"] = float(out["dividends_paid_m"] / out["fcf_m"] * 100.0)

    # runway om negativt FCF och kassa finns
    if out.get("fcf_m") is not None and out["fcf_m"] < 0 and out.get("cash_m") is not None and out["cash_m"] > 0:
        burn_per_year = abs(float(out["fcf_m"]))  # miljoner/år
        if burn_per_year > 0:
            out["runway_quarters"] = float(4.0 * (out["cash_m"] / burn_per_year))

    return out
