# stockapp/fetchers/fmp.py
"""
FMP fetcher – gratis-endpoints, robust felhantering och tydlig fältmapping.

Hämtar bara från öppna (free) endpoints:
- /v3/profile/{ticker}
- /v3/quote/{ticker}
- /v3/key-metrics-ttm/{ticker}?limit=1
- /v3/income-statement/{ticker}?period=annual&limit=1

Returnerar (dict, fetched_fields:list[str], warnings:list[str]).
"""

from __future__ import annotations
import os
import time
import json
import math
import typing as t
import requests

DEFAULT_BASE = "https://financialmodelingprep.com/api"
API_VERSION = "v3"

# ---- Hjälpare --------------------------------------------------------------

class FMPError(RuntimeError):
    pass

def _get_api_key() -> str:
    # Försök läsa från miljövariabel först; fall tillbaka till Streamlit secrets om möjligt.
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # type: ignore
        return st.secrets["FMP_API_KEY"]
    except Exception:
        pass
    raise FMPError("FMP_API_KEY saknas. Lägg in den i miljövariabler eller st.secrets.")

def _req(url: str, params: dict[str, t.Any], max_retries: int = 3, timeout: int = 20) -> t.Any:
    """GET med enkel backoff och särskild hantering för 402/429."""
    backoff = 1.2
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 402:
                # Premium endpoint – förklara tydligt vad som hände.
                raise FMPError("FMP svarade 402 (Premium Endpoint). Byt endpoint eller uppgradera plan.")
            if r.status_code == 429:
                # Rate limit – backoff och försök igen
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            # FMP returnerar ofta tomma listor [] vid “no data”.
            try:
                return r.json()
            except json.JSONDecodeError:
                raise FMPError(f"Kunde inte tolka JSON från {url}")
        except Exception as e:
            last_err = e
            time.sleep(backoff * attempt)
    # Sista försöket misslyckades
    if isinstance(last_err, FMPError):
        raise last_err
    raise FMPError(str(last_err) if last_err else "Okänt fel mot FMP")

def _endpoint(path: str) -> str:
    return f"{DEFAULT_BASE}/{API_VERSION}/{path.lstrip('/')}"

def _safe_num(x: t.Any) -> t.Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

# ---- Publika funktioner ----------------------------------------------------

def fetch_fmp(ticker: str) -> tuple[dict, list[str], list[str]]:
    """
    Hämtar data för givna 'ticker' från FMP gratis-endpoints.
    Returnerar: (data_dict, fetched_fields, warnings)
    """
    api_key = _get_api_key()
    params = {"apikey": api_key}

    data: dict[str, t.Any] = {}
    fetched: list[str] = []
    warnings: list[str] = []

    # Normalisera ticker (NVDA ska vara NVDA)
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return {}, [], ["Tom ticker"]

    # 1) PROFILE – namn, valuta (ibland), sektor/industri m.m.
    try:
        resp = _req(_endpoint(f"profile/{symbol}"), params)
        if isinstance(resp, list) and resp:
            prof = resp[0]
            name = prof.get("companyName") or prof.get("company_name")
            currency = prof.get("currency")  # kan vara None
            sector = prof.get("sector")
            industry = prof.get("industry")
            exchange = prof.get("exchangeShortName") or prof.get("exchange")

            if name:
                data["FMP:Company Name"] = name
                fetched.append("FMP:Company Name")
            if currency:
                data["FMP:Currency"] = currency
                fetched.append("FMP:Currency")
            if sector:
                data["FMP:Sector"] = sector
                fetched.append("FMP:Sector")
            if industry:
                data["FMP:Industry"] = industry
                fetched.append("FMP:Industry")
            if exchange:
                data["FMP:Exchange"] = exchange
                fetched.append("FMP:Exchange")
        else:
            warnings.append("Profile: inga data (tom lista)")
    except FMPError as e:
        warnings.append(f"Profile fel: {e}")

    # 2) QUOTE – pris, market cap etc (gratis)
    try:
        resp = _req(_endpoint(f"quote/{symbol}"), params)
        if isinstance(resp, list) and resp:
            q = resp[0]
            price = _safe_num(q.get("price"))
            market_cap = _safe_num(q.get("marketCap"))
            changes_pct = _safe_num(q.get("changesPercentage"))

            if price is not None:
                data["FMP:Price"] = price
                fetched.append("FMP:Price")
            if market_cap is not None:
                data["FMP:Market Cap"] = market_cap
                fetched.append("FMP:Market Cap")
            if changes_pct is not None:
                data["FMP:Change %"] = changes_pct
                fetched.append("FMP:Change %")
        else:
            warnings.append("Quote: inga data (tom lista)")
    except FMPError as e:
        warnings.append(f"Quote fel: {e}")

    # 3) KEY METRICS TTM – P/S TTM, revenue per share TTM, shares outstanding (ibland)
    try:
        resp = _req(_endpoint(f"key-metrics-ttm/{symbol}"), params | {"limit": 1})
        if isinstance(resp, list) and resp:
            km = resp[0]
            ps_ttm = _safe_num(km.get("priceToSalesRatioTTM") or km.get("priceToSalesTTM"))
            rps_ttm = _safe_num(km.get("revenuePerShareTTM"))
            shares = _safe_num(km.get("sharesOutstanding"))

            if ps_ttm is not None:
                data["FMP:P/S TTM"] = ps_ttm
                fetched.append("FMP:P/S TTM")
            if rps_ttm is not None:
                data["FMP:Revenue/Share TTM"] = rps_ttm
                fetched.append("FMP:Revenue/Share TTM")
            if shares is not None:
                data["FMP:Shares Outstanding"] = shares
                fetched.append("FMP:Shares Outstanding")
        else:
            warnings.append("Key-metrics TTM: inga data (tom lista)")
    except FMPError as e:
        # Vissa konton får 402 här – då noterar vi det och går vidare.
        warnings.append(f"Key-metrics TTM fel: {e}")

    # 4) INCOME STATEMENT – revenue (TTM saknas ofta på free; tar senaste års-värde)
    try:
        resp = _req(_endpoint(f"income-statement/{symbol}"), params | {"period": "annual", "limit": 1})
        if isinstance(resp, list) and resp:
            inc = resp[0]
            revenue = _safe_num(inc.get("revenue"))
            if revenue is not None:
                data["FMP:Revenue (Annual)"] = revenue
                fetched.append("FMP:Revenue (Annual)")
        else:
            warnings.append("Income-statement: inga data (tom lista)")
    except FMPError as e:
        warnings.append(f"Income-statement fel: {e}")

    # Sanity: Om vi inte fick currency från profile, gissa USD för US-exchanges (bättre än blankt)
    if "FMP:Currency" not in data:
        ex = data.get("FMP:Exchange")
        if isinstance(ex, str) and ex.upper() in {"NASDAQ", "NYSE", "AMEX"}:
            data["FMP:Currency"] = "USD"
            warnings.append("Currency saknades i profile – antog USD (US-börs).")

    # Särskilt fall: Om *allt* blev 0 fält – ge tydlig varning
    if not fetched:
        warnings.append(
            "FMP returnerade inga fält. Vanliga orsaker: saknad/fel API-nyckel, 402 (premium-endpoint), eller tomma svar."
        )

    return data, fetched, warnings


# ---- Hjälpfunktion för UI-loggning ----------------------------------------

def format_fetch_summary(source: str, fetched: list[str], warnings: list[str]) -> str:
    """
    Gör en läsbar summering till UI-loggen, ex:
    'FMP: Hämtade 3 fält: FMP:Price, FMP:Market Cap, FMP:P/S TTM. Varningar: …'
    """
    parts: list[str] = []
    if fetched:
        parts.append(f"{source}: Hämtade {len(fetched)} fält: " + ", ".join(fetched) + ".")
    else:
        parts.append(f"{source}: Hämtade 0 fält.")
    if warnings:
        parts.append("Varningar: " + " | ".join(warnings))
    return " ".join(parts)
