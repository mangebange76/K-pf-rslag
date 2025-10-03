# stockapp/fetchers/fmp.py
"""
FMP fetcher – fria endpoints + mappning till appens kolumnnamn.

Gratis-endpoints:
- /v3/profile/{ticker}
- /v3/quote/{ticker}
- /v3/key-metrics-ttm/{ticker}?limit=1
- /v3/income-statement/{ticker}?period=annual&limit=1

Publikt API:
- get_all(ticker) -> dict               (endast app-nycklar; robust mot fel)
- get_all_verbose(ticker) -> (dict, list[str], list[str])  (för debug/logg)
- format_fetch_summary(source, fetched, warnings) -> str
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

__all__ = [
    "get_all",
    "get_all_verbose",
    "format_fetch_summary",
]

# ── Feltyp ──────────────────────────────────────────────────────────────────

class FMPError(RuntimeError):
    pass

# ── Hjälpare ────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # type: ignore
        return st.secrets["FMP_API_KEY"]
    except Exception:
        pass
    raise FMPError("FMP_API_KEY saknas. Lägg in den i env eller st.secrets.")

def _endpoint(path: str) -> str:
    return f"{DEFAULT_BASE}/{API_VERSION}/{path.lstrip('/')}"

def _req(url: str, params: dict[str, t.Any], max_retries: int = 3, timeout: int = 20) -> t.Any:
    backoff = 1.2
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 402:
                raise FMPError("402 Premium Endpoint – kräver betalplan.")
            if r.status_code == 429:
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            try:
                return r.json()
            except json.JSONDecodeError:
                raise FMPError(f"Kunde inte tolka JSON från {url}")
        except Exception as e:
            last_err = e
            time.sleep(backoff * attempt)
    if isinstance(last_err, FMPError):
        raise last_err
    raise FMPError(str(last_err) if last_err else "Okänt fel mot FMP")

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

def _to_millions(x: t.Any) -> t.Optional[float]:
    v = _safe_num(x)
    if v is None:
        return None
    return v / 1_000_000.0

def _round4(v: t.Any) -> t.Any:
    n = _safe_num(v)
    return round(n, 4) if n is not None else None

# ── Inhämtning (rådata) ────────────────────────────────────────────────────

def _fetch_raw(ticker: str) -> tuple[dict, list[str], list[str]]:
    api_key = _get_api_key()
    params = {"apikey": api_key}

    raw: dict[str, t.Any] = {}
    fetched: list[str] = []
    warns: list[str] = []

    symbol = (ticker or "").strip().upper()
    if not symbol:
        return {}, [], ["Tom ticker"]

    # PROFILE
    try:
        resp = _req(_endpoint(f"profile/{symbol}"), params)
        if isinstance(resp, list) and resp:
            p = resp[0]
            if (v := p.get("companyName") or p.get("company_name")):
                raw["FMP:Company Name"] = v; fetched.append("FMP:Company Name")
            if (v := p.get("currency")):
                raw["FMP:Currency"] = v; fetched.append("FMP:Currency")
            if (v := p.get("sector")):
                raw["FMP:Sector"] = v; fetched.append("FMP:Sector")
            if (v := p.get("industry")):
                raw["FMP:Industry"] = v; fetched.append("FMP:Industry")
            if (v := p.get("exchangeShortName") or p.get("exchange")):
                raw["FMP:Exchange"] = v; fetched.append("FMP:Exchange")
        else:
            warns.append("Profile: inga data (tom lista).")
    except FMPError as e:
        warns.append(f"Profile fel: {e}")

    # QUOTE
    try:
        resp = _req(_endpoint(f"quote/{symbol}"), params)
        if isinstance(resp, list) and resp:
            q = resp[0]
            if (v := _safe_num(q.get("price"))) is not None:
                raw["FMP:Price"] = v; fetched.append("FMP:Price")
            if (v := _safe_num(q.get("marketCap"))) is not None:
                raw["FMP:Market Cap"] = v; fetched.append("FMP:Market Cap")
        else:
            warns.append("Quote: inga data (tom lista).")
    except FMPError as e:
        warns.append(f"Quote fel: {e}")

    # KEY METRICS TTM
    try:
        resp = _req(_endpoint(f"key-metrics-ttm/{symbol}"), params | {"limit": 1})
        if isinstance(resp, list) and resp:
            km = resp[0]
            ps = km.get("priceToSalesRatioTTM") or km.get("priceToSalesTTM")
            if (v := _safe_num(ps)) is not None:
                raw["FMP:P/S TTM"] = v; fetched.append("FMP:P/S TTM")
            if (v := _safe_num(km.get("sharesOutstanding"))) is not None:
                raw["FMP:Shares Outstanding"] = v; fetched.append("FMP:Shares Outstanding")
        else:
            warns.append("Key-metrics TTM: inga data (tom lista).")
    except FMPError as e:
        warns.append(f"Key-metrics TTM fel: {e}")

    # INCOME STATEMENT (senaste års omsättning)
    try:
        resp = _req(_endpoint(f"income-statement/{symbol}"), params | {"period": "annual", "limit": 1})
        if isinstance(resp, list) and resp:
            inc = resp[0]
            if (v := _safe_num(inc.get("revenue"))) is not None:
                raw["FMP:Revenue (Annual)"] = v; fetched.append("FMP:Revenue (Annual)")
        else:
            warns.append("Income-statement: inga data (tom lista).")
    except FMPError as e:
        warns.append(f"Income-statement fel: {e}")

    # Fallback currency för US-börser
    if "FMP:Currency" not in raw:
        ex = raw.get("FMP:Exchange")
        if isinstance(ex, str) and ex.upper() in {"NASDAQ", "NYSE", "AMEX"}:
            raw["FMP:Currency"] = "USD"
            warns.append("Currency saknades i profile – antog USD (US-börs).")

    if not fetched:
        warns.append("FMP returnerade inga fält (kontrollera API-nyckel/plan/kvot).")

    return raw, fetched, warns

# ── Mappning till dina rubriker ────────────────────────────────────────────

def _map_to_app(raw: dict) -> dict:
    """
    Mappar till DINA kolumner. Lägger även aliasfält för kompatibilitet.
    Skalning till miljoner där det är rimligt.
    """
    m: dict[str, t.Any] = {}

    # Metadata
    if raw.get("FMP:Company Name"):
        m["Bolagsnamn"] = raw["FMP:Company Name"]
    if raw.get("FMP:Currency"):
        m["Valuta"] = raw["FMP:Currency"]
    if raw.get("FMP:Exchange"):
        m["Börs"] = raw["FMP:Exchange"]
    if raw.get("FMP:Sector"):
        m["Sektor"] = raw["FMP:Sector"]
    if raw.get("FMP:Industry"):
        m["Industri"] = raw["FMP:Industry"]
        # vissa views använder "Bransch"
        m["Bransch"] = raw["FMP:Industry"]

    # Pris/MCAP
    if (v := _safe_num(raw.get("FMP:Price"))) is not None:
        m["Kurs"] = v

    if (v := _safe_num(raw.get("FMP:Market Cap"))) is not None:
        # Din sheet har "Market Cap" (ej (M) som primär).
        m["Market Cap"] = v
        # lägg även en skalerad variant i miljoner (om du har en (M)-kolumn i andra vyer)
        m["Market Cap (M)"] = _round4(v / 1_000_000.0)

    # P/S
    if (v := _safe_num(raw.get("FMP:P/S TTM"))) is not None:
        # Din kolumn heter "P/S"; lägg även alias "P/S TTM" för bakåtkomp.
        m["P/S"] = _round4(v)
        m["P/S TTM"] = _round4(v)
        m["P/S (TTM, modell)"] = _round4(v)

    # Shares outstanding (till miljoner)
    if (v := _to_millions(raw.get("FMP:Shares Outstanding"))) is not None:
        m["Utestående aktier (milj.)"] = _round4(v)
        # lägg även TS-kolumn om du använder den i vissa vyer
        m["TS_Utestående aktier"] = _round4(v)

    # Revenue (senaste års) → "Omsättning i år (M)"
    if (v := _to_millions(raw.get("FMP:Revenue (Annual)"))) is not None:
        m["Omsättning i år (M)"] = _round4(v)
        # vissa kalkyler använder även "TS_Omsättning idag"
        m["TS_Omsättning idag"] = _round4(v)

    return {k: v for k, v in m.items() if v is not None}

# ── Publikt API ────────────────────────────────────────────────────────────

def get_all(ticker: str) -> dict:
    """Stabil: returnerar endast dict (inga exceptions bubblar upp)."""
    try:
        raw, _f, _w = _fetch_raw(ticker)
        return _map_to_app(raw)
    except Exception:
        return {}

def get_all_verbose(ticker: str) -> tuple[dict, list[str], list[str]]:
    """För debug/loggning i UI."""
    raw, fetched, warns = _fetch_raw(ticker)
    mapped = _map_to_app(raw)
    mapped_fields = list(mapped.keys())
    return mapped, mapped_fields, warns

# ── UI-hjälp ───────────────────────────────────────────────────────────────

def format_fetch_summary(source: str, fetched: list[str], warnings: list[str]) -> str:
    parts: list[str] = []
    if fetched:
        parts.append(f"{source}: Hämtade {len(fetched)} fält: " + ", ".join(fetched) + ".")
    else:
        parts.append(f"{source}: Hämtade 0 fält.")
    if warnings:
        parts.append("Varningar: " + " | ".join(warnings))
    return " ".join(parts)
