# stockapp/fetchers/fmp.py
"""
FMP fetcher – fria endpoints + mappning till appens svenska kolumnnamn.

Endpoints (gratis):
- /v3/profile/{ticker}
- /v3/quote/{ticker}
- /v3/key-metrics-ttm/{ticker}?limit=1
- /v3/income-statement/{ticker}?period=annual&limit=1

Publikt API:
- get_all(ticker) -> dict                          (ENBART app-nycklar)
- get_all_verbose(ticker) -> (dict, list, list)    (app-nycklar, fetched_fields, warnings)
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

def _round4(x: t.Any) -> t.Any:
    v = _safe_num(x)
    return round(v, 4) if v is not None else None

# ── Kärninhämtning ─────────────────────────────────────────────────────────
def _fetch_raw_fmp(ticker: str) -> tuple[dict, list[str], list[str]]:
    """
    Hämtar rådata från FMP och returnerar nycklar med FMP-prefix.
    Return: (raw_dict, fetched_fields, warnings)
    """
    api_key = _get_api_key()
    params = {"apikey": api_key}

    raw: dict[str, t.Any] = {}
    fetched: list[str] = []
    warnings: list[str] = []

    symbol = (ticker or "").strip().upper()
    if not symbol:
        return {}, [], ["Tom ticker"]

    # PROFILE
    try:
        resp = _req(_endpoint(f"profile/{symbol}"), params)
        if isinstance(resp, list) and resp:
            p = resp[0]
            if (n := p.get("companyName") or p.get("company_name")):
                raw["FMP:Company Name"] = n; fetched.append("FMP:Company Name")
            if (c := p.get("currency")):
                raw["FMP:Currency"] = c; fetched.append("FMP:Currency")
            if (s := p.get("sector")):
                raw["FMP:Sector"] = s; fetched.append("FMP:Sector")
            if (i := p.get("industry")):
                raw["FMP:Industry"] = i; fetched.append("FMP:Industry")
            if (ex := p.get("exchangeShortName") or p.get("exchange")):
                raw["FMP:Exchange"] = ex; fetched.append("FMP:Exchange")
        else:
            warnings.append("Profile: inga data (tom lista).")
    except FMPError as e:
        warnings.append(f"Profile fel: {e}")

    # QUOTE
    try:
        resp = _req(_endpoint(f"quote/{symbol}"), params)
        if isinstance(resp, list) and resp:
            q = resp[0]
            if (v := _safe_num(q.get("price"))) is not None:
                raw["FMP:Price"] = v; fetched.append("FMP:Price")
            if (v := _safe_num(q.get("marketCap"))) is not None:
                raw["FMP:Market Cap"] = v; fetched.append("FMP:Market Cap")
            if (v := _safe_num(q.get("changesPercentage"))) is not None:
                raw["FMP:Change %"] = v; fetched.append("FMP:Change %")
        else:
            warnings.append("Quote: inga data (tom lista).")
    except FMPError as e:
        warnings.append(f"Quote fel: {e}")

    # KEY-METRICS TTM
    try:
        resp = _req(_endpoint(f"key-metrics-ttm/{symbol}"), params | {"limit": 1})
        if isinstance(resp, list) and resp:
            km = resp[0]
            ps = km.get("priceToSalesRatioTTM") or km.get("priceToSalesTTM")
            if (v := _safe_num(ps)) is not None:
                raw["FMP:P/S TTM"] = v; fetched.append("FMP:P/S TTM")
            if (v := _safe_num(km.get("revenuePerShareTTM"))) is not None:
                raw["FMP:Revenue/Share TTM"] = v; fetched.append("FMP:Revenue/Share TTM")
            if (v := _safe_num(km.get("sharesOutstanding"))) is not None:
                raw["FMP:Shares Outstanding"] = v; fetched.append("FMP:Shares Outstanding")
        else:
            warnings.append("Key-metrics TTM: inga data (tom lista).")
    except FMPError as e:
        warnings.append(f"Key-metrics TTM fel: {e}")

    # INCOME STATEMENT (senaste årsrevenue)
    try:
        resp = _req(_endpoint(f"income-statement/{symbol}"), params | {"period": "annual", "limit": 1})
        if isinstance(resp, list) and resp:
            inc = resp[0]
            if (v := _safe_num(inc.get("revenue"))) is not None:
                raw["FMP:Revenue (Annual)"] = v; fetched.append("FMP:Revenue (Annual)")
        else:
            warnings.append("Income-statement: inga data (tom lista).")
    except FMPError as e:
        warnings.append(f"Income-statement fel: {e}")

    # Fallback currency för US-börser
    if "FMP:Currency" not in raw:
        ex = raw.get("FMP:Exchange")
        if isinstance(ex, str) and ex.upper() in {"NASDAQ", "NYSE", "AMEX"}:
            raw["FMP:Currency"] = "USD"
            warnings.append("Currency saknades i profile – antog USD (US-börs).")

    if not fetched:
        warnings.append("FMP returnerade inga fält (kontrollera API-nyckel/plan/kvot).")

    return raw, fetched, warnings

# ── Mappning till appens kolumner ──────────────────────────────────────────
def _map_to_app_fields(raw: dict) -> dict:
    """
    Konverterar FMP:* nycklar till appens svenska kolumnnamn.
    Skalning:
      - Market Cap -> 'Market Cap (M)' i miljoner
      - Shares Outstanding -> 'Utestående aktier (milj.)' i miljoner
      - Revenue (Annual) -> 'Omsättning (M)' i miljoner
    """
    mapped: dict[str, t.Any] = {}

    # Namn / metadata
    if raw.get("FMP:Company Name"):
        mapped["Bolagsnamn"] = raw["FMP:Company Name"]
    if raw.get("FMP:Currency"):
        mapped["Valuta"] = raw["FMP:Currency"]
    if raw.get("FMP:Exchange"):
        mapped["Börs"] = raw["FMP:Exchange"]
    if raw.get("FMP:Sector"):
        mapped["Sektor"] = raw["FMP:Sector"]
    if raw.get("FMP:Industry"):
        # Kolumnen kan heta 'Bransch' eller 'Industri' i din sheet – exportera båda
        mapped["Bransch"] = raw["FMP:Industry"]
        mapped["Industri"] = raw["FMP:Industry"]

    # Pris / marknad
    if (v := _safe_num(raw.get("FMP:Price"))) is not None:
        mapped["Kurs"] = v
    if (v := _to_millions(raw.get("FMP:Market Cap"))) is not None:
        mapped["Market Cap (M)"] = _round4(v)
        # Ibland har man även en kolumn utan (M); lägg till båda för säkerhets skull
        mapped["Market Cap"] = raw.get("FMP:Market Cap")
    if (v := _safe_num(raw.get("FMP:Change %"))) is not None:
        mapped["Förändring %"] = _round4(v)

    # Nyckeltal
    if (v := _safe_num(raw.get("FMP:P/S TTM"))) is not None:
        mapped["P/S TTM"] = _round4(v)
    if (v := _to_millions(raw.get("FMP:Shares Outstanding"))) is not None:
        mapped["Utestående aktier (milj.)"] = _round4(v)
    if (v := _to_millions(raw.get("FMP:Revenue (Annual)"))) is not None:
        mapped["Omsättning (M)"] = _round4(v)
        # Om du har separata kolumner för 'Omsättning i år (M)'
        mapped["Omsättning i år (M)"] = _round4(v)

    return {k: v for k, v in mapped.items() if v is not None}

# ── Publikt API ────────────────────────────────────────────────────────────
def get_all(ticker: str) -> dict:
    """
    Returnerar ENBART en dict med appens svenska kolumnnamn.
    Fångar fel och returnerar {} vid problem, så UI inte kraschar.
    """
    try:
        raw, _fetched, _warn = _fetch_raw_fmp(ticker)
        return _map_to_app_fields(raw)
    except Exception:
        return {}

def get_all_verbose(ticker: str) -> tuple[dict, list[str], list[str]]:
    """
    Som get_all men behåller fetched_fields + warnings för loggning.
    """
    raw, fetched, warnings = _fetch_raw_fmp(ticker)
    mapped = _map_to_app_fields(raw)
    # Markera vilka fält som faktiskt mappades till appens nycklar
    mapped_fields = list(mapped.keys())
    return mapped, mapped_fields, warnings

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
