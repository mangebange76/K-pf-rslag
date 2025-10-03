# stockapp/fetchers/yahoo.py
from __future__ import annotations
import math, typing as t

try:
    import yfinance as yf  # type: ignore
except Exception as _imp_err:
    yf = None  # type: ignore
    _YF_IMPORT_ERR = _imp_err
else:
    _YF_IMPORT_ERR = None  # type: ignore

def _safe_num(x: t.Any) -> t.Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""): return None
        v = float(x)
        if math.isnan(v) or math.isinf(v): return None
        return v
    except Exception:
        return None

def _to_millions(x: t.Any) -> t.Optional[float]:
    v = _safe_num(x);  return None if v is None else v / 1_000_000.0

def _round4(v: t.Any) -> t.Any:
    n = _safe_num(v);  return round(n, 4) if n is not None else None

def _fetch_raw(ticker: str) -> tuple[dict, list[str], list[str]]:
    raw, fetched, warns = {}, [], []
    if yf is None:
        warns.append(f"yfinance saknas: {_YF_IMPORT_ERR}")
        return raw, fetched, warns

    tkr = yf.Ticker((ticker or '').strip().upper())
    info: dict[str, t.Any] = {}
    fast: dict[str, t.Any] = {}

    # fast_info
    try:
        fi = getattr(tkr, "fast_info", None)
        if fi:
            fast = {k: getattr(fi, k) for k in dir(fi) if not k.startswith("_")}
    except Exception as e:
        warns.append(f"fast_info fel: {e}")

    # info
    try:
        info = tkr.get_info()
    except Exception:
        try:
            info = tkr.info
        except Exception as e:
            warns.append(f"info fel: {e}")
            info = {}

    price = (_safe_num(fast.get("last_price")) or
             _safe_num(fast.get("lastTradePrice")) or
             _safe_num(info.get("currentPrice")) or
             _safe_num(info.get("regularMarketPrice")))
    if price is not None:
        raw["YH:Price"] = price; fetched.append("YH:Price")

    mcap = _safe_num(fast.get("market_cap")) or _safe_num(info.get("marketCap"))
    if mcap is not None:
        raw["YH:Market Cap"] = mcap; fetched.append("YH:Market Cap")

    shares = _safe_num(fast.get("shares")) or _safe_num(info.get("sharesOutstanding"))
    if shares is not None:
        raw["YH:Shares Outstanding"] = shares; fetched.append("YH:Shares Outstanding")

    currency = info.get("currency") or fast.get("currency")
    if isinstance(currency, str) and currency:
        raw["YH:Currency"] = currency; fetched.append("YH:Currency")

    name = info.get("longName") or info.get("shortName")
    if name: raw["YH:Company Name"] = name; fetched.append("YH:Company Name")
    if info.get("sector"):   raw["YH:Sector"] = info.get("sector"); fetched.append("YH:Sector")
    if info.get("industry"): raw["YH:Industry"] = info.get("industry"); fetched.append("YH:Industry")

    rev = (_safe_num(info.get("totalRevenue")) or
           _safe_num(info.get("trailingAnnualRevenue")))
    if rev is None:
        try:
            fin = tkr.get_income_stmt(freq="annual")
            for key in ["TotalRevenue","Total Revenue","Total_Revenue","Total revenue","Totalrevenue"]:
                if key in fin.index:
                    series = fin.loc[key].astype("float64").dropna()
                    if not series.empty:
                        rev = float(series.iloc[-1])
                    break
        except Exception as e:
            warns.append(f"income_stmt fel: {e}")
    if rev is not None:
        raw["YH:Revenue (Annual)"] = rev; fetched.append("YH:Revenue (Annual)")

    if not fetched:
        warns.append("Yahoo returnerade inga fält.")
    return raw, fetched, warns

def _map_to_app(raw: dict) -> dict:
    m: dict[str, t.Any] = {}
    if raw.get("YH:Company Name"): m["Bolagsnamn"] = raw["YH:Company Name"]
    if raw.get("YH:Currency"):     m["Valuta"] = raw["YH:Currency"]
    if raw.get("YH:Sector"):       m["Sektor"] = raw["YH:Sector"]
    if raw.get("YH:Industry"):     m["Industri"] = raw["YH:Industry"]; m["Bransch"] = raw["YH:Industry"]

    if (v := _safe_num(raw.get("YH:Price"))) is not None: m["Kurs"] = v
    if (v := _safe_num(raw.get("YH:Market Cap"))) is not None:
        m["Market Cap"] = v; m["Market Cap (M)"] = _round4(v / 1_000_000.0)
    if (v := _to_millions(raw.get("YH:Shares Outstanding"))) is not None:
        m["Utestående aktier (milj.)"] = _round4(v); m["TS_Utestående aktier"] = _round4(v)
    if (v := _to_millions(raw.get("YH:Revenue (Annual)"))) is not None:
        m["Omsättning i år (M)"] = _round4(v); m["TS_Omsättning idag"] = _round4(v)

    mc = _safe_num(raw.get("YH:Market Cap")); rv = _safe_num(raw.get("YH:Revenue (Annual)"))
    if mc is not None and rv and rv > 0:
        ps = mc / rv
        m["P/S"] = _round4(ps); m["P/S TTM"] = _round4(ps); m["P/S (TTM, modell)"] = _round4(ps)

    return {k: v for k, v in m.items() if v is not None}

def get_all(ticker: str) -> dict:
    try:
        raw, _f, _w = _fetch_raw(ticker); return _map_to_app(raw)
    except Exception:
        return {}

def get_all_verbose(ticker: str):
    raw, fetched, warns = _fetch_raw(ticker)
    mapped = _map_to_app(raw)
    return mapped, list(mapped.keys()), warns

def format_fetch_summary(source: str, fetched: list[str], warnings: list[str]) -> str:
    parts = [f"{source}: Hämtade {len(fetched)} fält." if fetched else f"{source}: Hämtade 0 fält."]
    if warnings: parts.append("Varningar: " + " | ".join(warnings))
    return " ".join(parts)
