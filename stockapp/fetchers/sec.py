# stockapp/fetchers/sec.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Optional

import requests
import yfinance as yf
import streamlit as st


# ---------------------- Hjälpare ----------------------
def _ua() -> str:
    # Ange kontakt i st.secrets["SEC_USER_AGENT"] för att vara en god medborgare mot SEC.
    # Ex: "MyApp/1.0 (kontakt: dinmail@domän.se)"
    return st.secrets.get("SEC_USER_AGENT", "K-pf-rslag/1.0 (contact: example@example.com)")


def _get_json(url: str) -> dict:
    r = requests.get(url, headers={"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.json()


def _parse_cik_map() -> Dict[str, str]:
    """
    Hämta mapping TICKER -> CIK (CIK utan ledande nollor).
    SEC uppdaterar filen ibland; cacha i session.
    """
    cache_key = "_sec_cik_map"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    url = "https://www.sec.gov/files/company_tickers.json"
    data = _get_json(url)
    out: Dict[str, str] = {}
    # Struktur: {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).upper().strip()
        cik = str(rec.get("cik_str", "")).strip()
        if t and cik:
            out[t] = cik
    st.session_state[cache_key] = out
    return out


def _nearest_close_price(ticker: str, target_date: dt.date, window_days: int = 7) -> float:
    """
    Hämta stängningskurs kring periodens slut (target_date ± window_days).
    """
    start = target_date - dt.timedelta(days=window_days)
    end = target_date + dt.timedelta(days=window_days)
    h = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if h is None or h.empty:
        return 0.0
    # Välj närmast target_date
    h = h.copy()
    h["_d"] = h.index.date
    h["_diff"] = h["_d"].apply(lambda d: abs((d - target_date).days))
    row = h.sort_values("_diff").head(1)
    try:
        return float(row["Close"].iloc[0])
    except Exception:
        return 0.0


def _choose_unit(facts: dict, *cands: str) -> Optional[dict]:
    """
    Hitta första existerande fakta-tag under någon av kandidaterna.
    Returnerar faktanoden (med "units" under).
    """
    for tag in cands:
        node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
        if node:
            return node
    return None


def _extract_series(node: dict, preferred_units: List[str]) -> List[dict]:
    """
    Ta ut observationer från en nod (facts-tag). Försök prioritera vissa units.
    Returnerar lista med observationer (dicts) med nycklar bl.a. 'end','val','fp','form'.
    """
    units = node.get("units", {})
    # hitta första unit som finns
    for u in preferred_units:
        arr = units.get(u)
        if isinstance(arr, list) and arr:
            # Normalisera fält
            out = []
            for it in arr:
                # SEC använder "val" eller "value" beroende på endpoint-version; hantera båda.
                val = it.get("val", it.get("value", None))
                end = it.get("end") or it.get("fy") or ""
                form = it.get("form", "")
                fp = it.get("fp", "")  # Q1/Q2/Q3/FY
                try:
                    dt.datetime.strptime(end, "%Y-%m-%d")
                except Exception:
                    # hoppa över observationer utan korrekt datum
                    continue
                out.append({"val": val, "end": end, "form": form, "fp": fp})
            return out
    return []


# ---------------------- Publik funktion ----------------------
def get_pb_quarters(ticker: str) -> Dict[str, Any]:
    """
    Beräknar P/B för de fyra senaste rapportperioderna (10-Q/10-K) med SEC-data.
    Steg:
      1) Hämta CIK för tickern
      2) Läs companyfacts (equity & shares outstanding)
      3) Matcha equity och shares per period-end
      4) Hämta kurs runt period-end från Yahoo
      5) P/B = Price / (Equity / Shares)
    Returnerar: {"pb_quarters":[(date, pb_float), ...], "warnings":[...]}
    """
    warnings: List[str] = []
    tkr = ticker.upper().strip()
    try:
        cik_map = _parse_cik_map()
        cik = cik_map.get(tkr)
        if not cik:
            return {"pb_quarters": [], "warnings": [f"CIK saknas för {tkr}."]}

        facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json")

        # Equity
        equity_node = _choose_unit(
            facts,
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "LiabilitiesAndStockholdersEquity",
        )
        if not equity_node:
            return {"pb_quarters": [], "warnings": ["Hittade ingen equity-tag i SEC-facts."]}

        equity_series = _extract_series(equity_node, ["USD"])
        # Filtrera till 10-Q/10-K
        equity_series = [x for x in equity_series if x.get("form") in ("10-Q", "10-K")]

        # Shares outstanding (prefererar period-end shares)
        shares_node = _choose_unit(
            facts,
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
        )
        if not shares_node:
            return {"pb_quarters": [], "warnings": ["Hittade ingen shares-tag i SEC-facts."]}

        shares_series = _extract_series(shares_node, ["shares"])
        shares_series = [x for x in shares_series if x.get("form") in ("10-Q", "10-K")]

        # Indexera shares på datum för snabb lookup
        shares_by_end = {s["end"]: float(s.get("val") or 0.0) for s in shares_series if s.get("end")}

        # Gå igenom equity-serien, beräkna BVPS, pris nära 'end', sen PB
        records: List[tuple] = []
        for e in equity_series:
            end = e.get("end")
            if not end:
                continue
            try:
                equity_val = float(e.get("val") or 0.0)
            except Exception:
                continue
            shares = shares_by_end.get(end, 0.0)
            if shares <= 0:
                # fallback: försök hitta närmaste shares-datum ±1 dag
                try:
                    end_dt = dt.datetime.strptime(end, "%Y-%m-%d").date()
                    for delta in (-1, 1, -2, 2, -3, 3):
                        cand = (end_dt + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
                        if cand in shares_by_end and shares_by_end[cand] > 0:
                            shares = shares_by_end[cand]
                            break
                except Exception:
                    pass
            if shares <= 0:
                warnings.append(f"Saknar shares för {end}.")
                continue

            bvps = equity_val / shares if shares else 0.0
            try:
                end_dt = dt.datetime.strptime(end, "%Y-%m-%d").date()
            except Exception:
                continue
            px = _nearest_close_price(tkr, end_dt, window_days=7)
            if px <= 0 or bvps <= 0:
                continue
            pb = px / bvps
            records.append((end, pb))

        # Sortera på datum (nyast först), plocka 4
        records.sort(key=lambda x: x[0], reverse=True)
        records = records[:4]
        return {"pb_quarters": records, "warnings": warnings}
    except requests.HTTPError as e:
        return {"pb_quarters": [], "warnings": [f"HTTP-fel från SEC: {e}"]}
    except Exception as e:
        return {"pb_quarters": [], "warnings": [f"SEC-uppslag misslyckades: {e}"]}
