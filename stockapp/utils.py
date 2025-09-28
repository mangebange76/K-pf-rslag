# stockapp/utils.py
# -*- coding: utf-8 -*-
"""
Allmänna utilities som används brett i appen.

Den här modulen försöker hålla NOLL beroenden utanför standardbiblioteket +
pandas/numpy, och den importerar endast konstanter från config.

Innehåll (urval):
- with_backoff(fn, *args, **kwargs)  -> robusta retries (t.ex. mot Google Sheets/API)
- parse_ts(s), now_ts(), ts_str(dt)  -> säkra tidsstämplar
- safe_float(x, default=0.0)
- human_amount(n), format_marketcap(n, currency="USD", show_raw=False)
- ensure_final_cols(df)  -> garanterar att FINAL_COLS finns (skapar tomma vid behov)
- set_timestamp(df, idx, field)      -> sätter fält(TS) med nuvarande ISO-sträng
- add_oldest_ts_col(df, out_col="_oldest_ts") -> beräknar äldsta TS per rad
- norm_ticker(t) & check_duplicate_tickers(df)
- chunked(iterable, size)
- progress_text(i, total) -> "i/total"
- label_risk_from_mcap(mcap_usd)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import math
import random
import time
import re

import numpy as np
import pandas as pd

from .config import (
    FINAL_COLS,
    TS_FIELDS,
    DEFAULTS,
    MARKETCAP_LABELS,
    RISK_BUCKETS,
)


# ----------------------------------------------------------------------
# Fel-typ för användarvänliga fel (kan fångas i vyer och visas snyggt)
# ----------------------------------------------------------------------
class AppUserError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Retry / backoff
# ----------------------------------------------------------------------
def with_backoff(
    fn: Callable[..., Any],
    *args,
    tries: int = 5,
    base_delay: float = 0.6,
    factor: float = 1.7,
    jitter: float = 0.4,
    exceptions: Tuple[type, ...] = (Exception,),
    **kwargs,
) -> Any:
    """
    Kör fn(*args, **kwargs) med exponentiell backoff.
    Återkastar sista felet om alla försök misslyckas.
    """
    last_exc: Optional[BaseException] = None
    delay = base_delay
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:  # pragma: no cover
            last_exc = exc
            if attempt >= tries:
                break
            sleep_for = delay + random.random() * jitter
            time.sleep(sleep_for)
            delay *= factor
    if last_exc:
        raise last_exc
    # borde inte nå hit
    return None


# ----------------------------------------------------------------------
# Tidsstämplar
# ----------------------------------------------------------------------
_TS_PATTERNS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]

def now_ts() -> datetime:
    return datetime.now(timezone.utc)

def ts_str(dt: Optional[datetime] = None) -> str:
    """ISO8601 med 'Z' för UTC."""
    if dt is None:
        dt = now_ts()
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_ts(s: Any) -> Optional[datetime]:
    """Försöker tolka s som tidsstämpel. Returnerar None om omöjligt."""
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
    text = str(s).strip()
    if not text:
        return None
    for pat in _TS_PATTERNS:
        try:
            dt = datetime.strptime(text, pat)
            # antag UTC om tidszon saknas
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    # Pandas kan ibland ha tidsstämplar i annan form
    try:
        dt2 = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(dt2):
            return None
        return dt2.to_pydatetime().astimezone(timezone.utc)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Tal/format
# ----------------------------------------------------------------------
def safe_float(x: Any, default: float = 0.0) -> float:
    """Robust float-omvandling (byter kommatecken etc.)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return default
    if isinstance(x, (int, float)):
        return float(x)
    try:
        txt = str(x).strip().replace(" ", "").replace(",", ".")
        # plocka ut första numeriska mönstret om skräp
        m = re.search(r"[-+]?\d+(\.\d+)?", txt)
        if m:
            return float(m.group(0))
        return float(txt)
    except Exception:
        return default


def human_amount(n: Union[int, float]) -> str:
    """Svensk förkortning (miljoner/miljarder/biljoner)."""
    try:
        val = float(n)
    except Exception:
        return "0"
    sign = "-" if val < 0 else ""
    val = abs(val)

    for threshold, label in MARKETCAP_LABELS:
        if val >= threshold:
            return f"{sign}{val/threshold:.2f} {label}"
    # < 1 miljon
    return f"{sign}{val:,.0f}".replace(",", " ")


def format_marketcap(n: Any, currency: str = "USD", show_raw: bool = False) -> str:
    """Formaterar market cap med enhet och ev. råtal i parentes."""
    val = safe_float(n, 0.0)
    pretty = human_amount(val)
    if show_raw:
        return f"{pretty} {currency} ({val:,.0f})".replace(",", " ")
    return f"{pretty} {currency}"


def label_risk_from_mcap(mcap_usd: Any) -> str:
    """
    Baserat på RISK_BUCKETS i config.
    Tar USD-belopp.
    """
    val = safe_float(mcap_usd, 0.0)
    for thr, label in RISK_BUCKETS:
        if val >= thr:
            return label
    return RISK_BUCKETS[-1][1] if RISK_BUCKETS else "Okänd"


# ----------------------------------------------------------------------
# DataFrame-hjälp
# ----------------------------------------------------------------------
def ensure_final_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garanterar att alla FINAL_COLS finns. Saknade skapas med NaN
    (och default om definierat i DEFAULTS).
    """
    work = df.copy()
    for c in FINAL_COLS:
        if c not in work.columns:
            work[c] = np.nan
        if c in DEFAULTS:
            # fyll bara NaN/None – skriv inte över befintliga värden
            work[c] = work[c].fillna(DEFAULTS[c])
    return work


def set_timestamp(df: pd.DataFrame, idx: int, field: str) -> None:
    """
    Sätter fält (TS) i given rad-index till nu.
    Om fältet inte finns skapas det.
    """
    if field not in df.columns:
        df[field] = np.nan
    df.at[idx, field] = ts_str()


def add_oldest_ts_col(df: pd.DataFrame, out_col: str = "_oldest_ts") -> pd.DataFrame:
    """
    Beräknar äldsta TS över alla TS_FIELDS och lägger som datetime i out_col.
    Saknas TS helt, blir None.
    """
    work = df.copy()
    vals: List[Optional[datetime]] = []
    for _, row in work.iterrows():
        dts: List[datetime] = []
        for f in TS_FIELDS:
            dt = parse_ts(row.get(f))
            if dt is not None:
                dts.append(dt)
        vals.append(min(dts) if dts else None)
    work[out_col] = vals
    return work


# ----------------------------------------------------------------------
# Ticker & dubblett
# ----------------------------------------------------------------------
def norm_ticker(t: Any) -> str:
    """Trim/upper – enkel normalisering."""
    return str(t or "").strip().upper()


def check_duplicate_tickers(df: pd.DataFrame) -> List[str]:
    """Returnerar lista med tickers som är dubbletter (case-insensitive)."""
    if "Ticker" not in df.columns:
        return []
    s = df["Ticker"].map(norm_ticker)
    dup = s[s.duplicated(keep=False)]
    return list(sorted(set(dup.tolist())))


# ----------------------------------------------------------------------
# Diverse
# ----------------------------------------------------------------------
def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Skär upp seq i bitar om 'size'."""
    if size <= 0:
        size = 1
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def progress_text(i: int, total: int) -> str:
    """Returnerar 'i/X' (1-baserad indexering)."""
    i = max(1, int(i))
    total = max(1, int(total))
    return f"{i}/{total}"


# ----------------------------------------------------------------------
# Små hjälp för beräkningar
# ----------------------------------------------------------------------
def ps_from_mcap_and_revenue(mcap: Any, revenue_ttm: Any) -> float:
    """
    P/S = Market Cap / Omsättning (TTM). Skyddar mot noll.
    """
    mc = safe_float(mcap, 0.0)
    rev = safe_float(revenue_ttm, 0.0)
    if rev <= 0:
        return 0.0
    return mc / rev
