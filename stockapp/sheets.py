# stockapp/sheets.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials as SACredentials
except Exception:  # gspread saknas i miljön
    gspread = None
    SACredentials = None  # type: ignore


# ---------------------------------------------
# Konfiguration
# ---------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Primär nyckel + några alternativa namn vi accepterar
CRED_KEYS = [
    "GOOGLE_CREDENTIALS",
    "GOOGLE_SERVICE_ACCOUNT",
    "GCP_SERVICE_ACCOUNT",
    "SERVICE_ACCOUNT",
    "GSHEETS_SERVICE_ACCOUNT",
]

# Spreadsheet källa: URL (helst) eller ID
SHEET_URL = st.secrets.get("SHEET_URL", "")
SHEET_ID  = st.secrets.get("SHEET_ID", "")


# ---------------------------------------------
# Backoff-hjälpare
# ---------------------------------------------
def _with_backoff(func, *args, **kwargs):
    delays = [0.0, 0.5, 1.0, 2.0, 3.0]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    if last:
        raise last
    raise RuntimeError("Okänt fel i _with_backoff")


# ---------------------------------------------
# Creds-läsning: robust mot dict/JSON/AttrDict
# ---------------------------------------------
def _as_plain_dict(x: Any) -> Dict[str, Any]:
    """Försök göra om AttrDict/Mapping → vanlig dict (rekursivt)."""
    if isinstance(x, dict):
        return {k: _as_plain_dict(v) for k, v in x.items()}
    try:
        # AttrDict (st.secrets) beter sig som mapping
        keys = list(x.keys())  # type: ignore[attr-defined]
        return {k: _as_plain_dict(x[k]) for k in keys}  # type: ignore[index]
    except Exception:
        return x  # lämnade som är (str, int, osv)


def _maybe_json_load(s: str) -> Dict[str, Any]:
    """Prova json.loads. Om det misslyckas: försök reparera mest vanliga private_key-escape."""
    # Vanligast: hela service-account JSON som str
    try:
        return json.loads(s)
    except Exception:
        pass

    # Om någon lagt in JSON där private_key innehåller redan riktiga radbrytningar
    # eller är felciterat – vi gör ett sista försök att byta \\n → \n
    try:
        s2 = s.replace("\\n", "\n")
        return json.loads(s2)
    except Exception:
        # Sista chans: ibland är strängen single-quoted – byt till double quotes
        try:
            s3 = s.replace("'", '"')
            return json.loads(s3)
        except Exception:
            raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas (ogiltig JSON-sträng).")


def _normalize_private_key(creds: Dict[str, Any]) -> Dict[str, Any]:
    """Säkerställ att private_key har riktiga radbrytningar."""
    if "private_key" in creds and isinstance(creds["private_key"], str):
        pk = creds["private_key"]
        # Om det är bokstäverna \n (två tecken), ersätt med riktig radbrytning
        if "\\n" in pk:
            pk = pk.replace("\\n", "\n")
        creds["private_key"] = pk
    return creds


def _load_credentials_dict() -> Tuple[Dict[str, Any], str]:
    """
    Försök läsa service account från secrets:
      - dict/AttrDict → använd direkt
      - str → JSON-dekoda
    Returnerar (creds_dict, key_hit)
    """
    for key in CRED_KEYS:
        if key not in st.secrets:
            continue
        raw = st.secrets.get(key)

        # 1) Mapping (dict/AttrDict)
        if isinstance(raw, dict) or hasattr(raw, "keys"):
            d = _as_plain_dict(raw)
            if not isinstance(d, dict):
                raise RuntimeError(f"{key} fanns men hade okänt format (varken dict eller JSON-sträng).")
            d = _normalize_private_key(d)
            # Minimikoll
            if not d.get("client_email") or not d.get("private_key"):
                raise RuntimeError(f"{key} hittades, men saknar client_email/private_key.")
            return d, key

        # 2) Sträng (JSON)
        if isinstance(raw, str):
            d = _maybe_json_load(raw)
            d = _normalize_private_key(d)
            if not d.get("client_email") or not d.get("private_key"):
                raise RuntimeError(f"{key} hittades som JSON-sträng, men saknar client_email/private_key.")
            return d, key

        # 3) Allt annat → fel
        raise RuntimeError(f"{key} fanns men hade okänt format (varken dict eller JSON-sträng).")

    # Hittade ingenting
    raise RuntimeError(
        "Hittade inga service account-uppgifter i secrets. "
        "Lägg in t.ex. st.secrets['GOOGLE_CREDENTIALS'] (hela service-account JSON)."
    )


def _client():
    if gspread is None or SACredentials is None:
        raise RuntimeError("gspread saknas i miljön.")
    creds_dict, hit_key = _load_credentials_dict()
    creds = SACredentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


def get_spreadsheet():
    cli = _client()
    if SHEET_URL:
        return _with_backoff(cli.open_by_url, SHEET_URL)
    if SHEET_ID:
        return _with_backoff(cli.open_by_key, SHEET_ID)
    raise RuntimeError("SHEET_URL eller SHEET_ID måste finnas i st.secrets.")


def _get_or_create_worksheet(title: str):
    ss = get_spreadsheet()
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        # skapa nytt blad med 1 rad rubriker
        ws = _with_backoff(ss.add_worksheet, title=title, rows=1000, cols=50)
        _with_backoff(ws.update, [[""]])  # init
        return ws


# ---------------------------------------------
# Publika funktioner som appen anropar
# ---------------------------------------------
def list_worksheet_titles() -> List[str]:
    ss = get_spreadsheet()
    return [w.title for w in _with_backoff(ss.worksheets)]


def delete_worksheet(title: str) -> None:
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, title)
    except Exception:
        return
    _with_backoff(ss.del_worksheet, ws)


def _unique_headers(headers: List[str]) -> List[str]:
    """
    Säkerställ unika kolumnnamn (undviker Arrow/Streamlit-kollisionsfel).
    Ex: ['A','B','A'] → ['A','B','A (2)']
    """
    out = []
    seen = {}
    for h in headers:
        base = (h or "").strip()
        if base == "":
            base = "Kolumn"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base} ({seen[base]})")
    return out


def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läs hela bladet som DataFrame.
    - Rad 1 = rubriker. Om saknas → tom DF.
    - Tomt blad → tom DF.
    """
    ws = _get_or_create_worksheet(title)
    rows = _with_backoff(ws.get_all_values)()
    if not rows:
        return pd.DataFrame()

    # Säkerställ minst en rubrikrad
    headers = [str(x).strip() for x in (rows[0] if rows else [])]
    if not any(h for h in headers):
        return pd.DataFrame()

    headers = _unique_headers(headers)
    data = rows[1:] if len(rows) > 1 else []
    # Pad/kapa rader så att kolumnlängden matchar
    width = len(headers)
    norm_rows = [r[:width] + [""] * max(0, width - len(r)) for r in data]
    df = pd.DataFrame(norm_rows, columns=headers)
    return df


def ws_write_df(title: str, df: pd.DataFrame) -> None:
    """
    Skriv DataFrame → bladet (cleara först).
    Alla värden skrivs som str för att undvika implicit formatering.
    """
    ws = _get_or_create_worksheet(title)
    if df is None or df.empty:
        _with_backoff(ws.clear)
        # lämna kvar rubrikrad tom → ok
        _with_backoff(ws.update, [[""]])
        return

    # Säkerställ strängvärden
    cols = [str(c) for c in df.columns]
    values: List[List[str]] = [cols]
    for _, row in df.iterrows():
        values.append([str(v) if v is not None else "" for v in row.tolist()])

    # Clear + skriv bulk
    _with_backoff(ws.clear)
    _with_backoff(ws.update, values)
