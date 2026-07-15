#!/usr/bin/env python3
"""
nse_adapter.py — NSE Bhavcopy -> canonical EOD series for a single symbol.
================================================================================
Upstream adapter for the sealed pipeline. Produces a raw CSV in the exact schema
data_pipeline.py ingests, so all downstream SHA-256 hashing / QC / CI is unchanged.

Dual-format aware:
  * UDiFF (modern, ~2024-07+):  BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
  * Legacy (pre-cutover):       .../historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
fetch tries the era-appropriate URL first and falls back to the other on 404, so a
fuzzy cutover date self-heals. extract auto-detects the column schema per file.

Immutability lives at the raw-zip cache. Overwriting lives only in the derived frame.
"""
from __future__ import annotations
import os, time, zipfile, tempfile
from datetime import date, timedelta
import pandas as pd

# ── versioning (folds into the hash-invalidation discipline) ─────────────────
ADAPTER_VERSION = 2     # bump on ANY logic change (v2: dual-format fetch/extract)

# ── endpoints ────────────────────────────────────────────────────────────────
UDIFF_BASE   = "https://nsearchives.nseindia.com/content/cm"
UDIFF_FILE   = "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
LEGACY_BASE  = "https://nsearchives.nseindia.com/content/historical/EQUITIES"
UDIFF_CUTOVER = date(2024, 7, 8)      # approx; fallback-on-404 makes this non-critical
CACHE_DIR    = os.path.join("data", "raw", "_bhav_cache")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# ── column maps: UDiFF (modern) and legacy (pre-2024) ────────────────────────
COL = {          # UDiFF
    "date": "TradDt", "sym": "TckrSymb", "series": "SctySrs",
    "open": "OpnPric", "high": "HghPric", "low": "LwPric",
    "close": "ClsPric", "vol": "TtlTradgVol",
}
COL_LEGACY = {   # legacy cm bhavcopy
    "date": "TIMESTAMP", "sym": "SYMBOL", "series": "SERIES",
    "open": "OPEN", "high": "HIGH", "low": "LOW",
    "close": "CLOSE", "vol": "TOTTRDQTY",
}

# ── corporate-action table: prices STRICTLY BEFORE ex_date are adjusted ───────
# NIFTYBEES 10->1 face-value split, ex-date 2019-12-19: pre-split /10, volume *10.
CORPORATE_ACTIONS = [
    {"symbol": "NIFTYBEES", "ex_date": "2019-12-19", "price_div": 10.0, "vol_mult": 10.0},
]


# ── URL resolution (pure -> unit-testable offline) ───────────────────────────
def _udiff_url(d: date):
    f = UDIFF_FILE.format(yyyymmdd=d.strftime("%Y%m%d"))
    return f"{UDIFF_BASE}/{f}", f

def _legacy_url(d: date):
    mon = d.strftime("%b").upper()                     # DEC
    f = f"cm{d.strftime('%d')}{mon}{d.strftime('%Y')}bhav.csv.zip"   # cm19DEC2019bhav.csv.zip
    return f"{LEGACY_BASE}/{d.strftime('%Y')}/{mon}/{f}", f

def _url_candidates(d: date):
    """Era-appropriate URL first, other as fallback. Returns [(url, cache_fname), ...]."""
    return [_udiff_url(d), _legacy_url(d)] if d >= UDIFF_CUTOVER \
        else [_legacy_url(d), _udiff_url(d)]


# ── component 1: downloader (dual-format, fallback-on-404) ───────────────────
def fetch_bhavcopy(d: date, retries: int = 3, timeout: int = 30) -> str:
    """Download ONE day's bhavcopy (UDiFF or legacy), cache immutably, return path.
       Raises FileNotFoundError only if BOTH formats 404 (genuine non-trading day)."""
    import requests  # lazy: offline components don't need the network dep
    os.makedirs(CACHE_DIR, exist_ok=True)
    candidates = _url_candidates(d)

    for _url, fname in candidates:                     # cache hit on either format
        p = os.path.join(CACHE_DIR, fname)
        if os.path.exists(p) and zipfile.is_zipfile(p):
            return p

    sess = requests.Session(); sess.headers.update(_HEADERS)
    all_404 = True; last_err = None
    for url, fname in candidates:
        path = os.path.join(CACHE_DIR, fname)
        for attempt in range(1, retries + 1):
            try:
                r = sess.get(url, timeout=timeout)
                if r.status_code == 404:
                    break                              # this format absent -> try next
                r.raise_for_status()
                all_404 = False
                if r.content[:2] != b"PK":
                    raise ValueError(f"{d}: not a zip (got {r.content[:60]!r})")
                fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(r.content)
                if not zipfile.is_zipfile(tmp):
                    os.remove(tmp)
                    raise ValueError(f"{d}: downloaded bytes are not a valid zip")
                os.replace(tmp, path)                  # atomic
                return path
            except Exception as e:
                all_404 = False
                last_err = e
                time.sleep(2 ** attempt)
    if all_404:
        raise FileNotFoundError(f"No bhavcopy for {d} in either format (non-trading day?)")
    raise RuntimeError(f"fetch_bhavcopy failed for {d}: {last_err}")


# ── component 2: extractor (auto-detects UDiFF vs legacy schema) ─────────────
def extract_symbol(zip_path: str, symbol: str = "NIFTYBEES", series: str = "EQ") -> dict | None:
    """Pull one symbol's OHLCV row from a cached zip (either schema).
       Returns None if the symbol wasn't traded that day (pre-listing)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{zip_path}: expected 1 CSV inside, found {members}")
        with z.open(members[0]) as fh:
            df = pd.read_csv(fh)
    df.columns = [c.strip() for c in df.columns]

    # dispatch on whichever schema's symbol column is present
    if COL["sym"] in df.columns:
        cmap = COL
    elif COL_LEGACY["sym"] in df.columns:
        cmap = COL_LEGACY
    else:
        raise KeyError(f"{zip_path}: unrecognized schema. header = {list(df.columns)}")

    missing = [v for v in cmap.values() if v not in df.columns]
    if missing:
        raise KeyError(f"{zip_path}: missing {missing}. header = {list(df.columns)}")

    row = df[(df[cmap["sym"]].astype(str).str.strip() == symbol) &
             (df[cmap["series"]].astype(str).str.strip() == series)]
    if len(row) == 0:
        return None
    if len(row) > 1:
        raise ValueError(f"{zip_path}: {symbol}/{series} matched {len(row)} rows — anomaly")

    r = row.iloc[0]
    return {"date": pd.to_datetime(str(r[cmap["date"]])).strftime("%Y-%m-%d"),  # normalize
            "open": float(r[cmap["open"]]), "high": float(r[cmap["high"]]),
            "low": float(r[cmap["low"]]), "close": float(r[cmap["close"]]),
            "volume": float(r[cmap["vol"]])}


# ── component 3: stitcher ────────────────────────────────────────────────────
def build_series(start: date, end: date, symbol: str = "NIFTYBEES",
                 max_consec_404: int = 4):
    """Stitch cached bhavcopies into one continuous series. Returns (df, skipped_log).
       Skips weekends deterministically; halts on a suspicious weekday-404 run."""
    end = min(end, date.today())
    rows, skipped = [], []
    consec_404, last_good = 0, None
    d = start
    while d <= end:
        if d.weekday() >= 5:                           # Sat/Sun: deterministic non-trading
            d += timedelta(days=1); continue
        try:
            zp = fetch_bhavcopy(d)
        except FileNotFoundError:
            consec_404 += 1
            skipped.append((d.isoformat(), "weekday-404"))
            if consec_404 > max_consec_404:
                raise RuntimeError(
                    f"{consec_404} consecutive weekday 404s ending {d} — publication lag "
                    f"or broken URL, NOT holidays. Last good date: {last_good}. Halting.")
            d += timedelta(days=1); continue
        consec_404 = 0                                 # file reachable -> reset (tracks FETCH)
        row = extract_symbol(zp, symbol)
        if row is None:
            skipped.append((d.isoformat(), "not-listed"))
        else:
            rows.append(row); last_good = d
        d += timedelta(days=1)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df, skipped


# ── component 4: corporate-action adjuster (pure fn) ─────────────────────────
def apply_adjustments(df: pd.DataFrame, symbol: str = "NIFTYBEES") -> pd.DataFrame:
    """Return a NEW adjusted frame. Prices strictly BEFORE each ex_date are divided
       by price_div; volume multiplied by vol_mult. Pure & idempotent: apply ONCE to
       the stitched raw. The engine consumes only this adjusted 'reality'; the raw
       cached zips remain the untouched source of truth."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for ca in CORPORATE_ACTIONS:
        if ca["symbol"] != symbol:
            continue
        ex = pd.Timestamp(ca["ex_date"])
        mask = out["date"] < ex                        # strictly before ex-date
        for c in ("open", "high", "low", "close"):
            out.loc[mask, c] = out.loc[mask, c] / ca["price_div"]
        out.loc[mask, "volume"] = out.loc[mask, "volume"] * ca["vol_mult"]
    return out


# ── orchestrator: stitch -> adjust -> self-QC ────────────────────────────────
def build_adjusted_series(start: date, end: date, symbol: str = "NIFTYBEES"):
    raw, skipped = build_series(start, end, symbol)
    adj = apply_adjustments(raw, symbol)
    if len(adj) > 1:                                   # QC: a real split-cliff must be GONE
        jumps = adj["close"].pct_change().abs()
        cliffs = adj.loc[jumps > 0.5, "date"].dt.date.astype(str).tolist()
        if cliffs:
            print(f"[WARN] residual >50% overnight move(s) at {cliffs} — "
                  f"unhandled corporate action? check CORPORATE_ACTIONS.")
    return adj, skipped


# ── writer: canonical schema for data_pipeline.py ────────────────────────────
def write_canonical(adj: pd.DataFrame, out_path: str, symbol: str = "NIFTYBEES"):
    """Emit the tab-delimited <DATE>\t<TIME>\t<OPEN>... file data_pipeline ingests."""
    out = pd.DataFrame({
        "<DATE>": adj["date"].dt.strftime("%Y.%m.%d"),
        "<TIME>": "00:00:00",
        "<OPEN>": adj["open"], "<HIGH>": adj["high"],
        "<LOW>": adj["low"], "<CLOSE>": adj["close"],
        "<TICKVOL>": adj["volume"].astype("int64"),
        "<VOL>": 0, "<SPREAD>": 0,
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    return out_path


# ── provenance: adapter version + corporate-action state -> hashed by pipeline ─
def write_provenance(raw_dir: str = os.path.join("data", "raw")):
    """Drop _provenance.json so data_pipeline folds adapter logic-state into the hash.
       Changing ADAPTER_VERSION *or* the CORPORATE_ACTIONS table changes this file,
       which changes dataset_hash even if the CSV numbers happen not to move."""
    import json, hashlib
    ca_sha = hashlib.sha256(json.dumps(CORPORATE_ACTIONS, sort_keys=True).encode()).hexdigest()
    os.makedirs(raw_dir, exist_ok=True)
    prov = {"adapter_version": ADAPTER_VERSION, "corporate_actions_sha256": ca_sha}
    with open(os.path.join(raw_dir, "_provenance.json"), "w") as f:
        json.dump(prov, f, sort_keys=True, indent=2)
    return prov


if __name__ == "__main__":
    import sys
    s = date(2019, 12, 10); e = date(2019, 12, 31)     # legacy window spanning the split
    adj, skipped = build_adjusted_series(s, e)
    print(adj.to_string(index=False)); print("skipped:", skipped)
    if len(sys.argv) > 1:
        write_provenance()
        write_canonical(adj, sys.argv[1])
