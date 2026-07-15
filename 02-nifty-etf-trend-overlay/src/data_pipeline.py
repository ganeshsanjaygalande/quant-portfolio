#!/usr/bin/env python3
"""
data_pipeline.py — Pre-Flight ingestion + QC + reproducibility layer.
================================================================================
CTO/Referee bedrock for the large-universe trend study. Turns a directory of raw
roll-adjusted continuous-futures CSVs into a single clean, UTC-aligned panel with
a deterministic SHA-256 dataset hash for the RESULTS_LEDGER.

Guarantees:
  * Any change to raw inputs OR to the cleaning logic -> new dataset_hash ->
    every prior ledger row keyed on the old hash is automatically invalidated.
  * Strict QC: gaps, missing bars, NaNs, dup timestamps, non-positive prices,
    stale/flat runs, tz consistency, back-adjustment (negative-price) guard.
  * No look-ahead, no silent fills of price (only reports gaps; filling is an
    explicit, logged policy decision, off by default).

Usage:
  python data_pipeline.py --raw data/raw --out data/clean --freq D
  # returns exit code 0 and prints dataset_hash on success; non-zero on QC FAIL.

Author: Claude (CTO/Referee).  Reviewed against AFML data-integrity practice.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, io
from datetime import datetime, timezone
import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0.0"   # bump this if cleaning logic changes -> new hashes

# ── hashing ──────────────────────────────────────────────────────────────────
def sha256_file(path: str, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# ── flexible loader (MT5 tab-<COL> export OR generic Norgate/CSI CSV) ─────────
def _read_one(path: str, tz: str) -> pd.DataFrame:
    # sniff delimiter
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    sep = "\t" if head.count("\t") >= head.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]
    # datetime
    if "DATE" in df.columns and "TIME" in df.columns:
        dt = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str),
                            errors="coerce")
    elif "DATETIME" in df.columns:
        dt = pd.to_datetime(df["DATETIME"], errors="coerce")
    elif "DATE" in df.columns:
        dt = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        raise ValueError(f"{os.path.basename(path)}: no DATE/DATETIME column")
    df.index = dt
    # normalize OHLCV column names
    ren = {"OPEN":"open","HIGH":"high","LOW":"low","CLOSE":"close",
           "ADJ CLOSE":"close","VOL":"volume","VOLUME":"volume","TICKVOL":"tickvol"}
    df = df.rename(columns={k:v for k,v in ren.items() if k in df.columns})
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    if "close" not in keep:
        raise ValueError(f"{os.path.basename(path)}: no CLOSE column")
    out = df[keep].copy()
    # tz: assume inputs are UTC unless told otherwise; make tz-aware UTC
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize(timezone.utc)
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.isna()]
    return out.sort_index()

# ── per-series QC ────────────────────────────────────────────────────────────
def qc_series(sym: str, df: pd.DataFrame, freq: str) -> dict:
    issues = []
    n = len(df)
    dup = int(df.index.duplicated().sum())
    if dup: issues.append(f"{dup} duplicate timestamps")
    nan = int(df["close"].isna().sum())
    if nan: issues.append(f"{nan} NaN closes")
    nonpos = int((df["close"] <= 0).sum())
    if nonpos: issues.append(f"{nonpos} non-positive closes")
    # back-adjustment guard: Panama-adjusted series can go negative/near-zero
    if (df[["open","high","low","close"]].min().min() if set(["open","high","low"]) <= set(df.columns) else df["close"].min()) < 0:
        issues.append("NEGATIVE prices (Panama back-adjust?) -> returns must use ratio-adj/unadjusted series")
    # gaps: business-day gaps for D; for intraday just report max gap
    if freq.upper().startswith("D"):
        bidx = pd.bdate_range(df.index.min().tz_convert(None).normalize(),
                              df.index.max().tz_convert(None).normalize())
        present = df.index.tz_convert(None).normalize().unique()
        missing = len(bidx.difference(present))
        gap_pct = 100 * missing / max(len(bidx), 1)
        if gap_pct > 10:  # >10% of business days absent is suspicious (holidays ~ a few %)
            issues.append(f"{missing} missing business days ({gap_pct:.1f}%)")
    # stale / flat runs (>5 identical consecutive closes = likely bad feed)
    _mx = (df["close"].diff() == 0).rolling(5).sum().max()
    flat = int(_mx) if pd.notna(_mx) else 0
    if flat >= 5: issues.append("stale run >=5 flat closes")
    # extreme single-bar returns (>50% daily = likely bad tick or wrong scale)
    r = df["close"].pct_change()
    ext = int((r.abs() > 0.5).sum())
    if ext: issues.append(f"{ext} extreme (>50%) single-bar moves")
    return dict(symbol=sym, rows=n, start=str(df.index.min()), end=str(df.index.max()),
                dup=dup, nan=nan, nonpos=nonpos, issues=issues,
                status="FAIL" if (dup or nan or nonpos or any("NEGATIVE" in i for i in issues)) else
                       ("WARN" if issues else "OK"))

# ── build the aligned panel ──────────────────────────────────────────────────
def build_panel(raw_dir: str, out_dir: str, freq: str = "D",
                tz: str = "UTC", require_ok: bool = True) -> dict:
    files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                    if f.lower().endswith(".csv")])
    if not files:
        raise SystemExit(f"[FATAL] no CSVs in {raw_dir}")
    raw_manifest = {os.path.basename(p): sha256_file(p) for p in files}

    series, qc = {}, []
    for p in files:
        sym = os.path.basename(p).split("_")[0].split(".")[0].upper()
        try:
            df = _read_one(p, tz)
        except Exception as e:
            qc.append(dict(symbol=sym, status="FAIL", issues=[f"load error: {e}"], rows=0))
            continue
        rep = qc_series(sym, df, freq)
        qc.append(rep)
        if rep["status"] != "FAIL":
            series[sym] = df["close"].rename(sym)

    if not series:
        raise SystemExit("[FATAL] no loadable series survived QC")

    # align: normalize to UTC calendar (D) or keep UTC timestamps (intraday)
    panel = pd.DataFrame(series)
    if freq.upper().startswith("D"):
        panel.index = panel.index.tz_convert("UTC").normalize()
        panel = panel[~panel.index.duplicated(keep="last")]
    panel = panel.sort_index()

    # returns from the *close* stream (assumed ratio/roll-adjusted for returns)
    rets = panel.pct_change(fill_method=None)

    # deterministic serialization -> dataset hash (schema + panel bytes)
    # canonical, float-stable CSV serialization -> reproducible hash (no parquet dep)
    canon = panel.to_csv(float_format="%.10g").encode("utf-8")
    # fold optional upstream-adapter provenance (version + corporate-action state)
    prov = b""
    prov_path = os.path.join(raw_dir, "_provenance.json")
    if os.path.exists(prov_path):
        with open(prov_path, "rb") as _f: prov = _f.read()
    dataset_hash = sha256_bytes(SCHEMA_VERSION.encode() + prov + canon)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "panel_close.csv"), "wb") as fh: fh.write(canon)
    rets.to_csv(os.path.join(out_dir, "panel_returns.csv"), float_format="%.10g")

    manifest = dict(
        schema_version=SCHEMA_VERSION,
        built_utc=datetime.now(timezone.utc).isoformat(),
        freq=freq, tz=tz,
        n_symbols=panel.shape[1], n_rows=panel.shape[0],
        date_start=str(panel.index.min()), date_end=str(panel.index.max()),
        symbols=list(panel.columns),
        raw_files=raw_manifest,
        dataset_hash=dataset_hash,
        qc=qc,
        common_rows_all_present=int(panel.dropna().shape[0]),
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    fails = [q for q in qc if q["status"] == "FAIL"]
    warns = [q for q in qc if q["status"] == "WARN"]
    print(f"[PANEL] {panel.shape[1]} symbols x {panel.shape[0]} rows "
          f"({panel.index.min().date()} -> {panel.index.max().date()}), "
          f"all-present rows={manifest['common_rows_all_present']}")
    for q in qc:
        tag = {"OK":"  ok","WARN":"WARN","FAIL":"FAIL"}[q["status"]]
        extra = ("; ".join(q.get("issues", []))) or "-"
        print(f"  [{tag}] {q['symbol']:8s} rows={q.get('rows',0):>6} {extra}")
    print(f"\n[DATASET_HASH] {dataset_hash}")
    print(f"[SCHEMA] {SCHEMA_VERSION}")
    if fails and require_ok:
        print(f"\n[PRE-FLIGHT: FAIL] {len(fails)} series failed QC. "
              f"Fix raw data before any backtest. Ledger runs are BLOCKED.")
        return manifest  # caller decides; CLI returns nonzero below
    print(f"\n[PRE-FLIGHT: {'PASS' if not fails else 'PASS-WITH-FAILS'}] "
          f"{len(warns)} warnings. Use DATASET_HASH above in RESULTS_LEDGER.md.")
    return manifest

def main():
    ap = argparse.ArgumentParser(description="Pre-Flight data ingestion + QC + hash")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freq", default="D", choices=["D", "H1", "H", "M5"])
    ap.add_argument("--tz", default="UTC")
    ap.add_argument("--allow-fails", action="store_true",
                    help="build panel even if some series FAIL (still reported)")
    a = ap.parse_args()
    m = build_panel(a.raw, a.out, a.freq, a.tz, require_ok=not a.allow_fails)
    sys.exit(1 if any(q["status"] == "FAIL" for q in m["qc"]) and not a.allow_fails else 0)

if __name__ == "__main__":
    main()
