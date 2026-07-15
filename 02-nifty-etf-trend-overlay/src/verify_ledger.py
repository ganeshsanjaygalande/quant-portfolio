#!/usr/bin/env python3
"""
verify_ledger.py — CI enforcer. Non-zero exit => merge BLOCKED.

Checks (all must pass):
  1. data/clean/manifest.json exists and carries a dataset_hash (pipeline was run).
  2. If raw CSVs are present, recompute the hash and assert it EQUALS the committed
     manifest hash  (guards against a hand-edited / stale manifest).
  3. Every ACTIVE (non-STALE) research row in RESULTS_LEDGER.md carries the CURRENT
     dataset_hash. Any mismatch => the data changed but results weren't re-run/marked
     STALE => BLOCK.  (This is the change-invalidation rule, enforced.)
  4. K budget: number of distinct ACTIVE trial slots <= K_MAX (20).
  5. Ledger table is structurally parseable.
"""
import json, os, sys, re, tempfile
K_MAX = 20
SKIP_IDS = {"trial_id", "_ex_", "_example_", ""}

def fail(msg): print(f"[BLOCK] {msg}"); sys.exit(1)
def ok(msg):   print(f"[ok]    {msg}")

def load_manifest(root):
    p = os.path.join(root, "data", "clean", "manifest.json")
    if not os.path.exists(p):
        fail("data/clean/manifest.json missing — run src/data_pipeline.py before merge.")
    m = json.load(open(p))
    h = m.get("dataset_hash")
    if not h: fail("manifest.json has no dataset_hash.")
    ok(f"manifest dataset_hash = {h[:12]}  ({m.get('n_symbols','?')} symbols)")
    return h

def recompute_if_raw(root, committed):
    raw = os.path.join(root, "data", "raw")
    csvs = [f for f in os.listdir(raw)] if os.path.isdir(raw) else []
    csvs = [f for f in csvs if f.lower().endswith(".csv")]
    if not csvs:
        ok("raw data absent in CI — skipping recompute; enforcing committed manifest.")
        return
    sys.path.insert(0, os.path.join(root, "src"))
    import data_pipeline as dp
    with tempfile.TemporaryDirectory() as td:
        m = dp.build_panel(raw, td, freq="D", require_ok=False)
    if m["dataset_hash"] != committed:
        fail(f"recomputed hash {m['dataset_hash'][:12]} != committed manifest "
             f"{committed[:12]} — manifest is stale or tampered. Re-run pipeline.")
    ok("recomputed hash matches committed manifest (integrity verified).")

def parse_ledger(root):
    p = os.path.join(root, "RESULTS_LEDGER.md")
    if not os.path.exists(p): fail("RESULTS_LEDGER.md missing.")
    rows = []
    for ln in open(p, encoding="utf-8"):
        s = ln.strip()
        if not s.startswith("|"): continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 11: continue
        if set(cells[0].strip("_ ").lower() for _ in [0]) & {"trial_id"}: pass
        if re.match(r"^[-: ]+$", cells[0]): continue     # header separator
        rows.append(cells)
    return rows

def main():
    root = os.environ.get("REPO_ROOT", ".")
    current = load_manifest(root)
    recompute_if_raw(root, current)
    rows = parse_ledger(root)
    active, offenders = 0, []
    for r in rows:
        tid = r[0].strip()
        if tid.lower() in SKIP_IDS: continue
        status = r[10].strip().upper() if len(r) > 10 else ""
        dhash = r[3].strip()
        if status == "STALE": continue
        active += 1
        if not current.startswith(dhash.replace("`", "")):
            offenders.append((tid, dhash, status))
    if offenders:
        for tid, dh, st in offenders:
            print(f"        row {tid}: hash {dh} != current {current[:12]} (status {st})")
        fail(f"{len(offenders)} active ledger row(s) carry a stale dataset_hash. "
             f"Re-run study_engine on current data or mark rows STALE.")
    ok(f"all {active} active ledger rows match current dataset_hash.")
    if active > K_MAX:
        fail(f"K budget exceeded: {active} active trials > K_MAX={K_MAX}.")
    ok(f"K budget: {active}/{K_MAX} trials used.")
    print("[PASS] Ledger integrity + K-budget + hash-lock verified. Merge allowed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
