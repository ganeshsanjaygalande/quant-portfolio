# CI_SPEC.md — Pre-Flight Harness Contract (for Jules & Ruflo)

The `Pre-Flight` GitHub Action is the ultimate enforcer. A red check **blocks the
merge** — no override. It runs on every PR to `main` (and on push to `main`).

## What the gate does (all must pass)
1. **selftest** (`tests/selftest.py`) — proves `data_pipeline.py` still has
   determinism, change-invalidation, and QC-FAIL-on-bad-data. Guards the code.
2. **verify_ledger** (`src/verify_ledger.py`) — the blocker:
   - `data/clean/manifest.json` must exist and carry a `dataset_hash`.
   - If raw data is present (fetched in CI), the hash is **recomputed and must
     equal** the committed manifest (no hand-edited manifests).
   - **Every active (non-STALE) row in `RESULTS_LEDGER.md` must carry the current
     `dataset_hash`.** If the data changed, old rows are invalid until re-run or
     marked STALE. This enforces change-invalidation at merge time.
   - Active trial count must be `<= K_MAX (20)`.

## Jules — engineering-plane PR checklist (do this, in order)
1. Branch off `main`. Make the change (bug fix, pipeline feature, CI, refactor).
2. If you touched anything under `src/` or the data: run locally first
   - `make ci`   (== selftest + verify; identical to the Action)
3. If your change alters `data/` or the cleaning logic (SCHEMA_VERSION):
   - run `make preflight`, commit the new `data/clean/manifest.json`,
   - and either **re-run** every affected study row or mark it `STALE` in the
     ledger. A stale hash on an active row = blocked merge.
4. Open the PR. Do **not** merge until `preflight-gate` is green.
5. Never edit a past ledger row except to flip `status` to `STALE`.

## Ruflo — research-plane orchestration
- Ruflo may only **stage** research runs. A run becomes real when `study_engine.py`
  output is committed as a ledger row via a PR that passes the gate.
- Ruflo must refuse to dispatch a run that would push active trials over K=20.
- One config + one dataset_hash = one row. Re-running an identical (config, hash)
  pair is a cache hit — reuse the row, do not add to K.

## Branch protection (repo admin, one-time)
Settings → Branches → protect `main`:
- [x] Require status checks to pass → **preflight-gate**
- [x] Require branches up to date before merging
- [x] Do not allow bypass (include admins)

## Optional: enable hash RECOMPUTE in CI
Add `scripts/fetch_data.sh` that pulls the raw panel from your store using repo
secrets `DATA_URL` / `DATA_TOKEN`. With it, CI recomputes and compares the hash;
without it, CI enforces the committed manifest + ledger lock (still airtight for
the invalidation rule).
