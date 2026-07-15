.PHONY: preflight verify selftest ci
preflight:      ## build panel + emit DATASET_HASH
	python src/data_pipeline.py --raw data/raw --out data/clean --freq D
selftest:       ## guard pipeline code (determinism/invalidation/QC-fail)
	python tests/selftest.py
verify:         ## the CI enforcer (ledger hash-lock + K budget)
	REPO_ROOT=$(PWD) python src/verify_ledger.py
ci: selftest verify   ## exactly what the GitHub Action runs
