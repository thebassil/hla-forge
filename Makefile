VENV := .venv
PY := $(VENV)/bin/python
HLA := $(VENV)/bin/hla

.PHONY: setup test smoke validate splits baselines plm report clean

setup:
	uv venv --python 3.12 $(VENV)
	uv pip install -e ".[dev,plm]"

test:
	$(PY) -m pytest -q

smoke:
	$(HLA) validate
	$(HLA) run configs/90_smoke.yaml --no-save
	$(PY) -m pytest -q

validate:
	$(HLA) validate

splits:
	$(HLA) splits --folds 5

baselines:
	$(HLA) sweep configs/00_mean_baseline.yaml configs/01_cheap_blosum_ridge.yaml configs/02_cheap_blosum_xgb.yaml

embed:
	$(HLA) embed --model esm2_t12 --pooling mean
	$(HLA) embed --model esm2_t12 --pooling flatten

plm: embed
	$(HLA) sweep configs/03_plm_esm2t12_ridge.yaml configs/04_plm_esm2t12_xgb.yaml configs/05_plm_interaction_all.yaml configs/06_plm_flatten_residue.yaml configs/07_hybrid_esm_blosum_xgb.yaml

report:
	$(HLA) report

clean:
	rm -f artifacts/results/*.json

clean-embeddings:
	rm -f artifacts/embeddings/*.npz
