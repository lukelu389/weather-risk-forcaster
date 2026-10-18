PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test demo reproduce live live-smoke eda baselines ml evaluate sensitivity portfolio history verify-results test-full package-source
test:
	$(PYTHON) scripts/run_tests.py

demo:
	$(PYTHON) -m weather_risk --mode demo --config config/demo.json

reproduce: test demo

live:
	$(PYTHON) -m weather_risk --mode live --config config/live.json

live-smoke:
	$(PYTHON) -m weather_risk --mode live --config config/smoke.json

EDA_PYTHON ?= .venv/bin/python
eda:
	$(EDA_PYTHON) -m weather_risk.eda --allow-latest-vintage $(if $(INPUT_RUN),--input-run $(INPUT_RUN))

baselines:
	$(EDA_PYTHON) -m weather_risk.baselines --allow-latest-vintage $(if $(INPUT_RUN),--input-run $(INPUT_RUN))

ml:
	$(EDA_PYTHON) -m weather_risk.ml --allow-latest-vintage

evaluate:
	$(EDA_PYTHON) -m weather_risk.final_evaluation --allow-latest-vintage

sensitivity:
	$(EDA_PYTHON) -m weather_risk.bootstrap_sensitivity

portfolio:
	$(EDA_PYTHON) -m weather_risk.portfolio

history:
	$(EDA_PYTHON) -m weather_risk.historical

# Core reviewer path: uses only the standard library and published derived evidence.
verify-results:
	$(PYTHON) scripts/verify_results.py

test-full:
	$(PYTHON) scripts/run_tests.py --require-no-skips

# Requires a clean, reviewed commit. Use a new OUTPUT name for each release.
OUTPUT ?= dist/weather-risk-forecaster-source.zip
package-source:
	$(PYTHON) scripts/package_source.py --output "$(OUTPUT)"
