PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test demo reproduce live live-smoke
test:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) -m weather_risk --mode demo --config config/demo.json

reproduce: test demo

live:
	$(PYTHON) -m weather_risk --mode live --config config/live.json

live-smoke:
	$(PYTHON) -m weather_risk --mode live --config config/smoke.json
