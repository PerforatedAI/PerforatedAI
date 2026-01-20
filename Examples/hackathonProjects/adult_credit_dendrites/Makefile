PYTHON ?= python

format:
	$(PYTHON) -m black .

test:
	$(PYTHON) test_setup.py

sweep:
	$(PYTHON) run_sweep.py

.PHONY: format test sweep
