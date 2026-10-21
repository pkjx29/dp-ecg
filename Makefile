# Full pipeline:  make install data verify stats train attack report
PY ?= .venv/bin/python
CLI = $(PY) -m dp_ecg.cli
export PYTHONPATH := $(CURDIR)/src

install:
	python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests
data:
	$(CLI) download && $(CLI) prepare
verify:
	$(CLI) verify-mechanisms
stats:
	$(CLI) summary-stats
train:
	$(CLI) train-baseline && $(CLI) train-dp
attack:
	$(CLI) attack
report:
	$(CLI) report
smoke:
	$(CLI) smoke
serve:
	$(CLI) serve
all: data verify stats train attack report
.PHONY: install test lint data verify stats train attack report smoke serve all
