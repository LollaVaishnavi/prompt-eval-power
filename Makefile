PYTHON ?= python3
PYTHONPATH := src

.PHONY: lint test experiments receipts evidence verify clean

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check src tests experiments tools; \
	else \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/lint_check.py; \
	fi

test:
	@if command -v pytest >/dev/null 2>&1; then \
		PYTHONPATH=$(PYTHONPATH) pytest --junitxml=junit.xml --cov=prompt_eval_power --cov-report=json:coverage.json --cov-report=xml:coverage.xml; \
	else \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_*.py" -v; \
	fi

experiments:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/exp01_small_suite_false_pass_rate.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/exp02_paired_vs_unpaired_power_gain.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/exp03_zero_failure_resolution_floor.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/exp04_langgraph_prompt_regression_audit.py

receipts:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/collect_metrics.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/check_numbers.py

evidence:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/render_diagrams.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/capture_screenshots.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/record_demo.py

verify: lint test receipts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache reports junit.xml coverage.json coverage.xml .coverage
