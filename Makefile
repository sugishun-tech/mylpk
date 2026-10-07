PYTHON ?= python

.PHONY: build test benchmark examples validate package

build:
	$(PYTHON) setup.py build_ext --inplace

test: build
	PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q --junitxml=reports/pytest.xml

benchmark: build
	PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) benchmarks/run.py

examples: build
	@for f in examples/*.py; do PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) "$$f" || exit 1; done

validate:
	$(PYTHON) tools/validate_wheel.py

package:
	$(PYTHON) tools/package.py
