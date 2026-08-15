.PHONY: install test fixture audit
install:
	pip install -e ".[dev]"
test:
	pytest -q
fixture:
	python scripts/make_fixture.py --output data/fixture
audit:
	bite2text all --data-root data/fixture --output-dir artifacts/fixture_run

