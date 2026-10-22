.PHONY: setup lint format typecheck test reproduce report serve

UV ?= uv

setup:
	$(UV) venv --python 3.13 .venv --allow-existing
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest --cov=reactive_lob_sim --cov-report=term

reproduce:
	$(UV) run sim reproduce

report:
	$(UV) run sim report

serve:
	$(UV) run sim serve
