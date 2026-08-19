.PHONY: install lint format test typecheck clean lock

install:
	pip install -e .[dev]
	pre-commit install

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v --cov=src/viforge

typecheck:
	mypy src/ --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

lock:
	pip-compile pyproject.toml -o requirements/base.lock --generate-hashes --allow-unsafe
	pip-compile pyproject.toml --extra dev -o requirements/dev.lock --generate-hashes --allow-unsafe
	pip-compile pyproject.toml --extra dev --extra aws --extra inference -o requirements/all.lock --generate-hashes --allow-unsafe --pip-args "--prefer-binary"
