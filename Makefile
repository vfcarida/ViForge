.PHONY: lock

lock:
	pip-compile pyproject.toml -o requirements/base.lock --generate-hashes --allow-unsafe
	pip-compile pyproject.toml --extra dev -o requirements/dev.lock --generate-hashes --allow-unsafe
	pip-compile pyproject.toml --extra dev --extra aws --extra inference -o requirements/all.lock --generate-hashes --allow-unsafe --pip-args "--prefer-binary"
