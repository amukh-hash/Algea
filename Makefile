fmt:
	black .

lint:
	ruff check .

test:
	pytest
