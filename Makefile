.PHONY: install lint format test demo clean

install:
	pip install -r requirements.lock.txt

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

demo:
	python examples/run_demo.py

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
