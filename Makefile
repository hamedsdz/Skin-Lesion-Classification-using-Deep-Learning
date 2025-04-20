.PHONY: setup clean train test lint

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements/dev.txt

# Clean Python cache files and logs
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.pyc" -delete
	find . -type d -name "*.pyo" -delete
	find . -type d -name "*.pyd" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	rm -rf logs/*
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

# Train the model
train:
	python src/train/train.py

# Run tests
test:
	pytest tests/

# Run linting
lint:
	flake8 src/
	black src/ --check
	isort src/ --check-only

# Format code
format:
	black src/
	isort src/

# Create required directories
init:
	mkdir -p logs models

# Start TensorBoard
tensorboard:
	tensorboard --logdir logs

