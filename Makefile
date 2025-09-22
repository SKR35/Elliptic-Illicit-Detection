venv:
	python -m venv .venv

install:
	pip install -r requirements.txt
	pre-commit install

train:
	python -m src.elliptic.train --config config/train.yaml

eval:
	python -m src.elliptic.evaluate --model_path models/registry/best_pipeline.joblib --config config/train.yaml

format:
	black .
	ruff --fix .