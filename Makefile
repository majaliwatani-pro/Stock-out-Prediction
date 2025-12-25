.PHONY: setup generate train api docker-test test lint

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

generate:
	python data/synthetic_generator.py --out data/sample_data.csv --start 2023-01-01 --days 365 --n_stores 10 --n_items 50

train:
	python src/train.py --data-path data/sample_data.csv --model-out models/stockout_model.pkl --horizon 7

api:
	uvicorn src.predict_api:app --reload

docker-test:
	docker-compose up --build

test:
	pytest -q
