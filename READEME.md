Predict inventory stock-outs (shortages) at item × store level using ML pipeline.

This repository includes:
- A synthetic dataset generator (useful for local testing).
- End-to-end pipeline (feature engineering, training, evaluation).
- FastAPI prediction service.
- Docker + docker-compose setup for local/dev deployment.
- GitHub Actions CI for tests.
- Example configuration for batch scheduling and monitoring hooks.

Quick links
- How to run locally (PyCharm / CLI): see "Run locally"
- Docker instructions: see "Docker"
- Use a public dataset: see "Using public datasets (Kaggle)"
- Portfolio article: PORTFOLIO_ARTICLE.md

Table of Contents
- Project layout
- Requirements
- Run locally (minimal)
- Generate synthetic dataset (built-in)
- Train model
- Run API
- Docker & docker-compose
- CI & tests
- Using real/open-source datasets
- Production considerations & next steps
- License

Project layout
- data/
  - synthetic_generator.py       # script to create sample datasets
  - sample_data.csv              # small sample (committed)
- notebooks/                     # optional EDA and reporting notebooks
- src/
  - data_prep.py                 # dataset ingestion & label creation
  - feature_engineering.py       # features used for modeling
  - model.py                     # train / evaluate / save / load
  - train.py                     # training entrypoint
  - predict_api.py               # FastAPI inference app
  - config.py                    # central config
- models/                        # saved models after training
- tests/                         # unit tests
- Dockerfile
- docker-compose.yml
- requirements.txt
- Makefile
- .github/workflows/ci.yml
- README.md
- PORTFOLIO_ARTICLE.md

Requirements
- Python 3.9+
- Linux/Mac/Windows (PyCharm supported)
- Optional: Docker & docker-compose for containerized run

Install (recommended in venv)
pip install -r requirements.txt

Run locally (minimal)
1. Generate a dataset (synthetic)
   python data/synthetic_generator.py --out data/sample_data.csv --start 2023-01-01 --days 365 --n_stores 10 --n_items 50

2. Train
   python src/train.py --data-path data/sample_data.csv --model-out models/stockout_model.pkl --horizon 7

3. Run API (after training)
   uvicorn src.predict_api:app --reload --host 0.0.0.0 --port 8000
   Example call:
     POST http://127.0.0.1:8000/predict
     JSON body example:
     {
       "store_id": 1,
       "item_id": 10,
       "date": "2024-01-05",
       "sales_lag_1": 5,
       "sales_rmean_7": 6.2,
       "days_of_cover": 2.5,
       "on_promotion": 0
     }

Generate synthetic dataset (what it does)
- Creates daily sales for N stores × M items.
- Simulates random shipments (replenishments) and computes stock_on_hand using inventory flow.
- Includes promotions, price, holiday flags.
- Saves a CSV with columns:
  date, store_id, item_id, sales, stock_on_hand, shipments_received, price, on_promotion, is_holiday

Using public datasets (Kaggle)
- This repo supports adapting Kaggle datasets (Rossmann, Favorita, M5). Kaggle requires API credentials to download programmatically.
- See scripts/download_kaggle.sh for instructions and how to place CSVs in `data/`.
- If your dataset lacks stock_on_hand, use the synthetic generator approach or run the inventory simulation to create stock_on_hand from receipts + sales (sample code included in src/data_prep.py).

Docker
- Build image:
  docker build -t stockout-app:latest .
- Run with docker-compose:
  docker-compose up --build
- The API will be available at http://localhost:8000

CI & tests
- GitHub Actions workflow runs tests in `tests/` on push.
- Local run:
  pytest -q

Production considerations
- Data access: use secure connections to databases, not CSVs; store secrets in a vault.
- Model registry: store artifacts in S3 / MLFlow.
- Scheduling: use Airflow/Prefect for orchestration.
- Monitoring: log predictions, track drift (feature distributions), and business metrics.
- Retraining policy: automated retraining on new data, with validation & gating.
