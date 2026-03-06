# Aspect-Based Sentiment Analysis — Amazon Electronics

Extracts structured sentiment triplets (aspect, opinion, polarity) from Amazon product reviews using PyABSA.

## Setup
```bash
conda create -n absa_env python=3.10
conda activate absa_env
pip install -r requirements.txt
```

## How to run
```bash
# Step 1 — preprocess (edit WANTED_ASINS in this file to change products)
python src/preprocessor.py

# Step 2 — run ASTE model (~30-40 min)
python src/run_absa.py

# Step 3 — aggregate results
python src/aggregate.py

# Step 4 — launch dashboard
streamlit run src/dashboard.py
# Open http://localhost:8501
```

## To change products
1. Open `src/preprocessor.py` — edit `WANTED_ASINS` list
2. Open `src/aggregate.py` — add product names to `PRODUCT_NAMES` dict
3. Re-run steps 1-4 above

## Project structure
```
absa_project/
├── data/
│   ├── raw/              ← download electronics_reviews.csv here (not in git)
│   └── processed/        ← auto-generated, not in git
├── output/
│   └── results/          ← auto-generated, not in git
├── src/
│   ├── preprocessor.py   ← Phase 2: filter & split by product
│   ├── run_absa.py       ← Phase 3: run ASTE model
│   ├── aggregate.py      ← Phase 4: combine results
│   └── dashboard.py      ← Phase 5: Streamlit dashboard
├── requirements.txt
└── README.md
```

## Data
Place `electronics_reviews.csv` in `data/raw/` before running.
The raw dataset is not included in git due to file size.
