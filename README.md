# 🏠 GharIQ — India Home Value Estimator

ML-powered estimator for Indian residential property prices. Enter a city, locality,
size and configuration, and GharIQ returns a market value with a **likely range**, a
**per-sq.ft rate**, a **comparison to the city average**, an **EMI estimate**, and the
value **indexed to 2025** using an official RBI/BIS house-price index.

Built end-to-end: data cleaning → model → REST API → web UI, containerised with Docker
and tested via GitHub Actions CI.

> 🔗 **Live demo:** _add your Hugging Face / Render link here after deploy_

---

## ✨ Features

- **Price estimate + likely range** instead of a single misleading number
- **Per-sq.ft rate** and **comparison to the city's median** (e.g. "12% above Mumbai avg")
- **Typical prices** for the chosen city (1/2/3 BHK medians, from real listings)
- **RBI/BIS price-trend chart** (2018 → 2025) and an **"Adjust to 2025 prices"** toggle
- **Compare two homes** side by side with a price-difference verdict
- **EMI estimate** with adjustable loan tenure (15 / 20 / 25 yr)
- City-aware **locality dropdown** populated from the dataset

## 🧰 Tech stack

Python · scikit-learn (`HistGradientBoostingRegressor`) · FastAPI · Uvicorn · pandas ·
Docker · GitHub Actions

## ⚙️ How it works

1. **Data** — two public Indian-listings datasets are normalised to a single schema
   (`city, locality, bhk, bathrooms, area_sqft, price_inr`), de-duplicated and
   outlier-trimmed. See [`DATA_SOURCES.md`](DATA_SOURCES.md).
2. **Model** — a gradient-boosting regressor with one-hot encoded city/locality and a
   log-transformed price target.
3. **API** — FastAPI serves `/predict` (JSON in → price out) and `/` (the web UI).

### Model performance
Test **R² ≈ 0.79**. Notably, cleaning the data (removing the extreme 2% of
price-per-sq.ft outliers) lifted R² from **~0.10 → ~0.79** — most of the quality came
from data cleaning, not from adding more rows.

## 📁 Project structure

```
.
├── data/                       # cleaned, combined dataset (CSV)
├── models/                     # trained model.joblib
├── src/
│   ├── train.py                # trains the model and saves model.joblib
│   ├── merge_data.py           # combines + de-duplicates datasets
│   └── api.py                  # FastAPI app (REST API + web UI)
├── tests/                      # smoke tests
├── .github/workflows/ci.yml    # CI: install, test on every push
├── Dockerfile
├── requirements.txt
└── DATA_SOURCES.md
```

## 🚀 Run locally

```bash
# 1. create + activate a virtual environment (Python 3.12)
python -m venv venv
venv\Scripts\activate        # Windows PowerShell
# source venv/bin/activate   # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) retrain the model
python src/train.py

# 4. start the server
uvicorn src.api:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## 🐳 Run with Docker

```bash
docker build -t ghariq .
docker run -p 8000:7860 ghariq
```

## ⚠️ Limitations

- Trained on **2020–2023** listings (asking prices, not closing prices); estimates are
  optionally indexed to 2025 but are a **ballpark, not a valuation**.
- Final features don't include floor, age, view or furnishing.
- For genuinely live prices, a property-portal data feed + retraining pipeline would be
  needed — out of scope for this snapshot-based project.

## 👤 Author

**Akhil Kumar** — [GitHub @Akhil-60](https://github.com/Akhil-60)
