# 🏠 GharIQ — India Home Value Estimator

[![CI](https://github.com/Akhil-60/gharIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/Akhil-60/gharIQ/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/akhil060/GharIQ)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)

ML-powered estimator for Indian residential property prices. Enter a city, locality,
size and configuration, and GharIQ returns a market value with a **likely range**, a
**per-sq.ft rate**, a **comparison to the city average**, an **EMI estimate**, and the
value **indexed to 2025** using an official RBI/BIS house-price index.

Built end-to-end: data cleaning -> model -> REST API -> web UI, containerised with Docker,
tested via GitHub Actions CI, and deployed on Hugging Face Spaces.

🔗 **Live demo:** https://huggingface.co/spaces/akhil060/GharIQ

---

## ✨ Features

- **Price estimate + likely range** instead of a single misleading number
- **Per-sq.ft rate** and **comparison to the city's median** (e.g. "12% above Mumbai avg")
- **Typical prices** for the chosen city (1/2/3 BHK medians, from real listings)
- **RBI/BIS price-trend chart** (2018 -> 2025) and an **"Adjust to 2025 prices"** toggle
- **Compare two homes** side by side with a price-difference verdict
- **EMI estimate** with adjustable loan tenure (15 / 20 / 25 yr)
- City-aware **locality dropdown** populated from the dataset

## 🧰 Tech stack

Python · scikit-learn (`HistGradientBoostingRegressor`) · FastAPI · Uvicorn · pandas ·
Docker · GitHub Actions · Hugging Face Spaces

## ⚙️ How it works

1. **Data** — real Indian-listings datasets are normalised, cleaned and combined
   (details below).
2. **Model** — a gradient-boosting regressor (`HistGradientBoostingRegressor`) with
   one-hot encoded city/locality and a log-transformed price target.
3. **API** — FastAPI serves `/predict` (JSON in -> price out) and `/` (the web UI).

## 📊 Data & how the model was trained

### Where the data comes from

The model is trained on **real, publicly available property-listing datasets** sourced
from **Kaggle and other open data platforms** — not synthetic or auto-generated data.
A separate official index from the **U.S. Federal Reserve's FRED** (BIS Residential
Property Price Index for India) powers the trend chart and the 2025 price adjustment.

| Source | What it is | Platform | Vintage |
|--------|------------|----------|---------|
| Scraped residential listings (MagicBricks-style) | city, locality, area, bathrooms, price | Kaggle | ~2023 |
| "Predicting House Prices in India" dataset (Quikr-style) | BHK, sqft, address, price | MachineHack / Kaggle | ~2020 |
| Residential Property Price Index for India (BIS) | quarterly house-price index | FRED | 2009 → 2025 |

➡️ **The model is trained on listings spanning roughly `2020–2023`** (about 3 years of
real market data). like Kaggle , HF and other Platform 

### Working with *genuine* data only

While building this, several "India house price" datasets were evaluated — and **two
were rejected because they were synthetic (fake), not real market data**:

- One had **placeholder localities** (`Locality_84`, `Locality_490`) and a tell-tale flaw:
  1-BHK and 3-BHK homes had the *same* average size — i.e. size was random and carried
  no real signal.
- Another was the well-known **Seattle (King County) dataset relabelled** with Indian
  city names, containing impossible rows (a 5-bedroom home at 353 sq.ft).

Training on fake data would have produced a meaningless model, so only verified-real
datasets were used. (This data-quality check is documented in `DATA_SOURCES.md`.)

### How the raw data was cleaned (pipeline)

Raw listings are messy, so the data goes through a reproducible cleaning pipeline before
training:

1. **Parse** messy text fields — e.g. `"42 Lac"` / `"1.40 Cr"` → rupees, `"2 BHK"` from
   titles, `"473 sqft"` → number, and `ADDRESS` → city + locality.
2. **Normalise** everything to one schema: `city, locality, bhk, bathrooms, area_sqft, price_inr`.
3. **Drop** rows missing essentials (city, BHK, area, price).
4. **Range-filter** out impossible values (e.g. 100–20,000 sq.ft, ₹1 Lakh–₹1,000 Cr).
5. **De-duplicate** — removed ~115,000 duplicate listings from the scraped source.
6. **Outlier-trim** the extreme 2% of price-per-sq.ft on each end.

The final cleaned dataset has **~84,000 rows**.

### Model performance
Test **R² ≈ 0.79**. The single biggest gain came from cleaning, not from more data:
removing the price-per-sq.ft outliers (step 6) lifted R² from **~0.10 → ~0.79**. This is
the project's key lesson — **clean, genuine data beats more data.**

## 📁 Project structure

```
.
├── data/                       # cleaned, combined dataset (CSV)
├── models/                     # trained model.joblib (Git LFS)
├── src/
│   ├── train.py                # trains the model and saves model.joblib
│   ├── merge_data.py           # combines + de-duplicates datasets
│   └── api.py                  # FastAPI app (REST API + web UI)
├── tests/                      # smoke tests
├── .github/workflows/ci.yml    # CI: install + test on every push
├── Dockerfile
├── requirements.txt
└── DATA_SOURCES.md
```

## 🚀 Run locally

> Prefer zero setup? The **[live demo](https://huggingface.co/spaces/akhil060/GharIQ)**
> is always on. The steps below are for running your *own* copy on your machine.

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

> ℹ️ **Note:** `127.0.0.1` (localhost) only works on *your own machine while the server
> is running* — it is not a public link, so it won't load if the server is stopped or
> the laptop is off. To try GharIQ anytime without installing anything, use the
> always-on **[live demo on Hugging Face](https://huggingface.co/spaces/akhil060/GharIQ)**.

## 🐳 Run with Docker

```bash
docker build -t ghariq .
docker run -p 8000:7860 ghariq
```

## 📡 API usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"Mumbai","locality":"Unknown","bhk":2,"bathrooms":2,"area_sqft":1000}'
# -> {"predicted_price_inr": 22398657.0}
```

## ⚠️ Limitations

- Trained on **2020-2023** listings (asking prices, not closing prices); estimates are
  optionally indexed to 2025 but are a **ballpark, not a valuation**.
- Final features don't include floor, age, view or furnishing.
- For genuinely live prices, a property-portal data feed + retraining pipeline would be
  needed — out of scope for this snapshot-based project.

## 👤 Author

**Akhil Kumar** — [GitHub @Akhil-60](https://github.com/Akhil-60)
