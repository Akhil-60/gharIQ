import pandas as pd

EXISTING = "data/combined_house_prices.csv"
EXTRA    = "data/house_prices_extra.csv"
OUTPUT   = "data/combined_house_prices.csv"

COLS = ["city", "locality", "bhk", "bathrooms", "area_sqft", "price_inr"]

old = pd.read_csv(EXISTING)
new = pd.read_csv(EXTRA)
print(f"existing rows: {len(old):,}   |   extra rows: {len(new):,}")

old = old[COLS]
new = new[COLS]

combined = pd.concat([old, new], ignore_index=True)
before = len(combined)

combined = combined.drop_duplicates().reset_index(drop=True)
removed = before - len(combined)
print(f"jodne ke baad: {before:,}   |   duplicate hatae: {removed:,}   |   final: {len(combined):,}")

combined["locality"] = combined["locality"].fillna("Unknown")

combined.to_csv(OUTPUT, index=False)
print(f"saved -> {OUTPUT}")