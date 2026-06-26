import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/combined_house_prices.csv")
df["locality"] = df["locality"].fillna("Unknown")

X = df[["city", "locality", "bhk", "bathrooms", "area_sqft"]]
y = df["price_inr"]

pre = ColumnTransformer([
    ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
    ("loc", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), ["locality"]),
], remainder="passthrough")

est = HistGradientBoostingRegressor(random_state=42, max_iter=500, learning_rate=0.08)
reg = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
model = Pipeline([("pre", pre), ("reg", reg)])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)
print("R2 score:", round(model.score(X_te, y_te), 4))

joblib.dump(model, "models/model.joblib")
print("Saved -> models/model.joblib")