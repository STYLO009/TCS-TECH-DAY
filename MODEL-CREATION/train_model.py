import pandas as pd
import numpy as np
import joblib
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

# Get the folder where train_model.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the CSV path automatically
FILE_PATH = os.path.join(BASE_DIR, "supply_chain_data.csv")

print("Looking for dataset at:")
print(FILE_PATH)

# Check whether file exists
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"\nDataset not found!\nExpected location:\n{FILE_PATH}"
    )

df = pd.read_csv(FILE_PATH)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())
# ============================================================
# 2. CLEAN DATA
# ============================================================

# Convert Value to numeric
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Convert YYYYMM to string first
df["YYYYMM"] = df["YYYYMM"].astype(str)

# Convert YYYYMM into date
df["Date"] = pd.to_datetime(
    df["YYYYMM"],
    format="%Y%m",
    errors="coerce"
)

# Remove invalid rows
df = df.dropna(subset=["Date", "Value"])

# Sort chronologically
df = df.sort_values(["MSN", "Date"]).reset_index(drop=True)


# ============================================================
# 3. CREATE TIME FEATURES
# ============================================================

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Sequential time index
df["Time_Index"] = (
    df["Year"] * 12 + df["Month"]
)

# Create lag features
# Previous month's emission
df["Lag_1"] = df.groupby("MSN")["Value"].shift(1)

# Previous 2 months
df["Lag_2"] = df.groupby("MSN")["Value"].shift(2)

# Previous 3 months
df["Lag_3"] = df.groupby("MSN")["Value"].shift(3)

# Rolling average of previous 3 months
df["Rolling_Mean_3"] = (
    df.groupby("MSN")["Value"]
      .shift(1)
      .rolling(3)
      .mean()
      .reset_index(level=0, drop=True)
)


# ============================================================
# 4. REMOVE ROWS WITH MISSING LAG VALUES
# ============================================================

df = df.dropna(
    subset=[
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Rolling_Mean_3"
    ]
).reset_index(drop=True)


# ============================================================
# 5. SELECT FEATURES
# ============================================================

features = [
    "Year",
    "Month",
    "Time_Index",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_Mean_3"
]

target = "Value"

X = df[features]
y = df[target]


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# For time-series data we should NOT randomly shuffle data.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. CREATE XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===================================")
print("       FEATURE IMPORTANCE")
print("===================================")

print(importance.to_string(index=False))


# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

results = df.iloc[split_index:].copy()

results["Predicted_Value"] = predictions

results.to_csv(
    "prediction_results.csv",
    index=False
)

print("\nPrediction results saved:")
print("prediction_results.csv")


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "carbon_emission_model.pkl"
)

print("\nModel saved:")
print("carbon_emission_model.pkl")


# ============================================================
# 14. SAVE FEATURE INFORMATION
# ============================================================

joblib.dump(
    features,
    "model_features.pkl"
)

print("Feature information saved:")
print("model_features.pkl")


# ============================================================
# 15. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n===================================")
print("       SAMPLE PREDICTIONS")
print("===================================")

sample = results[
    [
        "Date",
        "MSN",
        "Description",
        "Value",
        "Predicted_Value",
        "Unit"
    ]
].head(10)

print(sample.to_string(index=False))


print("\n===================================")
print("       TRAINING COMPLETE")
print("===================================")