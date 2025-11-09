"""
predict.py — Energy404 Deployment Inference Pipeline
----------------------------------------------------
Loads pre-trained ensemble models (LGBM + XGB + RF + ET + Ridge meta)
and predicts annual rooftop solar energy potential (kWh/m²)
for a given city, building type, and roof tilt.

Usage example:
--------------
>>> from predict import predict_energy
>>> predict_energy(city="Accra", building_type="commercial", tilt=25)
268.4  # predicted kWh/m² per year
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models_local_backup"
DATA_DIR = BASE_DIR / "data"

# === Load model artifacts ===
print("🔹 Loading trained model components...")
lgb_models = joblib.load(MODELS_DIR / "lgb_models.pkl")
xgb_models = joblib.load(MODELS_DIR / "xgb_models.pkl")
rf_models  = joblib.load(MODELS_DIR / "rf_models.pkl")
et_models  = joblib.load(MODELS_DIR / "et_models.pkl")
meta_model = joblib.load(MODELS_DIR / "meta_model.pkl")
config     = joblib.load(MODELS_DIR / "feature_config.pkl")

NUM = config["NUM"]
CAT = config["CAT"]
building_categories = config["BuildingType_categories"]

# === Load city-level weather data ===
weather_path = DATA_DIR / "city_weather.csv"
weather_df = pd.read_csv(weather_path)

# === Core prediction function ===
def predict_energy(city: str, building_type: str, tilt: float) -> float:
    """
    Predict rooftop solar potential (kWh/m²/year)
    for the given city, building type, and roof tilt.
    """

    # --- Validate inputs ---
    if city not in weather_df["City"].values:
        raise ValueError(f"❌ City '{city}' not found in city_weather.csv")
    if building_type not in building_categories:
        raise ValueError(f"❌ BuildingType '{building_type}' not recognized")

    # --- Lookup city weather ---
    row = weather_df.loc[weather_df["City"] == city].iloc[0]
    GHI = row["avg_GHI_kWhm2_day"]
    Temp = row["avg_temp_C"]
    Clear = row["clearness_index"]
    Precip = row["precip_mm_day"]

    # --- Compute tilt features ---
    tilt2 = tilt ** 2
    tilt_sin = np.sin(np.radians(tilt))
    tilt_cos = np.cos(np.radians(tilt))

    # --- Create full feature dict ---
    feats = {
        "tilt": tilt,
        "tilt2": tilt2,
        "tilt_sin": tilt_sin,
        "tilt_cos": tilt_cos,
        "GHI_kWh_per_m2_day": GHI,
        "AvgTemp_C": Temp,
        "ClearnessIndex": Clear,
        "Precip_mm_per_day": Precip,
        # interactions (same as training)
        "tilt_x_GHI": tilt * GHI,
        "temp_sq": Temp ** 2,
        "clear_x_tiltcos": Clear * tilt_cos,
        "precip_x_clear": Precip * (1.0 - Clear),
        "BuildingType": building_type,
    }

    X = pd.DataFrame([feats])
    X["BuildingType"] = pd.Categorical(X["BuildingType"], categories=building_categories)

    # Encoded version for models that need numeric cat
    X_enc = X.copy()
    X_enc["BuildingType"] = X_enc["BuildingType"].cat.codes

    # --- Generate predictions from each model ---
    pred_lgb = np.mean([np.expm1(m.predict(X)) for m in lgb_models], axis=0)
    pred_xgb = np.mean([np.expm1(m.predict(X_enc)) for m in xgb_models], axis=0)
    pred_rf  = np.expm1(rf_models[0].predict(X_enc))
    pred_et  = np.expm1(et_models[0].predict(X_enc))

    # --- Meta prediction (Ridge ensemble) ---
    meta_X = np.column_stack([pred_lgb, pred_xgb, pred_rf, pred_et])
    pred_final = meta_model.predict(meta_X)[0]

    return round(float(pred_final), 3)


# === Optional: quick test when run standalone ===
if __name__ == "__main__":
    test_city = "Accra"
    test_type = "commercial"
    test_tilt = 20

    pred = predict_energy(test_city, test_type, test_tilt)
    print(f"\n☀️ Predicted Solar Potential for {test_city} ({test_type}, tilt={test_tilt}°): {pred} kWh/m²/year\n")
