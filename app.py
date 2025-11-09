# === app.py ===
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# === Ensure we can import from pipeline/ ===
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "pipeline"))

from predict import predict_energy  # ✅ import your model inference function


# === Load static data (city weather) ===
city_df = pd.read_csv(BASE_DIR / "data" / "city_weather.csv")

# Sort cities alphabetically for dropdown
available_cities = sorted(city_df["City"].unique())

# Define available building types (must match model training)
building_types = [
    "single family residential",
    "multifamily residential",
    "commercial",
    "small commercial",
    "industrial",
    "public sector",
    "schools",
    "public health facilities",
    "hotels",
    "peri-urban settlement",
]

# === Streamlit UI ===
st.set_page_config(page_title="☀️ Energy404: Rooftop Solar Potential Predictor", layout="centered")

st.title("☀️ Energy404: Rooftop Solar Potential Predictor")
st.markdown(
    """
    Use this tool to estimate annual **solar energy potential (kWh/m²/year)** 
    for rooftops across 20 global cities using the tuned LGBM+XGB+RF ensemble model.
    """
)

# === Sidebar inputs ===
st.sidebar.header("🔧 Input Parameters")

selected_city = st.sidebar.selectbox("Select City", available_cities)
selected_type = st.sidebar.selectbox("Select Building Type", building_types)
tilt = st.sidebar.slider("Tilt (degrees)", min_value=0, max_value=60, value=20)
surface_area = st.sidebar.number_input("Roof Surface Area (m²)", min_value=10, max_value=1000, value=100)

# === Fetch weather data for selected city ===
row = city_df.loc[city_df["City"] == selected_city].iloc[0]
GHI = row["avg_GHI_kWhm2_day"]
TEMP = row["avg_temp_C"]
CLEAR = row["clearness_index"]
PRECIP = row["precip_mm_day"]

# === Prediction button ===
if st.sidebar.button("Predict Solar Potential"):
    with st.spinner("Predicting... ☀️"):
        prediction = predict_energy(
    city=selected_city,
            building_type=selected_type,
            tilt=tilt
        )

        annual_energy = prediction * 1.0  # already kWh/m²/year

        total_energy = annual_energy * surface_area

        st.success(f"☀️ **Predicted Solar Potential for {selected_city} ({selected_type}, tilt={tilt}°):**")
        st.metric("Energy per m²", f"{annual_energy:.2f} kWh/m²/year")
        st.metric("Total Roof Output", f"{total_energy:.1f} kWh/year")
        st.caption("Estimated using ensemble model (LGBM + XGB + RF)")

st.markdown("---")
st.markdown(
    """
    🌍 *Rooftop Solar Potential Predictor*  
    **Team: Energy404** | November 2025
    """
)
