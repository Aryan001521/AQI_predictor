import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# ===============================
# DARK THEME CSS
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}

h1, h2, h3, h4, p, span, label {
    color: #e5e7eb !important;
}

.metric-card {
    background: #020617;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 0 25px rgba(0,255,255,0.15);
    text-align: center;
    border: 1px solid #1e293b;
}

.metric-value {
    font-size: 42px;
    font-weight: 700;
    color: #38bdf8;
}

.metric-label {
    font-size: 14px;
    color: #94a3b8;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & ENCODERS
# ===============================
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
city_encoder_path = os.path.join(BASE_DIR, "models", "labelencoder_city.pkl")
location_encoder_path = os.path.join(BASE_DIR, "models", "labelencoder_location.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load models
model = joblib.load(model_path)
le_city = joblib.load(city_encoder_path)
le_location = joblib.load(location_encoder_path)
scaler = joblib.load(scaler_path)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Input Parameters")

city = st.sidebar.selectbox("City", list(le_city.classes_))
location = st.sidebar.selectbox("Location", list(le_location.classes_))

date = st.sidebar.date_input("Date", datetime.now())
time = st.sidebar.time_input("Time", datetime.now().time())

temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 70.0)
pressure = st.sidebar.slider("Pressure (hPa)", 900.0, 1100.0, 990.0)
wind_speed = st.sidebar.slider("Wind Speed", 0.0, 20.0, 1.0)
wind_direction = st.sidebar.slider("Wind Direction", 0.0, 360.0, 180.0)

pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 120.0)
pm10 = st.sidebar.slider("PM10", 0.0, 500.0, 160.0)
no2 = st.sidebar.slider("NO2", 0.0, 500.0, 45.0)
so2 = st.sidebar.slider("SO2", 0.0, 500.0, 20.0)
o3 = st.sidebar.slider("O3", 0.0, 500.0, 55.0)
co = st.sidebar.slider("CO", 0.0, 50.0, 10.0)

# ===============================
# FEATURE ENGINEERING
# ===============================
dt = datetime.combine(date, time)

def lag(val): 
    return val, val, val, val

pm25_l1, pm25_l2, pm25_l3, pm25_r = lag(pm25)
pm10_l1, pm10_l2, pm10_l3, pm10_r = lag(pm10)
o3_l1, o3_l2, o3_l3, o3_r = lag(o3)
no2_l1, no2_l2, no2_l3, no2_r = lag(no2)
so2_l1, so2_l2, so2_l3, so2_r = lag(so2)
co_l1, co_l2, co_l3, co_r = lag(co)

input_df = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "pressure": pressure,
    "wind_speed": wind_speed,
    "wind_direction": wind_direction,
    "pm25": pm25,
    "pm10": pm10,
    "no2": no2,
    "so2": so2,
    "o3": o3,
    "co": co,

    "pm25_lag1": pm25_l1, "pm25_lag2": pm25_l2, "pm25_lag3": pm25_l3, "pm25_roll3": pm25_r,
    "pm10_lag1": pm10_l1, "pm10_lag2": pm10_l2, "pm10_lag3": pm10_l3, "pm10_roll3": pm10_r,
    "o3_lag1": o3_l1, "o3_lag2": o3_l2, "o3_lag3": o3_l3, "o3_roll3": o3_r,
    "no2_lag1": no2_l1, "no2_lag2": no2_l2, "no2_lag3": no2_l3, "no2_roll3": no2_r,
    "so2_lag1": so2_l1, "so2_lag2": so2_l2, "so2_lag3": so2_l3, "so2_roll3": so2_r,
    "co_lag1": co_l1, "co_lag2": co_l2, "co_lag3": co_l3, "co_roll3": co_r,

    "Year": dt.year,
    "Month": dt.month,
    "Day": dt.day,
    "Dayofweek": dt.weekday(),
    "Hour": dt.hour,
    "Hour_sin": np.sin(2*np.pi*dt.hour/24),
    "Hour_cos": np.cos(2*np.pi*dt.hour/24),

    "location_id_encoded": le_location.transform([location])[0],
    "city_encoded": le_city.transform([city])[0]
}])

input_scaled = scaler.transform(input_df)
predicted_aqi = model.predict(input_scaled)[0]

# ===============================
# AQI CATEGORY
# ===============================
def aqi_cat(a):
    if a <= 50: return "Good 🟢"
    elif a <= 100: return "Satisfactory 🟡"
    elif a <= 200: return "Moderate 🟠"
    elif a <= 300: return "Poor 🔴"
    elif a <= 400: return "Very Poor 🟣"
    else: return "Severe ⚫"

# ===============================
# MAIN UI
# ===============================
st.title("🌫️ Air Quality Index Prediction")
st.caption("Dark Mode | ML Powered AQI Forecasting System")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted AQI</div>
        <div class="metric-value">{predicted_aqi:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">AQI Category</div>
        <div class="metric-value">{aqi_cat(predicted_aqi)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Location</div>
        <div class="metric-value">{location}</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# GRAPH SECTION
# ===============================
st.markdown("## 📊 Pollutant Contribution Snapshot")

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]
values = [pm25, pm10, no2, so2, o3, co]

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(pollutants, values)
ax.set_facecolor("#020617")
fig.patch.set_facecolor("#020617")
ax.tick_params(colors="white")
ax.set_ylabel("Concentration", color="white")
ax.set_title("Current Pollutant Levels", color="white")

st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ using XGBoost & Streamlit")

