import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly


# CONFIGURATION
st.set_page_config(page_title="IT Resource Optimization", layout="wide")
st.title("IT Resource Optimization Dashboard")

# UPLOAD FILE
uploaded_file = st.file_uploader(" Upload Resource Usage CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    st.success(" File uploaded and loaded successfully.")
else:
    st.info(" Please upload a CSV file to continue.")
    st.stop()


# SIDEBAR
st.sidebar.header("Forecast Configuration")
metric = st.sidebar.selectbox(
    "Select Metric",
    [
        "cpu_usage",
        "memory_usage",
        "disk_io",
        "network_usage_mbps"
    ]
)
from datetime import datetime, timedelta

max_date = df["timestamp"].max().date()
st.sidebar.markdown(f" Data ends on: **{max_date}**")

# Date  selection
start_date = st.sidebar.date_input("Forecast Start Date", value=datetime(2024, 1, 1).date())
end_date = st.sidebar.date_input("Forecast End Date", value=datetime(2024, 1, 7).date())

# check date validity
if start_date >= end_date:
    st.sidebar.error("End date must be after start date.")
    st.stop()

forecast_start = pd.to_datetime(start_date)
forecast_end = pd.to_datetime(end_date)
periods = int((forecast_end - forecast_start).total_seconds() // 3600)  # convert to hours

# Cost modeling input
st.sidebar.header("Cost Settings ($/hour)")
cost_up = st.sidebar.number_input("Cost per hour (Scale UP)", value=5.0)
cost_down = st.sidebar.number_input("Cost per hour (Scale DOWN)", value=2.0)
cost_maintain = st.sidebar.number_input("Cost per hour (Maintain)", value=3.0)

# FORECASTING WITH PROPHET
def generate_forecast(df, column, periods):
    df_prophet = df[["timestamp", column]].rename(columns={"timestamp": "ds", column: "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)
    full_future = model.make_future_dataframe(periods=periods + 1000, freq="H")
    future = full_future[(full_future["ds"] >= forecast_start) & (full_future["ds"] <= forecast_end)]
    forecast = model.predict(future)
    return model, forecast

metric_labels = {
    "cpu_usage": "CPU Usage",
    "memory_usage": "Memory Usage",
    "disk_io": "Disk I/O",
    "network_usage_mbps": "Network Usage (Mbps)"
}
st.subheader(f" {metric_labels.get(metric, metric)} Forecast")

model, forecast = generate_forecast(df, metric, periods)
fig_forecast = px.line(
    forecast.tail(periods),
    x="ds",
    y="yhat",
    title=f"{metric.replace('_', ' ').title()} Forecast",
    labels={"ds": "Timestamp", "yhat": "Forecasted Value"},
)
fig_forecast.update_layout(
    xaxis=dict(rangeslider_visible=False),
    showlegend=False
)
st.plotly_chart(fig_forecast, use_container_width=True)

# --------------------------------------------
# PEAK USAGE HIGHLIGHT
# --------------------------------------------
st.markdown("### Peak Usage Periods")
forecast_tail = forecast[["ds", "yhat"]].tail(periods).rename(columns={"ds": "timestamp", "yhat": f"{metric}_forecast"})
top_peaks = forecast_tail.sort_values(by=f"{metric}_forecast", ascending=False).head(5)
st.dataframe(top_peaks.reset_index(drop=True), use_container_width=True)

# # --------------------------------------------
# # ANOMALY DETECTION (Z-score Method)
# # --------------------------------------------
# def detect_anomalies_zscore(series, threshold=3):
#     mean = series.mean()
#     std = series.std()
#     z_scores = (series - mean) / std
#     return z_scores.abs() > threshold

# # Apply anomaly detection
# forecast_tail["anomaly"] = detect_anomalies_zscore(forecast_tail[f"{metric}_forecast"])

# # DISPLAY ANOMALIES

# st.markdown("### Detected Anomalies")
# anomalies = forecast_tail[forecast_tail["anomaly"]]
# if not anomalies.empty:
#     st.dataframe(
#         anomalies[["timestamp", f"{metric}_forecast"]].rename(
#             columns={f"{metric}_forecast": "Forecasted Value"}
#         ).reset_index(drop=True),
#         use_container_width=True
#     )
# else:
#     st.info("No significant anomalies detected in the forecast.")


# SCALING ACTIONS
def get_scaling_action(val, metric):
    thresholds = {
        "cpu_usage": (75, 35),
        "memory_usage": (80, 50),
        "disk_io": (70, 30),
        "network_usage_mbps": (350, 250),  
    }
    upper, lower = thresholds.get(metric, (75, 35))  
    if val > upper:
        return "Scale UP"
    elif val < lower:
        return "Scale DOWN"
    else:
        return "Maintain"


forecast_tail["scaling_action"] = forecast_tail[f"{metric}_forecast"].apply(lambda x: get_scaling_action(x, metric))
st.markdown("### Scaling Recommendation")
scaling_counts = forecast_tail["scaling_action"].value_counts().to_frame("count").reset_index().rename(columns={"index": "scaling_action"})
st.dataframe(scaling_counts, use_container_width=True)

fig_scale = px.bar(scaling_counts, x="scaling_action", y="count", color="scaling_action",
                   color_discrete_map={"Scale UP": "red", "Scale DOWN": "green", "Maintain": "orange"})
st.plotly_chart(fig_scale, use_container_width=True)

# COST MODELING
st.markdown("### Estimated Cost Impact")

count_up = (forecast_tail["scaling_action"] == "Scale UP").sum()
count_down = (forecast_tail["scaling_action"] == "Scale DOWN").sum()
count_maintain = (forecast_tail["scaling_action"] == "Maintain").sum()

total_cost = (
    (count_up * cost_up) +
    (count_down * cost_down) +
    (count_maintain * cost_maintain)
)

cost_summary = {
    "Scale UP Hours": count_up,
    "Scale DOWN Hours": count_down,
    "Maintain Hours": count_maintain,
    "Total Hours": count_up + count_down + count_maintain,
    "Estimated Total Cost ($)": round(total_cost, 2)
}

st.dataframe(pd.DataFrame([cost_summary]), use_container_width=True)

# Visual cost breakdown
cost_data = pd.DataFrame({
    "Action": ["Scale UP", "Scale DOWN", "Maintain"],
    "Hours": [count_up, count_down, count_maintain],
    "Total Cost ($)": [
        round(count_up * cost_up, 2),
        round(count_down * cost_down, 2),
        round(count_maintain * cost_maintain, 2)
    ]
})

fig_cost = px.bar(cost_data, x="Action", y="Total Cost ($)", color="Action",
                  color_discrete_map={"Scale UP": "red", "Scale DOWN": "green", "Maintain": "orange"})
st.plotly_chart(fig_cost, use_container_width=True)

# FINAL SUMMARY RECOMMENDATIONS

summary_data = {
    "Metric": metric_labels[metric],
    "Forecast Period (hrs)": periods,
    "Peak Forecasted Value": round(forecast_tail[f"{metric}_forecast"].max(), 2),
    "Scale UP Hours": count_up,
    "Scale DOWN Hours": count_down,
    "Maintain Hours": count_maintain,
    "Total Estimated Cost ($)": round(total_cost, 2),
}

if count_up > 0.8 * periods:
    recommendation = "High resource usage expected. Consider scaling up capacity significantly."
elif count_down > 0.8 * periods:
    recommendation = "Low usage predicted. You may scale down resources to save cost."
else:
    recommendation = "Balanced usage. Maintain current resource levels with slight scaling."

summary_data["Recommendation"] = recommendation

final_recommendations_df = pd.DataFrame([summary_data])

# Show in app
st.markdown("###  Final Recommendation Summary")
st.dataframe(final_recommendations_df, use_container_width=True)

# DOWNLOAD FORECAST CSV
csv_forecast = forecast_tail.to_csv(index=False).encode("utf-8")
st.download_button(" Download Forecast CSV", csv_forecast, f"{metric}_forecast_raw.csv", "text/csv")

# MODEL ACCURACY (TRAIN / TEST EVALUATION)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df_prophet = df[["timestamp", metric]].rename(columns={"timestamp": "ds", metric: "y"})
split_index = int(len(df_prophet) * 0.8)
train_df = df_prophet.iloc[:split_index]
test_df = df_prophet.iloc[split_index:]

model_accuracy = Prophet(daily_seasonality=True, weekly_seasonality=True)
model_accuracy.fit(train_df)

future_accuracy = model_accuracy.make_future_dataframe(periods=len(test_df), freq='H')
forecast_accuracy = model_accuracy.predict(future_accuracy)
forecast_test = forecast_accuracy.tail(len(test_df)).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

mae = mean_absolute_error(test_df["y"], forecast_test["yhat"])
rmse = np.sqrt(mean_squared_error(test_df["y"], forecast_test["yhat"]))
r2 = r2_score(test_df["y"], forecast_test["yhat"])

st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.2f}")
st.write(f"**R² Score**: {r2:.2f}")