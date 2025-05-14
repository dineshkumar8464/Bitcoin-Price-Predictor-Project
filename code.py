# Final Year ML Project: Enhanced Bitcoin Price Prediction using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import io
import time

# Page Config
st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'> Bitcoin Price Prediction using ML</h1>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("C:\\Users\\dines\\Downloads\\coin_Bitcoin.csv")
df.drop(columns=["SNo", "Name", "Symbol", "Volume"], errors="ignore", inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

# Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day_of_Week"] = df["Date"].dt.dayofweek
df["Price_Diff"] = df["Close"] - df["Open"]
df["High_Low_Diff"] = df["High"] - df["Low"]
df["Avg_Price"] = (df["High"] + df["Low"]) / 2
df["Close_Change_Pct"] = df["Close"].pct_change() * 100
df.dropna(inplace=True)

# Normalize
scaler = MinMaxScaler()
numeric_features = ["Open", "High", "Low", "Marketcap", "Price_Diff", "High_Low_Diff", "Avg_Price", "Year", "Month", "Day_of_Week"]
df[numeric_features] = scaler.fit_transform(df[numeric_features])
X = df[numeric_features]
y = df["Close"]

# Feature Selection using RF
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
top_features = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).index[:7]
X = X[top_features]

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_aug = X_train + np.random.normal(0, 0.01, X_train.shape)

# Train Models
start_time = time.time()
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions={
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
        "bootstrap": [True]
    },
    cv=3,
    n_iter=5,
    scoring="neg_mean_absolute_error"
)
rf_search.fit(X_train_aug, y_train)
best_rf_model = rf_search.best_estimator_

xgb_model = XGBRegressor(n_estimators=100, max_depth=6, alpha=0.01)
xgb_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Sidebar Inputs
st.sidebar.title("Input Parameters")

with st.sidebar.expander("Sample Market Data"):
    st.dataframe(df[["Date", "Open", "High", "Low", "Marketcap"]].tail(5))

with st.sidebar.expander("Input Tips"):
    st.markdown("""
    - **Prices** are in USD  
      • Example: `Open = 27000.00`, `High = 27500.00`, `Low = 26500.00`
    - **Marketcap** should be in full numeric form  
      • Example: `800000000000` = $800 Billion  
    - Use realistic values from recent trends
    """)

use_slider = st.sidebar.toggle("Use sliders instead of manual input", value=False)
date = st.sidebar.date_input("Select Date", datetime.date.today())

if use_slider:
    open_price = st.sidebar.slider("Open Price", 1000.0, 70000.0, 27000.0, help="Bitcoin opening price on selected date")
    high_price = st.sidebar.slider("High Price", 1000.0, 80000.0, 27500.0, help="Highest BTC price for the day")
    low_price = st.sidebar.slider("Low Price", 1000.0, 70000.0, 26500.0, help="Lowest BTC price for the day")
    marketcap = st.sidebar.slider("Marketcap ($)", 1e9, 1.5e12, 8e11, help="Total Bitcoin market capitalization")
else:
    open_price = st.sidebar.number_input("Open Price", min_value=0.0, format="%.2f", help="Bitcoin opening price on selected date")
    high_price = st.sidebar.number_input("High Price", min_value=0.0, format="%.2f", help="Highest BTC price for the day")
    low_price = st.sidebar.number_input("Low Price", min_value=0.0, format="%.2f", help="Lowest BTC price for the day")
    marketcap = st.sidebar.number_input("Marketcap", min_value=0.0, format="%.2f", help="Total Bitcoin market capitalization")

# Predict Function
def predict_price():
    if low_price > high_price:
        st.error("Low price cannot be greater than High price.")
        return None, None

    year = date.year
    month = date.month
    day_of_week = date.weekday()
    price_diff = open_price - low_price
    high_low_diff = high_price - low_price
    avg_price = (high_price + low_price) / 2

    input_data = pd.DataFrame([[open_price, high_price, low_price, marketcap,
                                 price_diff, high_low_diff, avg_price,
                                 year, month, day_of_week]],
                               columns=["Open", "High", "Low", "Marketcap", "Price_Diff",
                                        "High_Low_Diff", "Avg_Price", "Year", "Month", "Day_of_Week"])

    input_scaled = scaler.transform(input_data[numeric_features])
    input_scaled_top = pd.DataFrame(input_scaled, columns=numeric_features)[top_features]

    pred_rf = best_rf_model.predict(input_scaled_top)[0]
    pred_xgb = xgb_model.predict(input_scaled_top)[0]
    return pred_rf, pred_xgb

# Predict
if st.sidebar.button("Predict Price"):
    pred_rf, pred_xgb = predict_price()

    if pred_rf is not None:
        rf_price = pred_rf * 50000
        xgb_price = pred_xgb * 50000

        st.success(f"RF Prediction: ${rf_price:.2f}  |  ⚡ XGBoost: ${xgb_price:.2f}")
        direction = "Increase" if rf_price > open_price else "Decrease"
        st.info(f"Prediction Direction from Open Price: {direction}")

        # Explanation Block
        st.markdown("###What does this mean?")
        st.markdown(f"""
        - **RF Model Prediction**: Estimated closing price of **${rf_price:,.2f}**
        - **XGBoost Prediction**: Estimated closing price of **${xgb_price:,.2f}**
        - **Trend Direction**: Prediction indicates a possible **{direction.lower()}** compared to opening price.
        - Use these insights for market **analysis**, not financial advice.
        """)

        # Confidence Range (based on RMSE)
        rmse_rf = np.sqrt(mean_squared_error(y_test, best_rf_model.predict(X_test)))
        error_margin = rmse_rf * 50000
        st.markdown(f"**Confidence Range (RF): ${rf_price - error_margin:.2f} - ${rf_price + error_margin:.2f}**")

        # Summary Table
        st.markdown("### Prediction Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Date", "Open", "High", "Low", "Marketcap"],
            "Value": [
                str(date),
                f"${open_price:,.2f}",
                f"${high_price:,.2f}",
                f"${low_price:,.2f}",
                f"${marketcap:,.0f}"
            ]
        })
        st.table(summary_df)

        # Feature Importance - Horizontal Bar Chart
        st.markdown("### Feature Importance (Random Forest)")
        fig_imp, ax_imp = plt.subplots()
        importance_series = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values()
        importance_series.plot(kind='barh', ax=ax_imp, color='skyblue', edgecolor='black')
        ax_imp.set_title("Top Feature Contributions to Prediction")
        st.pyplot(fig_imp)

        # Metrics
        st.markdown("###  Model Performance Metrics")
        y_pred_rf = best_rf_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)
        metrics = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost"],
            "MAE": [mean_absolute_error(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_xgb)],
            "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_rf)), np.sqrt(mean_squared_error(y_test, y_pred_xgb))],
            "R2": [r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_xgb)]
        })
        st.dataframe(metrics, use_container_width=True)

        # Export CSV
        if st.button(" Download Prediction"):
            buffer = io.StringIO()
            pd.DataFrame({
                "Date": [date],
                "RandomForest_Predicted_Close": [rf_price],
                "XGBoost_Predicted_Close": [xgb_price]
            }).to_csv(buffer, index=False)
            st.download_button("Download as CSV", buffer.getvalue(), file_name="bitcoin_prediction.csv", mime="text/csv")

        # Actual vs Predicted Plot
        st.markdown("### Actual vs Predicted (Test Set)")
        fig2, ax2 = plt.subplots()
        ax2.plot(y_test.values[:50], label="Actual", marker='o')
        ax2.plot(y_pred_rf[:50], label="RF Predicted", linestyle="--")
        ax2.plot(y_pred_xgb[:50], label="XGB Predicted", linestyle=":")
        ax2.set_title("Sample Actual vs Predicted Close Price")
        ax2.legend()
        st.pyplot(fig2)

        # Training Time
        st.caption(f" Model Training Time: {training_time:.2f} seconds")
