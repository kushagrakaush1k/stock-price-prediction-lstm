import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")

st.title("📈 Google Stock Price Predictor")
st.markdown("LSTM deep learning model trained on historical GOOG closing prices. Predicts future stock movement based on the last 60 days of data.")
st.markdown("---")

@st.cache_resource
def load_and_train():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    df = pd.read_csv('dataset/GOOG.csv')
    data = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    def create_sequences(data, window=60):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_split=0.1, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return df, data, scaled_data, scaler, model, predictions, y_actual, train_size

with st.spinner("Training LSTM model on first load — takes ~2 minutes ☕"):
    df, data, scaled_data, scaler, model, predictions, y_actual, train_size = load_and_train()

st.success("✅ Model trained and ready!")
st.markdown("---")

st.subheader("📊 Historical Closing Price")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Close'].values, linewidth=1.5, color='steelblue')
ax1.set_xlabel('Days')
ax1.set_ylabel('Close Price (USD)')
ax1.set_title('Google (GOOG) Closing Price History')
plt.tight_layout()
st.pyplot(fig1)

st.markdown("---")


st.subheader("Predicted vs Actual — Test Set")

rmse = math.sqrt(mean_squared_error(y_actual, predictions))

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(y_actual, label='Actual Price', linewidth=1.5, color='steelblue')
ax2.plot(predictions, label='Predicted Price',
         linewidth=1.5, linestyle='--', color='tomato')
ax2.set_xlabel('Days')
ax2.set_ylabel('Close Price (USD)')
ax2.set_title('LSTM — Predicted vs Actual')
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

col1, col2 = st.columns(2)
col1.metric("RMSE", f"${rmse:.2f}")
col2.metric("Test Samples", len(y_actual))

st.markdown("---")


st.subheader("Future Price Forecast")
st.markdown("Choose how many days ahead you want the model to predict:")

forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7, step=1)

if st.button("Generate Forecast", use_container_width=True):
    future_preds = []
    last_60 = scaled_data[-60:].reshape(1, 60, 1).copy()

    for _ in range(forecast_days):
        next_pred = model.predict(last_60, verbose=0)[0][0]
        future_preds.append(next_pred)
        last_60 = np.append(last_60[:, 1:, :], [[[next_pred]]], axis=1)

    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    )

    last_actual_price = data[-1][0]
    forecast_days_list = list(range(1, forecast_days + 1))

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(forecast_days_list, future_preds,
             marker='o', linewidth=2, color='mediumseagreen', label='Forecasted Price')
    ax3.axhline(y=last_actual_price, color='steelblue',
                linestyle='--', linewidth=1, label=f'Last Actual: ${last_actual_price:.2f}')
    ax3.set_xlabel('Days from Today')
    ax3.set_ylabel('Predicted Close Price (USD)')
    ax3.set_title(f'LSTM Forecast — Next {forecast_days} Days')
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)

    forecast_df = pd.DataFrame({
        'Day': [f'Day +{i}' for i in forecast_days_list],
        'Predicted Price (USD)': [f"${p[0]:.2f}" for p in future_preds],
        'Change from Last Close': [f"{'+' if p[0]-last_actual_price > 0 else ''}{p[0]-last_actual_price:.2f}" for p in future_preds]
    })
    st.dataframe(forecast_df, use_container_width=True)

    predicted_direction = "📈 UP" if future_preds[-1][0] > last_actual_price else "📉 DOWN"
    st.info(f"**Model predicts trend is going {predicted_direction}** over the next {forecast_days} days.")

st.markdown("---")
st.caption("⚠️ For educational purposes only. Not financial advice.")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** Stacked LSTM (2 layers, 50 units)  
    **Look-back:** 60 days  
    **Training split:** 80/20  
    **Optimizer:** Adam  
    **Loss:** Mean Squared Error  
    **Epochs:** 20
    """)
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    1. Model trained on 80% of GOOG history
    2. Uses last 60 days as input window
    3. Predicts next day, slides window forward
    4. Repeats for chosen forecast horizon
    """)
    st.markdown("---")
    st.markdown("Built by **Kushagra Shekhar Kaushik**")
    st.markdown("[GitHub](https://github.com/kushagrakaush1k/stock-price-prediction-lstm)")