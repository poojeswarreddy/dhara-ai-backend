import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =============================
# LOAD DATA
# =============================

data = pd.read_csv("guntur_clean.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

prices = data["Modal_Price"].values.reshape(-1, 1)

# =============================
# SCALE
# =============================

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# =============================
# LOAD MODEL (IMPORTANT FIX)
# =============================

model = load_model("guntur_lstm_model.h5", compile=False)

sequence_length = 30

# Get last 30 days
last_sequence = scaled_prices[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, 1)

future_predictions = []

# =============================
# RECURSIVE FORECAST
# =============================

for _ in range(7):

    next_pred = model.predict(last_sequence, verbose=0)

    # Save prediction
    future_predictions.append(next_pred[0][0])

    # Reshape correctly to (1,1,1)
    next_pred_reshaped = next_pred.reshape(1,1,1)

    # Remove first day and append new prediction
    last_sequence = np.concatenate(
        (last_sequence[:,1:,:], next_pred_reshaped),
        axis=1
    )

# =============================
# INVERSE SCALE
# =============================

future_predictions = np.array(future_predictions).reshape(-1,1)
future_prices = scaler.inverse_transform(future_predictions)

# =============================
# PRINT RESULTS
# =============================

print("\n📅 Next 7 Days Predicted Prices:\n")

for i, price in enumerate(future_prices):
    print(f"Day {i+1}: ₹ {round(price[0],2)}")

# =============================
# PLOT
# =============================

plt.figure(figsize=(10,5))

plt.plot(prices[-60:], label="Last 60 Days Real Price")

future_x = np.arange(len(prices[-60:]), len(prices[-60:]) + 7)
plt.plot(future_x, future_prices.flatten(), marker='o', label="7-Day Forecast")

plt.title("Guntur Dry Chillies 7-Day Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()