import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# =============================
# LOAD DATA
# =============================

data = pd.read_csv("guntur_clean.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

prices = data["Modal_Price"].values.reshape(-1, 1)

print("Total usable rows:", len(prices))

# =============================
# SCALE
# =============================

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

# =============================
# SEQUENCES
# =============================

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 30
X, y = create_sequences(scaled, sequence_length)

# Split
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =============================
# MODEL
# =============================

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# =============================
# TRAIN
# =============================

model.fit(X_train, y_train, epochs=60, batch_size=16)

# =============================
# PREDICT
# =============================

predictions = model.predict(X_test)

predicted = scaler.inverse_transform(predictions)
real = scaler.inverse_transform(y_test)

# =============================
# PLOT
# =============================

plt.figure(figsize=(12,6))
plt.plot(real.flatten(), label="Real Price")
plt.plot(predicted.flatten(), label="Predicted Price")
plt.title("Improved Guntur Dry Chillies Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

model.save("guntur_lstm_model.h5")

print("✅ Improved model trained successfully")