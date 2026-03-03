from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = FastAPI()

# Load data
data = pd.read_csv("guntur_clean.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

prices = data["Modal_Price"].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

model = load_model("guntur_lstm_model.h5", compile=False)

sequence_length = 30

@app.get("/predict")
def predict():

    last_sequence = scaled_prices[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, 1)

    future_predictions = []

    for _ in range(7):
        next_pred = model.predict(last_sequence, verbose=0)
        future_predictions.append(next_pred[0][0])

        next_pred_reshaped = next_pred.reshape(1,1,1)
        last_sequence = np.concatenate(
            (last_sequence[:,1:,:], next_pred_reshaped),
            axis=1
        )

    future_predictions = np.array(future_predictions).reshape(-1,1)
    future_prices = scaler.inverse_transform(future_predictions)

    return {
        "tomorrow_price": round(float(future_prices[0][0]),2),
        "seven_day_forecast": [round(float(x[0]),2) for x in future_prices]
    }