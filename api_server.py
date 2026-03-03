from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = FastAPI()

# Load trained model
model = load_model("guntur_lstm_model.h5", compile=False)

# Load saved scaler
scaler = joblib.load("scaler.save")

sequence_length = 30

@app.get("/predict")
def predict():

    # Dummy base sequence (last known average price scaled)
    # Later we can improve this with DB or storage
    base_value = 0.5  # mid scaled value
    last_sequence = np.array([[base_value]] * sequence_length)
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
        "tomorrow_price": round(float(future_prices[0][0]), 2),
        "seven_day_forecast": [
            round(float(x[0]), 2) for x in future_prices
        ]
    }