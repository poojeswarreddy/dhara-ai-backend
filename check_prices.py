import pandas as pd

data = pd.read_csv("guntur_clean.csv")

print(data.head())
print("\nColumn types:")
print(data.dtypes)

print("\nSample Modal_Price values:")
print(data["Modal_Price"].head(10))