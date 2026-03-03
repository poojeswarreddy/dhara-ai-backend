import pandas as pd

# Load all 3 year files
data_2022 = pd.read_csv("Data/2022.csv")
data_2023 = pd.read_csv("Data/2023.csv")
data_2024 = pd.read_csv("Data/2024.csv")

# Combine
data = pd.concat([data_2022, data_2023, data_2024], ignore_index=True)

print("Total rows before filtering:", len(data))

# ==========================
# FILTER CONDITIONS
# ==========================

filtered = data[
    (data["State"] == "Andhra Pradesh") &
    (data["Market Name"].str.contains("Guntur", case=False, na=False)) &
    (data["Commodity Name"].str.contains("Dry Chillies", case=False, na=False))
].copy()

print("Rows after filtering:", len(filtered))

# ==========================
# SELECT REQUIRED COLUMNS
# ==========================

filtered = filtered[[
    "Calendar Day",
    "Modal Price For The Commodity (UOM:INR(IndianRupees)), Scaling Factor:1"
]]

# Rename columns
filtered.columns = ["Date", "Modal_Price"]

# ==========================
# CLEAN DATE
# ==========================

filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
filtered = filtered.dropna(subset=["Date"])

# ==========================
# CLEAN PRICE
# ==========================

filtered["Modal_Price"] = pd.to_numeric(filtered["Modal_Price"], errors="coerce")
filtered = filtered.dropna(subset=["Modal_Price"])

# ==========================
# REMOVE OUTLIERS
# ==========================

filtered = filtered[
    (filtered["Modal_Price"] > 10000) &
    (filtered["Modal_Price"] < 30000)
]

# ==========================
# GROUP BY DATE (IMPORTANT)
# ==========================

filtered = filtered.groupby("Date")["Modal_Price"].mean().reset_index()

# Sort
filtered = filtered.sort_values("Date")

print("Final usable rows:", len(filtered))

# Save clean dataset
filtered.to_csv("guntur_clean.csv", index=False)

print("✅ Clean dataset created successfully")