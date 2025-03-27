# feature engineering

import pandas as pd
import numpy as np
from config import TRAINING_DATASET, ENGINEERED_FEATURES

# Load dataset
df = pd.read_csv(TRAINING_DATASET, parse_dates=["date", "issuing_date"])

# Ensure dataset is sorted by customer_id and date
df = df.sort_values(by=["customer_id", "date"])

df["transaction_amount"] = df.groupby("customer_id")["transaction_amount"].transform(lambda x: x.fillna(x.mean()))
most_frequent_plan = df.groupby("customer_id")["plan_type"].apply(lambda x: x.mode()[0])
df["plan_type"] = df["plan_type"].fillna(df["customer_id"].map(most_frequent_plan))


# Customer-Based Features ---
# Calculate customer tenure in days
df["customer_tenure"] = (df["date"] - df["issuing_date"]).dt.days 

# Transactions statistics
df["total_transactions"] = df.groupby("customer_id")["transaction_amount"].transform("count")
df["avg_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("mean")
df["std_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("std").fillna(0)
df["max_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("max")
df["min_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("min")

# Time-Based Features ---
# Lagged transaction amounts for trend analysis
df["prev_transaction"] = df.groupby("customer_id")["transaction_amount"].shift(1)
df["transaction_trend"] = df["transaction_amount"] - df["prev_transaction"]

# Number of inactive months (months without transactions)
df["inactive_months"] = df.groupby("customer_id")["date"].diff().dt.days.fillna(0) / 30

# Date of first and last transactions
df["first_transaction"] = df.groupby("customer_id")["date"].transform("min")
df["last_transaction"] = df.groupby("customer_id")["date"].transform("max")

# Days since last transaction
df["days_since_last_txn"] = (pd.to_datetime("2024-03-01") - df["last_transaction"]).dt.days


# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

df["rolling_avg_3m"] = (
    df.groupby("customer_id")["transaction_amount"].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
)

df["rolling_avg_6m"] = (
    df.groupby("customer_id")["transaction_amount"].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
)

# Plan-Based Features ---
# First and last plan type
df["first_plan"] = df.groupby("customer_id")["plan_type"].transform("first")
df["last_plan"] = df.groupby("customer_id")["plan_type"].transform("last")

# Plan switching count

df["plan_switch_count"] = df.groupby("customer_id")["plan_type"].transform("nunique") - 1

print(df["plan_type"])
print(df["plan_switch_count"])

# Plan ratios
df["premium_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Premium").sum() / len(x))
df["standard_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Standard").sum() / len(x))
df["basic_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Basic").sum() / len(x))

# Upgrade/Downgrade Indicators
df["is_downgraded"] = df.groupby("customer_id")["plan_type"].apply(lambda x: ((x.shift(1) == "Premium") & (x != "Premium")).astype(int)).fillna(0)
df["is_upgraded"] = df.groupby("customer_id")["plan_type"].apply(lambda x: ((x.shift(1) != "Premium") & (x == "Premium")).astype(int)).fillna(0)


# Define the transaction amount levels and corresponding labels
bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, float("inf")]
labels = list(range(len(bins) - 1))  # Assign labels as 0,1,2,...

# Calculate cumulative sum of transactions for each customer
df["cumulative_transaction"] = df.groupby("customer_id")["transaction_amount"].cumsum()

# Assign transaction level based on cumulative sum
df["transaction_level"] = pd.cut(df["cumulative_transaction"], bins=bins, labels=labels, right=True)

# Convert the transaction level to integer type
df["transaction_level"] = df["transaction_level"].astype(int)

# Display the modified DataFrame
#print(df[["customer_id", "date", "transaction_amount", "cumulative_transaction", "transaction_level"]])

# Save the enhanced dataset
df.to_csv(ENGINEERED_FEATURES, index=False)
print("Feature engineering complete. Saved to 'insurance_data_enhanced.csv'.")

