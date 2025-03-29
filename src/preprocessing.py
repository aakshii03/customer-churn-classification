import pandas as pd
import numpy as np

"""
Feature Engineering for Customer Transaction Data
This script performs feature engineering on customer transaction data to support churn prediction and behavioral analysis.

Key functionalities:
1. Data Loading & Sorting:
   - Loads data from a CSV file.
   - Sorts data by customer ID and date for sequential processing.
2. Missing Value Handling:
   - Fills missing transaction amounts using the mean transaction value per customer.
   - Replaces missing plan types with the most frequent plan per customer.
3. Transaction Statistics:
   - Computes tenure, transaction count, mean, standard deviation, and min/max transaction amounts.
4. Time-Based Features:
   - Computes lagged transaction amounts, transaction trends, inactive months, and days since the last transaction.
5. Rolling Features:
   - Computes rolling averages for transaction amounts over 3-month and 6-month periods.
6. Plan-Based Features:
   - Identifies first and last plan types, counts plan switches, and calculates upgrade/downgrade indicators.
7. Transaction Level Assignment:
   - Categorizes cumulative transaction amounts into predefined levels.
8. Data Saving:
   - Saves the engineered dataset to a CSV file.
Usage:
Call `data_preperocessing(training_dataset, engineered_features)` to process a dataset and save the output.
"""

def load_and_sort_data(filepath: str) -> pd.DataFrame:
    """Load dataset and sort by customer_id and date."""
    df = pd.read_csv(filepath, parse_dates=["date", "issuing_date"])
    return df.sort_values(by=["customer_id", "date"])


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for transaction amount and plan type."""
    df["transaction_amount"] = df.groupby("customer_id")["transaction_amount"].transform(lambda x: x.fillna(x.mean()))
    
    most_frequent_plan = df.groupby("customer_id")["plan_type"].apply(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    df["plan_type"] = df["plan_type"].fillna(df["customer_id"].map(most_frequent_plan))
    
    return df


def transactions_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate customer-level features."""
    df["customer_tenure_days"] = (df["date"] - df["issuing_date"]).dt.days  
    df["total_transactions_count"] = df.groupby("customer_id")["transaction_amount"].transform("count")
    df["avg_transaction_amount"] = df.groupby("customer_id")["transaction_amount"].transform("mean")
    df["std_transaction_amount"] = df.groupby("customer_id")["transaction_amount"].transform("std").fillna(0)
    df["max_transaction_amount"] = df.groupby("customer_id")["transaction_amount"].transform("max")
    df["min_transaction_amount"] = df.groupby("customer_id")["transaction_amount"].transform("min")
    
    return df


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate time-based features such as lagged transaction amounts, inactive months, first/last transactions, and days since last transaction."""
    df["prev_transaction"] = df.groupby("customer_id")["transaction_amount"].shift(1)
    df["transaction_trend"] = df["transaction_amount"] - df["prev_transaction"]
    df["inactive_months"] = df.groupby("customer_id")["date"].diff().dt.days.fillna(0) / 30
    df["first_transaction_amount"] = df.groupby("customer_id")["date"].transform("min")
    df["last_transaction_amount"] = df.groupby("customer_id")["date"].transform("max")
    df["days_since_last_txn"] = (pd.to_datetime("2024-03-01") - df["last_transaction_amount"]).dt.days  

    return df


def generate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rolling averages for transaction amounts."""
    df["rolling_avg_3m"] = (
        df.groupby("customer_id")["transaction_amount"].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    )
    df["rolling_avg_6m"] = (
        df.groupby("customer_id")["transaction_amount"].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
    )
    
    return df


def generate_plan_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate plan-based features such as first/last plan, plan switches, and upgrade/downgrade indicators."""
    df["first_plan"] = df.groupby("customer_id")["plan_type"].transform("first")
    df["last_plan"] = df.groupby("customer_id")["plan_type"].transform("last")
    df["plan_switch_count"] = df.groupby("customer_id")["plan_type"].transform("nunique") - 1
    
    df["premium_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Premium").sum() / len(x))
    df["standard_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Standard").sum() / len(x))
    df["basic_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Basic").sum() / len(x))

    df["is_plan_downgraded"] = df.groupby("customer_id")["plan_type"].apply(lambda x: ((x.shift(1) == "Premium") & (x != "Premium")).astype(int)).fillna(0)
    df["is_plan_upgraded"] = df.groupby("customer_id")["plan_type"].apply(lambda x: ((x.shift(1) != "Premium") & (x == "Premium")).astype(int)).fillna(0)

    return df


def generate_transaction_level(df: pd.DataFrame) -> pd.DataFrame:
    """Assign transaction levels based on cumulative sum."""
    bins = list(range(0, 3300, 100)) + [float("inf")]  
    labels = list(range(len(bins) - 1))  
    
    df["cumulative_transaction_amount"] = df.groupby("customer_id")["transaction_amount"].cumsum()
    df["transaction_amount_level"] = pd.cut(df["cumulative_transaction_amount"], bins=bins, labels=labels, right=True).astype(int)
    
    return df


def save_dataset(df: pd.DataFrame, filepath: str):
    """Save the processed dataset."""
    df.to_csv(filepath, index=False)
    print(f"Feature engineering complete. Saved to {filepath}")


def data_preperocessing(training_dataset: str, engineered_features: str):
    df = load_and_sort_data(training_dataset)
    df = fill_missing_values(df)
    df = transactions_statistics(df)
    df = generate_time_features(df)
    df = generate_rolling_features(df)
    df = generate_plan_features(df)
    df = generate_transaction_level(df)

    save_dataset(df, engineered_features)
