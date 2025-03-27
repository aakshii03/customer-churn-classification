import pandas as pd
import numpy as np
from config import TRAINING_DATASET, ENGINEERED_FEATURES

def loadAndSortData(filepath: str) -> pd.DataFrame:
    """Load dataset and sort by customer_id and date."""
    df = pd.read_csv(filepath, parse_dates=["date", "issuing_date"])
    return df.sort_values(by=["customer_id", "date"])


def fillMissingValues(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for transaction amount and plan type."""
    df["transaction_amount"] = df.groupby("customer_id")["transaction_amount"].transform(lambda x: x.fillna(x.mean()))
    
    most_frequent_plan = df.groupby("customer_id")["plan_type"].apply(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    df["plan_type"] = df["plan_type"].fillna(df["customer_id"].map(most_frequent_plan))
    
    return df

#Transactions statistics
def transactionsStatistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate customer-level features."""
    df["customer_tenure"] = (df["date"] - df["issuing_date"]).dt.days  
    df["total_transactions"] = df.groupby("customer_id")["transaction_amount"].transform("count")
    df["avg_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("mean")
    df["std_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("std").fillna(0)
    df["max_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("max")
    df["min_transaction"] = df.groupby("customer_id")["transaction_amount"].transform("min")
    
    return df


def generateTimeFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """Generate time-based features such as
    Lagged transaction amounts for trend analysis
    Number of inactive months (months without transactions)
    Date of first and last transactions
    Days since last transaction"""

    df["prev_transaction"] = df.groupby("customer_id")["transaction_amount"].shift(1)
    df["transaction_trend"] = df["transaction_amount"] - df["prev_transaction"]
    df["inactive_months"] = df.groupby("customer_id")["date"].diff().dt.days.fillna(0) / 30
    df["first_transaction"] = df.groupby("customer_id")["date"].transform("min")
    df["last_transaction"] = df.groupby("customer_id")["date"].transform("max")
    df["days_since_last_txn"] = (pd.to_datetime("2024-03-01") - df["last_transaction"]).dt.days  

    return df


def generateRollingFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rolling averages for transaction amounts."""
    df["rolling_avg_3m"] = (
        df.groupby("customer_id")["transaction_amount"].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    )
    df["rolling_avg_6m"] = (
        df.groupby("customer_id")["transaction_amount"].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
    )
    
    return df


def generatePlanFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """Generate plan-based features such as first, last plan, plan switches and Upgrade/Downgrade Indicators."""
    df["first_plan"] = df.groupby("customer_id")["plan_type"].transform("first")
    df["last_plan"] = df.groupby("customer_id")["plan_type"].transform("last")
    df["plan_switch_count"] = df.groupby("customer_id")["plan_type"].transform("nunique") - 1
    
    df["premium_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Premium").sum() / len(x))
    df["standard_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Standard").sum() / len(x))
    df["basic_ratio"] = df.groupby("customer_id")["plan_type"].transform(lambda x: (x == "Basic").sum() / len(x))

    df["is_downgraded"] = df.groupby("customer_id")["plan_type"].apply(lambda x: ((x.shift(1) == "Premium") & (x != "Premium")).astype(int)).fillna(0)
    df["is_upgraded"] = df.groupby("customer_id")["plan_type"].apply(lambda x: ((x.shift(1) != "Premium") & (x == "Premium")).astype(int)).fillna(0)

    return df


def generateTransactionLevel(df: pd.DataFrame) -> pd.DataFrame:
    """Assign transaction levels:
    Define the transaction amount levels and corresponding labels
    Calculate cumulative sum of transactions for each customer
    Assign transaction level based on cumulative sum
    Convert the transaction level to integer type"""
    bins = list(range(0, 3300, 100)) + [float("inf")]  
    labels = list(range(len(bins) - 1))  
    
    df["cumulative_transaction"] = df.groupby("customer_id")["transaction_amount"].cumsum()
    df["transaction_level"] = pd.cut(df["cumulative_transaction"], bins=bins, labels=labels, right=True).astype(int)
    
    return df


def saveDataset(df: pd.DataFrame, filepath: str):
    """Save the processed dataset."""
    df.to_csv(filepath, index=False)
    print(f"Feature engineering complete. Saved to {filepath}")


def main():
    df = loadAndSortData(TRAINING_DATASET)
    df = fillMissingValues(df)
    df = transactionsStatistics(df)
    df = generateTimeFeatures(df)
    df = generateRollingFeatures(df)
    df = generatePlanFeatures(df)
    df = generateTransactionLevel(df)
    
    saveDataset(df, ENGINEERED_FEATURES)


if __name__ == "__main__":
    main()
