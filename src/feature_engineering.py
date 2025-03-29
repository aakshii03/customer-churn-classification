import pandas as pd

"""
Feature Engineering and Feature Selection for Churn Prediction

This script further processes an engineered dataset by creating additional derived features and selecting relevant columns for model training.

Key functionalities:
1. Data Loading:
   - Loads the engineered dataset from a CSV file.
2. Feature Engineering:
   - Computes `transaction_gap`: Ratio of days since last transaction to customer tenure.
   - Computes `spending_variability`: Standard deviation of transactions divided by the mean transaction amount.
   - Computes `loyalty_score`: A heuristic score based on customer tenure, cumulative transactions, and plan switches.
   - Flags `high_churn_risk`: Identifies customers inactive for more than 2 months.
   - Computes `transaction_trend_score`: Difference between 3-month and 6-month rolling transaction averages to identify trends.
3. Feature Selection:
   - Retains only relevant features for churn modeling.
4. Data Saving:
   - Saves the processed dataset to a CSV file.
Usage:
Call `feature_engineering(engineered_features, preprocessed_features)` to process a dataset and save the output.
"""


def load_data(file_path):
    """Load the engineered dataset."""
    return pd.read_csv(file_path)


def engineer_features(df):
    """Perform feature engineering on the dataset."""
    df["transaction_gap"] = df["days_since_last_txn"] / df["customer_tenure_days"]
    df["spending_variability"] = df["std_transaction_amount"] / df["avg_transaction_amount"]
    df["loyalty_score"] = df["customer_tenure_days"] * df["cumulative_transaction_amount"] / (df["plan_switch_count"] + 1)
    df["high_churn_risk"] = (df["inactive_months"] > 0.933333333333333).astype(int)  # Flag for customers inactive for >2 months
    df["transaction_trend_score"] = df["rolling_avg_3m"] - df["rolling_avg_6m"]  # Short vs Long term trend
    return df


def select_relevant_features(df):
    """Retain only relevant columns."""
    selected_features = [
        "customer_id",
        "transaction_amount",
        "transaction_gap",
        "spending_variability",
        "loyalty_score",
        "high_churn_risk",
        "transaction_trend_score",
        "plan_type",
        "is_plan_downgraded",
        "is_plan_upgraded",
        "total_transactions_count",
        "premium_ratio",
        "standard_ratio",
        "basic_ratio",
        "churn"
    ]
    return df[selected_features]


def save_preprocessed_data(df, output_path):
    """Save the preprocessed dataset."""
    df.to_csv(output_path, index=False)
    print("Feature extraction complete.")


def feature_engineering(preprocessed_features, engineered_features):
    """Main function to run the feature extraction pipeline."""
    df = load_data(preprocessed_features)
    df = engineer_features(df)
    df_selected = select_relevant_features(df)
    save_preprocessed_data(df_selected, engineered_features)
