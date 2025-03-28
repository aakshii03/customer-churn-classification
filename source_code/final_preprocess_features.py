import pandas as pd


def load_data(file_path):
    """Load the engineered dataset."""
    return pd.read_csv(file_path)


def feature_engineering(df):
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


def preprocess_features(engineered_features, preprocessed_features):
    """Main function to run the feature extraction pipeline."""
    df = load_data(engineered_features)
    df = feature_engineering(df)
    df_selected = select_relevant_features(df)
    save_preprocessed_data(df_selected, preprocessed_features)
