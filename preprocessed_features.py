import pandas as pd
from config import ENGINEERED_FEATURES, PREPROCESSED_FEATURES

def loadData(file_path):
    """Load the engineered dataset."""
    return pd.read_csv(file_path)

def featureEngineering(df):
    """Perform Feature Engineering on the dataset."""
    df["transaction_gap"] = df["days_since_last_txn"] / df["customer_tenure"]
    df["spending_variability"] = df["std_transaction"] / df["avg_transaction"]
    df["loyalty_score"] = df["customer_tenure"] * df["cumulative_transaction"] / (df["plan_switch_count"] + 1)
    df["high_churn_risk"] = (df["inactive_months"] > 0.933333333333333).astype(int)  # Flag for customers inactive for >2 months
    df["transaction_trend_score"] = df["rolling_avg_3m"] - df["rolling_avg_6m"]  # Short vs Long term trend
    return df

def selectRelevantFeatures(df):
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
        "is_downgraded",
        "is_upgraded",
        "total_transactions",
        "premium_ratio",
        "standard_ratio",
        "basic_ratio",
        "churn"
    ]
    return df[selected_features]

def savePreprocessedData(df, output_path):
    """Save the preprocessed dataset."""
    df.to_csv(output_path, index=False)
    print("Feature extraction complete.")

def main():
    """Main function to run the feature extraction pipeline."""
    df = loadData(ENGINEERED_FEATURES)
    df = featureEngineering(df)
    df_selected = selectRelevantFeatures(df)
    savePreprocessedData(df_selected, PREPROCESSED_FEATURES)

if __name__ == "__main__":
    main()
