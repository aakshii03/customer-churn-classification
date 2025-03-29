import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

"""
Machine Learning Pipeline for Churn Prediction
This script implements a complete machine learning pipeline to train an XGBoost model for predicting customer churn.

Key functionalities:
1. Data Loading:
   - Loads the preprocessed dataset from a CSV file.
2. Data Preprocessing:
   - Drops irrelevant columns (e.g., customer ID, dates, first/last transactions, first/last plans).
   - Encodes categorical features using `LabelEncoder`.
   - Handles missing values by imputing with the mean.
3. Data Splitting and Scaling:
   - Splits data into training and test sets (80/20 split).
   - Standardizes numerical features using `StandardScaler`.
   - Saves the scaler for future use.
4. Model Training:
   - Trains an `XGBClassifier` on the processed data.
   - Uses `logloss` as the evaluation metric.
5. Model Saving:
   - Saves the trained XGBoost model in JSON format.
   - Saves label encoders for categorical features.
6. Execution Pipeline:
   - The train_pipeline function integrates all the steps and saves the required components for future inference.

Usage:
Call `train_pipeline(models_dir, preprocessed_features, results_dir)` to execute the full pipeline.
"""

def load_data(file_path):
    """Load the preprocessed dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Handle missing values, drop irrelevant columns, and encode categorical features."""
    drop_columns = ["customer_id", "date", "issuing_date", "first_transaction", "last_transaction", "first_plan", "last_plan"]
    df.drop(columns=drop_columns, inplace=True, errors="ignore")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df, label_encoders

def split_and_scale_data(df, models_dir):
    """Split data into training and testing sets, then scale it."""
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

def save_xgboost_model(model, models_dir):
    """Save the XGBoost model in JSON format."""
    model.save_model(os.path.join(models_dir, "xgboost_model.json"))

def save_label_encoders(label_encoders, models_dir):
    """Save label encoders for future use."""
    with open(os.path.join(models_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)


def train_pipeline(models_dir, preprocessed_features, results_dir):
    """Main function to execute the ML pipeline."""
    df = load_data(preprocessed_features)
    df, label_encoders = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = split_and_scale_data(df, models_dir)

    model = train_xgboost(X_train_scaled, y_train)

    print("Saving XGBoost model and label encoders...")
    save_xgboost_model(model, models_dir)
    save_label_encoders(label_encoders, models_dir)

    return model, X_test_scaled, y_test, results_dir, X_train_scaled, feature_names