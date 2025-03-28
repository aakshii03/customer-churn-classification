import pandas as pd
import numpy as np
import joblib
import pickle
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef


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

def evaluate_model(model, X_test, y_test, results_dir):
    """Evaluate the trained XGBoost model."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report["matthews_corrcoef"] = mcc  

    print("\nXGBoost Model Evaluation:")
    print("Accuracy:", accuracy)
    print("MCC:", mcc)
    print(classification_report(y_test, y_pred))

    model_filename = "xgboost"
    
    report_path = os.path.join(results_dir, f"{model_filename}_metrics.json")
    with open(report_path, "w") as json_file:
        json.dump(report, json_file, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fig_path = os.path.join(results_dir, f"{model_filename}_confusion_matrix.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

def save_xgboost_model(model, models_dir):
    """Save the XGBoost model in JSON format."""
    model.save_model(os.path.join(models_dir, "xgboost_model.json"))

def save_label_encoders(label_encoders, models_dir):
    """Save label encoders for future use."""
    with open(os.path.join(models_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)

def generate_lime_explanation(model, X_train, X_test, feature_names, results_dir):
    """Generate LIME explanations for the XGBoost model with improved layout."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["No Churn", "Churn"], discretize_continuous=True
    )

    instance_idx = 5  
    instance = X_test[instance_idx].reshape(1, -1)

    exp = explainer.explain_instance(instance.flatten(), model.predict_proba)

    # Increase figure size and adjust layout to avoid cutting off feature names
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 8)  # Set larger figure size
    plt.tight_layout()  # Ensure everything fits within the frame
    plt.savefig(os.path.join(results_dir, "lime_explanation.png"), bbox_inches="tight", dpi=300)
    plt.show()

def train_pipeline(models_dir, preprocessed_features, results_dir):
    """Main function to execute the ML pipeline."""
    df = load_data(preprocessed_features)
    df, label_encoders = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = split_and_scale_data(df, models_dir)

    print("Training XGBoost model...")
    model = train_xgboost(X_train_scaled, y_train)

    print("Evaluating XGBoost model...")
    evaluate_model(model, X_test_scaled, y_test, results_dir)

    print("Saving XGBoost model and label encoders...")
    save_xgboost_model(model, models_dir)
    save_label_encoders(label_encoders, models_dir)

    print("Generating LIME explanation...")
    generate_lime_explanation(model, X_train_scaled, X_test_scaled, feature_names, results_dir)

    print("All tasks completed successfully!")