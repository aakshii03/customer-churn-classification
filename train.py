import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
import lime
import lime.lime_tabular
import seaborn as sns
from config import MODELS, PREPROCESSED_FEATURES, RESULTS
import matplotlib.pyplot as plt
import os
import json

# Load Data
df = pd.read_csv(PREPROCESSED_FEATURES)

# Drop Irrelevant Columns
df.drop(columns=["customer_id", "date", "issuing_date", "first_transaction", "last_transaction", "first_plan", "last_plan"], inplace=True, errors='ignore')

#Encode Categorical Columns
categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle Missing & Infinite Values
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert inf to NaN
imputer = SimpleImputer(strategy="mean")  # Impute NaN values with mean
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


X = df.drop(columns=["churn"])
y = df["churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.shape)

joblib.dump(scaler, MODELS+"/scaler.pkl")

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"\nModel: {name}")
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)  # Additional Metric (MCC)
    report = classification_report(y_test, y_pred, output_dict=True)  # Convert to dictionary

    print("Accuracy:", accuracy)
    print("Matthews Correlation Coefficient (MCC):", mcc)
    print(classification_report(y_test, y_pred))

    # Save the model
    model_filename = name.replace(" ", "_").lower()
    model_path = f"models/{model_filename}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    # Save classification report & MCC as JSON
    report["matthews_corrcoef"] = mcc  # Add MCC to JSON
    report_path = os.path.join(RESULTS, f"{model_filename}_metrics.json")
    with open(report_path, "w") as json_file:
        json.dump(report, json_file, indent=4)
    print(f"Classification metrics saved: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the figure
    fig_path = os.path.join(RESULTS, f"{model_filename}_confusion_matrix.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"Confusion matrix saved: {fig_path}")

    plt.close()

# Save XGBoost Model in JSON
xgb_model = models["XGBoost"]
xgb_model.save_model(MODELS+"/xgboost_model.json")
# xgb_model.save_model("models/xgboost_model.h5")

# Save Label Encoders (for categorical variables)
with open(MODELS+"/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("All models & encoders saved successfully.")

# LIME Explanation
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled, feature_names=X.columns, class_names=["No Churn", "Churn"], discretize_continuous=True
)

instance_idx = 5  # Choose a test instance
instance = X_test_scaled[instance_idx].reshape(1, -1)

best_model = models["XGBoost"]

exp = explainer.explain_instance(instance.flatten(), best_model.predict_proba)
exp.as_pyplot_figure()
plt.savefig(RESULTS+"/lime_explanation.png")
plt.show()
