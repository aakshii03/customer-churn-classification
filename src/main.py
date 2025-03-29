import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import MODELS
import numpy as np
import joblib
import xgboost as xgb
import os
from config import MODELS, PREPROCESSED_DATA, RESULTS, TRAINING_DATASET, ENGINEERED_FEATURE
from train import train_pipeline
import lime.lime_tabular
import matplotlib.pyplot as plt
from feature_engineering import feature_engineering
from preprocessing import data_preperocessing
from evaluate import evaluate_model
from explain import generate_lime_explanation

"""
This script executes the full machine learning pipeline for customer churn prediction.
It involves data preprocessing, feature engineering, model training, evaluation, and model explanation using LIME.

Steps:
1. Data Preprocessing: Cleans and prepares the training dataset.
2. Feature Engineering: Creates relevant features to improve model performance.
3. Model Training: Trains an XGBoost model on the processed dataset.
4. Model Evaluation: Assesses model performance using test data.
5. Model Explanation: Uses LIME to explain the model's predictions.
6. Making Predictions: Loads the trained model and scaler to predict customer churn.
7. Generating LIME Explanation: Visualizes feature importance for a sample prediction.
"""


data_preperocessing(TRAINING_DATASET, PREPROCESSED_DATA)
feature_engineering(PREPROCESSED_DATA, ENGINEERED_FEATURE)
model, X_test_scaled, y_test, results_dir, X_train_scaled, feature_names = train_pipeline(MODELS, ENGINEERED_FEATURE, RESULTS)
evaluate_model(model, X_test_scaled, y_test, results_dir)
generate_lime_explanation(model, X_train_scaled, X_test_scaled, feature_names, results_dir)


# Load the trained scaler
scaler_path = os.path.join(MODELS, "scaler.pkl")
scaler = joblib.load(scaler_path)

# Load the trained XGBoost model
model_path = os.path.join(MODELS, "xgboost_model.json")
loaded_model = xgb.XGBClassifier()
loaded_model.load_model(model_path)

#sample_data = np.array([[193.524658, 0.135618, 0.540071, 4.328502e+04, 0.0, 0.000000, 0.0, 0.0, 0.0, 12.0, 0.250000, 0.500000, 0.250000]])
sample_data = np.array([[261.97487482267,0.09639830508474577,0.5400714147079638,732806.940733319,1,-48.1582417576617,1,0,12,0.25,0.5,0.25,1]])
sample_data_scaled = scaler.transform(sample_data)

prediction = loaded_model.predict(sample_data_scaled)

if prediction[0] == 1:
    print("Prediction: Churn")
else:
    print("Prediction: No Churn")



# Generate LIME explanation
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=scaler.transform(np.random.rand(100, sample_data.shape[1])),  # Dummy training data with same shape
    feature_names=["customer_id",
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
        "basic_ratio"],
    class_names=["No Churn", "Churn"],
    discretize_continuous=True
)

# Explain the sample instance
exp = explainer.explain_instance(sample_data_scaled[0], loaded_model.predict_proba, num_features=sample_data.shape[1])

# Plot and save LIME explanation
fig = exp.as_pyplot_figure()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "lime_explanation_result.png"), bbox_inches="tight", dpi=300)
plt.show()
