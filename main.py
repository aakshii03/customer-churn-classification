import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import MODELS
import numpy as np
import joblib
import xgboost as xgb
import os
from config import MODELS, PREPROCESSED_FEATURES, RESULTS, TRAINING_DATASET, ENGINEERED_FEATURES, ENGINEERED_FEATURES, PREPROCESSED_FEATURES
from source_code.data_engineering import data_engineering
from source_code.final_preprocess_features import preprocess_features
from source_code.train import train_pipeline
import lime.lime_tabular
import matplotlib.pyplot as plt

data_engineering(TRAINING_DATASET, ENGINEERED_FEATURES)
preprocess_features(ENGINEERED_FEATURES, PREPROCESSED_FEATURES)
train_pipeline(MODELS, PREPROCESSED_FEATURES, RESULTS)



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
        "basic_ratio"],  # Replace with actual feature names
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
