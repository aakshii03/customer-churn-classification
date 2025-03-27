import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import MODELS
import numpy as np
import joblib
import xgboost as xgb
import os



# Load the trained scaler
scaler_path = os.path.join(MODELS, "scaler.pkl")
scaler = joblib.load(scaler_path)

# Load the trained XGBoost model
model_path = os.path.join(MODELS, "xgboost_model.json")
loaded_model = xgb.XGBClassifier()
loaded_model.load_model(model_path)

sample_data = np.array([[193.524658, 0.135618, 0.540071, 4.328502e+04, 0.0, 0.000000, 0.0, 0.0, 0.0, 12.0, 0.250000, 0.500000, 0.250000]])

sample_data_scaled = scaler.transform(sample_data)

prediction = loaded_model.predict(sample_data_scaled)

if prediction[0] == 1:
    print("Prediction: Churn")
else:
    print("Prediction: No Churn")
