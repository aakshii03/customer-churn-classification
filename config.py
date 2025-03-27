import os
from dotenv import load_dotenv

load_dotenv()

TRAINING_DATASET = os.getenv("TRAINING_DATASET")
ENGINEERED_FEATURES = os.getenv("ENGINEERED_FEATURES")
MODELS = os.getenv("MODELS")
PREPROCESSED_FEATURES = os.getenv("PREPROCESSED_FEATURES")
RESULTS = os.getenv("RESULTS")