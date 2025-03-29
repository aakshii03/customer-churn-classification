import os
from dotenv import load_dotenv

load_dotenv()

TRAINING_DATASET = os.getenv("TRAINING_DATASET")
PREPROCESSED_DATA = os.getenv("PREPROCESSED_DATA")
MODELS = os.getenv("MODELS")
ENGINEERED_FEATURE = os.getenv("ENGINEERED_FEATURE")
RESULTS = os.getenv("RESULTS")