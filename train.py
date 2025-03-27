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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from config import MODELS, PREPROCESSED_FEATURES, RESULTS


def loadData(filePath):
    """Load the preprocessed dataset."""
    return pd.read_csv(filePath)


def preprocessData(df):
    """Handle missing values, drop irrelevant columns, and encode categorical features."""
    dropColumns = ["customer_id", "date", "issuing_date", "first_transaction", "last_transaction", "first_plan", "last_plan"]
    df.drop(columns=dropColumns, inplace=True, errors="ignore")

    categoricalCols = df.select_dtypes(include=["object"]).columns
    labelEncoders = {}

    for col in categoricalCols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        labelEncoders[col] = le

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df, labelEncoders


def splitAndScaleData(df):
    """Split data into training and testing sets, then scale it."""
    X = df.drop(columns=["churn"])
    y = df["churn"]
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    joblib.dump(scaler, os.path.join(MODELS, "scaler.pkl"))

    return XTrainScaled, XTestScaled, yTrain, yTest, X.columns


def trainModels(XTrain, yTrain):
    """Train multiple models and return them."""
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    for name, model in models.items():
        model.fit(XTrain, yTrain)

    return models


def evaluateModels(models, XTest, yTest):
    """Evaluate trained models and save reports, confusion matrices."""
    for name, model in models.items():
        yPred = model.predict(XTest)

        accuracy = accuracy_score(yTest, yPred)
        mcc = matthews_corrcoef(yTest, yPred)
        report = classification_report(yTest, yPred, output_dict=True)
        report["matthewsCorrcoef"] = mcc  

        print(f"\nModel: {name}")
        print("Accuracy:", accuracy)
        print("MCC:", mcc)
        print(classification_report(yTest, yPred))

        modelFilename = name.lower()
        joblib.dump(model, os.path.join(MODELS, f"{modelFilename}.pkl"))

        reportPath = os.path.join(RESULTS, f"{modelFilename}_metrics.json")
        with open(reportPath, "w") as jsonFile:
            json.dump(report, jsonFile, indent=4)

        cm = confusion_matrix(yTest, yPred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        figPath = os.path.join(RESULTS, f"{modelFilename}_confusion_matrix.png")
        plt.savefig(figPath, bbox_inches="tight", dpi=300)
        plt.close()


def saveXGBoostModel(models):
    """Save the XGBoost model in JSON format."""
    if "XGBoost" in models:
        models["XGBoost"].save_model(os.path.join(MODELS, "xgboost_model.json"))


def saveLabelEncoders(labelEncoders):
    """Save label encoders for future use."""
    with open(os.path.join(MODELS, "label_encoders.pkl"), "wb") as f:
        pickle.dump(labelEncoders, f)


def generateLIMEExplanation(models, XTrain, XTest, featureNames):
    """Generate LIME explanations for XGBoost model."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        XTrain, feature_names=featureNames, class_names=["No Churn", "Churn"], discretize_continuous=True
    )

    instanceIdx = 5  
    instance = XTest[instanceIdx].reshape(1, -1)

    bestModel = models["XGBoost"]
    exp = explainer.explain_instance(instance.flatten(), bestModel.predict_proba)
    exp.as_pyplot_figure()
    plt.savefig(os.path.join(RESULTS, "lime_explanation.png"))
    plt.show()


def main():
    """Main function to execute the ML pipeline."""
    df = loadData(PREPROCESSED_FEATURES)
    df, labelEncoders = preprocessData(df)
    XTrainScaled, XTestScaled, yTrain, yTest, featureNames = splitAndScaleData(df)

    print("Training models...")
    models = trainModels(XTrainScaled, yTrain)

    print("Evaluating models...")
    evaluateModels(models, XTestScaled, yTest)

    print("Saving models and label encoders...")
    saveXGBoostModel(models)
    saveLabelEncoders(labelEncoders)

    print("Generating LIME explanation...")
    generateLIMEExplanation(models, XTrainScaled, XTestScaled, featureNames)

    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()


