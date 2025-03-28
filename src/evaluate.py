from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os


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
