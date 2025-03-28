import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import os

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
    fig.set_size_inches(10, 8) 
    plt.tight_layout() 
    plt.savefig(os.path.join(results_dir, "lime_explanation.png"), bbox_inches="tight", dpi=300)
    plt.show()
