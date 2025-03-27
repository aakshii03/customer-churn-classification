# atiDot_churnPrediction

Predicting Customer Churn Using Time-Series Data


Repository Structure

    dataset folder: original churn dataset csv files
    models folder: to save the model's .plk, .json, etc files
    preprocessedDataset: csv file of feature engineering and final features
    results folder: save the images and json files of results



Steps to use the repository
1. update the ENV.env file name to .env
2. update the file and folder paths in .env file according to you system
3. run "dataEngineering.py" file to generate the features
4. run "preprocessed_features.py" file to generate more relevent features and choose important features.
5. run "train.py" file to get the traing models and analysis results
6. run "useModel.py" to load the model and use it for inputs


Exercise: Predicting Customer Churn Using Time-Series Data

1.​ Data Preprocessing:
    ○​ Handle missing values.
        used mean value for "transaction_amount" column and used most frequent plan to fill "plan_type" column's missing values

    ○​ Appropriately encode (1-hot, embeddings, etc.) categorical features.
        used One Hot Emcoding for categorical features

2.​ Data Engineering:
    ○​ Generate at least 3 relevant date-dependent features.
        added date-dependent features "transaction_gap", "inactive_months", "customer_tenure", "days_since_last_txn" and many more depending on time and "transaction_amount", etc.

    ○​ Enrich the data with one relevant external source (e.g., economic indicators).
        these are the features with external source spending_variability, transaction_trend_score, avg_transaction, std_transaction, transaction_trend, rolling_avg_3m, plan_switch_count, is_downgraded, is_upgraded

3.​ Modeling:
    ○ ​Select an appropriate algorithm for churn classification, and implement it. Make
    sure to systematically tune and validate your model.
    tried different models Logistic Regression, Decision trees, random forest and XGBoost. XGBoost performs the best.

    ○ ​Be aware of data leakage, and provide a clear rationale for your choices.
        removed features "customer_id", "date", "issuing_date", "first_transaction", "last_transaction", "first_plan", "last_plan"

        below are some used features and their rationale

        Feature	                	            Rationale

        customer_tenure	                        Static, useful for long-term retention
        total_transactions                      Past behavior, not future-dependent
        avg_transaction, std_transaction	    Reflects customer spending behavior
        transaction_trend	                   	Helps predict declines in activity
        inactive_months	                    	Captures engagement level
        rolling_avg_3m, rolling_avg_6m	    	Lagged rolling averages avoid leakage
        is_downgraded, is_upgraded	        	If based on past events only
        plan_switch_count	                	Historical engagement metric

4.​ Model explanation:
    ○​ Use an appropriate method for brief model explanation (e.g., SHAP, PDP, LIME).
        Explained using LIME method

    ○​ Shortly describe why the method you chose is technically relevant.
        LIME is technically relevant because it helps interpret complex, black-box machine learning models by providing local explanations.
        It explains individual predictions rather than the entire model and Identifies which features influence a specific prediction.

Conclusion: XGBoost model outperforms according to accuracy and matthews_corrcoef measures. Suggestion is to use XGBoost model for churn prediction.