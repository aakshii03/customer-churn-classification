# atiDot_churnPrediction

Predicting Customer Churn Using Time-Series Data


Repository Structure

    dataset folder: original churn dataset csv files
    models folder: to save the model's .plk, .json, etc files
    preprocessedDataset: csv file of feature engineering and final features
    results folder: save the images and json files of results



Steps to use the repository
1. update the ENV.txt file name to .env
2. update the file and folder paths in .env file according to you system
3. run main.py file it has entire pipeline for 
    "data_engineering.py" file to generate the features
    "final_preprocess_features.py" file to generate more relevent features and choose important features.
    "train.py" file to get the traing models and analysis results


Objective: Build a robust classification model that predicts which customers will churn
between Jan. 1, 2024 and Feb. 28, 2024, using time-series data with a double index
(customer_id and date). Demonstrate modeling and feature engineering skills, as well as
production-aware design practices.

Approach Taken:

Data Engineering: The data engineering pipeline processes customer transaction data by loading and sorting records chronologically. Missing values in transaction amounts are filled with the customer's average, while missing plan types are replaced with the most frequent plan. Various statistical features are generated, including total transactions, average spending, and transaction variability. Time-based features capture transaction trends, inactivity periods, and days since the last transaction. Rolling averages provide short- and long-term spending patterns, while plan-based features track plan switches, upgrades, and downgrades. Customers are also assigned transaction levels based on cumulative spending. The final processed dataset is saved for further analysis and modeling.

Feature Processing and selection: This script performs final feature generation and selection for customer transaction data. It first loads the engineered dataset and computes additional features such as transaction gap (ratio of inactivity to tenure), spending variability (standard deviation vs. mean transaction amount), loyalty score (customer tenure weighted by total spending and plan stability), and transaction trend score (short-term vs. long-term transaction difference). It also flags high churn risk for customers inactive for over two months. After feature engineering, the script retains only the most relevant columns, including transaction behavior, plan details, and churn labels. The processed dataset is then saved for modeling.
    
    transaction_gap – Ratio of days since the last transaction to the customer's total tenure, indicating inactivity relative to account age. Measures customer engagement. A higher gap suggests the customer is using the service less frequently, indicating a potential risk of churn.

    spending_variability – Ratio of the standard deviation of transaction amounts to the average transaction amount, measuring spending consistency.  Captures spending habits. High variability may suggest inconsistent usage, which could signal uncertainty in customer commitment.

    loyalty_score – A weighted measure of customer tenure and cumulative spending, adjusted by plan switch count to assess loyalty. Reflects long-term customer commitment. Higher values suggest a more engaged and stable customer, reducing the likelihood of churn.

    high_churn_risk – A binary flag (1 = high risk) indicating if a customer has been inactive for more than two months. Directly flags inactive customers, making it a strong indicator for predicting churn.

    transaction_trend_score – The difference between the 3-month and 6-month rolling average transaction amounts, capturing short-term vs. long-term spending trends. Identifies changes in spending behavior. A declining trend may indicate reduced engagement, which is often a precursor to churn.

Train XGBoost Model: This script builds a machine learning pipeline to predict customer churn using XGBoost. It loads and preprocesses data, handling missing values and encoding categorical features. The data is then split into training and testing sets, followed by feature scaling. The pipeline automates the entire workflow, ensuring efficient training and evaluation of the churn prediction model.

Results: The XGBoost model is trained and evaluated using metrics like accuracy and Matthews Correlation Coefficient (MCC). f1 score, precision, recall are also generated in results json file. The model and preprocessing tools are saved for future use, and LIME explanations are generated to interpret model predictions.

Accuracy: 94.16%
MCC: 86.45


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

Conclusion: I have tried models like logistic regression, Rando forest, decision tree and XGBoost. XGBoost model outperforms according to accuracy and matthews_corrcoef measures. Suggestion is to use XGBoost model for churn prediction.