# atiDot_churnPrediction

## Predicting Customer Churn Using Time-Series Data

### Repository Structure

- **`data`**: Contains original churn dataset, feature engineered and final processed CSV files.
- **`models`**: Stores the trained model files (`.pkl`, `.json`, etc.).
- **`src`**: it has .py scripts
- **`output`**: Stores images and JSON files with results and metrics.

---

## Steps to Use the Repository

1. **Update the environment file**: Rename `ENV.txt` to `.env`.
2. **Modify paths**: Update file and folder paths in the `.env` file according to your system.
3. **Run the pipeline**: Execute `main.py` to initiate the full pipeline, which includes:
   - `data_engineering.py`: Generates initial features.
   - `final_preprocess_features.py`: Creates more refined features and selects important ones.
   - `train.py`: Trains models and analyzes results.

---

## Objective

Build a robust **classification model** to predict **customer churn** between **Jan 1, 2024, and Feb 28, 2024**, using **time-series data** with a **double index** (`customer_id`, `date`). The project focuses on **modeling, feature engineering, and production-aware design**.

---

## Approach Taken

### 1. Data Engineering

- **Preprocessing**: Loads and sorts transaction records chronologically.
- **Handling Missing Values**:
  - Missing **transaction amounts** → filled with customer’s average spending.
  - Missing **plan types** → replaced with the most frequent plan.
- **Feature Generation**:
  - **Transaction Statistics**: Total transactions, average spending, spending variability.
  - **Time-based Features**: Transaction trends, inactivity periods, days since last transaction.
  - **Rolling Averages**: Short-term vs. long-term spending patterns.
  - **Plan-Based Features**: Tracks plan switches, upgrades, and downgrades.
- **Final Processed Dataset**: Saved for further analysis and modeling.

---

### 2. Feature Processing & Selection

- Computes additional features for **customer transaction behavior analysis**.
- Retains only the most relevant columns for **better model performance**.

#### Generated Features & Their Importance

| **Feature** | **Description & Impact** |
|------------|---------------------------|
| **`transaction_gap`** | Ratio of days since the last transaction to the customer's total tenure, indicating inactivity relative to account age. Measures customer engagement. A higher gap suggests the customer is using the service less frequently, indicating a potential risk of churn. |
| **`spending_variability`** | Ratio of the standard deviation of transaction amounts to the average transaction amount, measuring spending consistency.  Captures spending habits. High variability may suggest inconsistent usage, which could signal uncertainty in customer commitment. |
| **`loyalty_score`** | A weighted measure of customer tenure and cumulative spending, adjusted by plan switch count to assess loyalty. Reflects long-term customer commitment. Higher values suggest a more engaged and stable customer, reducing the likelihood of churn. |
| **`high_churn_risk`** | A binary flag (1 = high risk) indicating if a customer has been inactive for more than 0.93 months. Directly flags inactive customers, making it a strong indicator for predicting churn. |
| **`transaction_trend_score`** | The difference between the 3-month and 6-month rolling average transaction amounts, capturing short-term vs. long-term spending trends. Identifies changes in spending behavior. A declining trend may indicate reduced engagement, which is often a precursor to churn. |

---

### 3. Training XGBoost Model

- **Pipeline**: Loads & preprocesses data, handles missing values, encodes categorical features, and scales features.
- **Model Training**: Uses **XGBoost**, a high-performance gradient-boosting algorithm.

---

### 4. Model Evaluation & Results

- **Evaluation Metrics**:
  - **Accuracy**: `94.16%`
  - **Matthews Correlation Coefficient (MCC)**: `86.45%`
  - **Additional Metrics**: F1-score, precision, and recall (saved in results JSON file).
- **Results**:
  - Model files and preprocessing tools are saved for future use.
  - **LIME explanations** help interpret predictions.

---

## Exercise: Predicting Customer Churn Using Time-Series Data

### 1. Data Preprocessing
**Handling Missing Values**:
   - Used mean value for `transaction_amount`.
   - Used the most frequent plan type for missing values in `plan_type`.

**Encoding Categorical Features**:
   - Used **One-Hot Encoding** for categorical data.

---

### 2. Data Engineering
**Generated Date-Dependent Features**:
   - `transaction_gap`, `inactive_months`, `customer_tenure`, `days_since_last_txn`, etc.

**External Data Features**:
   - Added **spending variability, transaction trend score, average transaction, standard deviation, plan switch count, etc.**

---

### 3. Modeling & Algorithm Selection
**Model Selection**:
   - Evaluated **Logistic Regression, Decision Trees, Random Forest, and XGBoost**.
   - **XGBoost performed best**.

**Avoided Data Leakage**:
   - Removed **irrelevant features**: `customer_id`, `date`, `issuing_date`, `first_transaction`, `last_transaction`, `first_plan`, `last_plan`.

**Key Features & Their Importance**:

| **Feature** | **Rationale** |
|------------|--------------|
| `customer_tenure_days` | Indicates long-term retention. |
| `total_transactions_count` | Represents past behavior. |
| `avg_transaction_amount`, `std_transaction` | Reflects spending behavior. |
| `transaction_trend` | Helps predict declining engagement. |
| `inactive_months` | Measures engagement levels. |
| `rolling_avg_3m`, `rolling_avg_6m` | Captures short- and long-term trends. |
| `is_plan_downgraded`, `is_upgraded` | Tracks past customer decisions. |
| `plan_switch_count` | Indicates historical engagement. |

---

### 4. Model Explanation
**Used LIME for Model Interpretation**:
   - LIME helps explain complex models by breaking down individual predictions.

**Why LIME?**
   - It provides local explanations for predictions rather than the entire model.
   - Highlights key features influencing a specific decision.

---

## Conclusion

**Tried models:** Logistic Regression, Decision Tree, Random Forest, and **XGBoost**.  
**Best Performing Model:** **XGBoost** (highest accuracy and MCC).  
**Recommendation:** Use **XGBoost** for **customer churn prediction**.  
