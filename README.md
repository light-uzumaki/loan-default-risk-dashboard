# Loan Default Risk Prediction & Analysis Dashboard

An end-to-end machine learning and analytics project that predicts the probability of credit card payment default using customer financial behavior, repayment history, and demographic features.

This project combines **data cleaning, SQL-based analysis, machine learning model comparison, threshold tuning, feature importance analysis, and an interactive Streamlit dashboard**.

---

## 🚀 Live Demo

Streamlit App: (https://loan-default-risk-dashboard.streamlit.app/) 
GitHub Repository: https://github.com/light-uzumaki/loan-default-risk-dashboard

---

## 📌 Project Overview

Credit default prediction is an important problem in financial analytics. Banks and lenders need to identify customers who may fail to repay on time so they can make better risk-management decisions.

This project predicts whether a customer is likely to default on next month’s credit card payment.

The dashboard helps answer:

- Which customer segments have higher default risk?
- Which financial and repayment behavior features are most useful for predicting default?
- How do different risk thresholds affect business decisions?
- What is the predicted default risk for a new customer?

---

## 🧠 Problem Statement

The goal is to build a machine learning system that classifies customers as default-risk or non-default-risk based on their credit limit, bill amounts, payment amounts, repayment delays, age, education, marital status, and other customer attributes.

This is a binary classification problem:

```text
0 = No Default
1 = Default
```

---

## 🛠️ Tech Stack

| Category | Tools Used |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| SQL Analysis | SQLite |
| Machine Learning | scikit-learn |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Model Saving | Joblib |
| Version Control | Git, GitHub |

---

## 📊 Dataset

This project uses the **Default of Credit Card Clients** dataset.

The dataset contains customer-level information such as:

- Credit limit
- Age
- Sex
- Education
- Marital status
- Past repayment status
- Bill amounts
- Payment amounts
- Default status for next month

The original target column is:

```text
default_payment_next_month
```

After cleaning, it is renamed to:

```text
is_default
```

---

## 🔧 Data Cleaning & Feature Engineering

The cleaning pipeline is written inside `utils.py`.

Main cleaning steps:

- Standardized column names
- Removed unnecessary index columns
- Renamed the target variable
- Converted numeric columns safely
- Handled missing values
- Mapped categorical variables such as sex, education, and marriage
- Created additional analytical features for risk analysis and modeling

Engineered features include:

| Feature | Description |
|---|---|
| `avg_bill_amt` | Average bill amount across previous months |
| `avg_pay_amt` | Average payment amount across previous months |
| `payment_ratio` | Ratio of average payment to average bill amount |
| `months_with_delay` | Number of months with delayed payments |
| `max_delay` | Maximum repayment delay |
| `avg_delay` | Average repayment delay |
| `age_group` | Customer age segment |
| `limit_bucket` | Credit limit segment |

---

## 🧾 SQL-Based Risk Analysis

The cleaned dataset is loaded into a SQLite database for business-style analysis.

SQL queries are used to calculate:

- Default rate by age group
- Default rate by credit limit bucket
- Default rate by education level
- Default rate by marital status
- Default rate by delayed-payment months
- High-risk customer segments

SQL output files are saved inside:

```text
reports/sql_outputs/
```

---

## 🤖 Machine Learning Pipeline

The project trains and compares multiple machine learning models using scikit-learn.

Models used:

- Logistic Regression
- Random Forest Classifier
- HistGradientBoostingClassifier

The final model is selected based on **PR-AUC**, because default prediction is an imbalanced classification problem where the minority class, defaulters, is more important to identify correctly.

---

## 📈 Model Evaluation Metrics

The models are evaluated using:

| Metric | Meaning |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Out of predicted defaulters, how many actually defaulted |
| Recall | Out of actual defaulters, how many were correctly identified |
| F1-score | Balance between precision and recall |
| ROC-AUC | Model's ranking ability across thresholds |
| PR-AUC | Performance on the minority/default class |
| Confusion Matrix | Breakdown of correct and incorrect predictions |

Accuracy alone is not enough for this problem because most customers do not default. A model can achieve high accuracy by mostly predicting “No Default” while still missing risky customers.

That is why this project focuses on **PR-AUC, recall, precision, and threshold tuning**.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard with five pages.

---

### 1. Executive Overview

Shows high-level KPIs:

- Total customers
- Default rate
- Average credit limit
- Average age
- Customers with delayed payments

Also includes:

- Default vs non-default distribution
- Credit limit distribution

---

### 2. Risk Analysis

Analyzes default risk across different customer segments:

- Age group
- Credit limit bucket
- Education
- Marriage
- Sex
- Months with delayed payments
- Maximum delay

This page helps identify which customer groups have higher default risk.

---

### 3. Model Performance

Displays:

- Model comparison table
- ROC-AUC, PR-AUC, and F1-score comparison
- Feature importance chart

This page shows which model performed best and which features influenced predictions most.

---

### 4. Threshold Simulator

Allows users to adjust the default-risk threshold and observe how model behavior changes.

For each threshold, the dashboard shows:

- Precision
- Recall
- F1-score
- Customers flagged as risky
- True positives
- False positives
- False negatives
- True negatives
- Confusion matrix

This is useful because real risk decisions often require custom thresholds instead of using the default `0.50` cutoff.

---

### 5. Customer Risk Predictor

A real-time prediction form where users can enter customer details such as:

- Credit limit
- Age
- Sex
- Education
- Marital status
- Current payment delay
- Previous payment delay
- Bill amount
- Payment amount

The app predicts:

- Default risk probability
- Risk band
- Recommended action

Example output:

```text
Default Risk: 64.25%
Risk Band: High Risk
Recommended Action: High-risk customer. Strong manual review required.
```

---

## 📂 Project Structure

```text
loan-default-risk-dashboard/
│
├── data/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv
│   └── processed/
│       └── credit_clean.csv
│
├── database/
│   └── credit_risk.db
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── reports/
│   ├── sql_outputs/
│   │   ├── default_by_age_group.csv
│   │   ├── default_by_credit_limit.csv
│   │   ├── default_by_education.csv
│   │   ├── default_by_marriage.csv
│   │   ├── default_by_delay_months.csv
│   │   └── high_risk_segments.csv
│   │
│   ├── model_comparison.csv
│   ├── threshold_metrics.csv
│   ├── feature_importance.csv
│   └── test_predictions.csv
│
├── utils.py
├── 01_clean_data.py
├── 02_sql_analysis.py
├── 03_train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/light-uzumaki/loan-default-risk-dashboard.git
cd loan-default-risk-dashboard
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

---

### 3. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For Mac/Linux:

```bash
source venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Add the raw dataset

Place the dataset file here:

```text
data/raw/UCI_Credit_Card.csv
```

If the processed dataset and trained model are already included in the repository, you can directly run the Streamlit app.  
If you want to reproduce the full pipeline from scratch, continue with the steps below.

---

### 6. Run data cleaning

```bash
python 01_clean_data.py
```

This creates:

```text
data/processed/credit_clean.csv
```

---

### 7. Run SQL analysis

```bash
python 02_sql_analysis.py
```

This creates:

```text
database/credit_risk.db
reports/sql_outputs/
```

---

### 8. Train machine learning models

```bash
python 03_train_model.py
```

This creates:

```text
models/best_model.pkl
models/model_metadata.json
reports/model_comparison.csv
reports/threshold_metrics.csv
reports/feature_importance.csv
reports/test_predictions.csv
```

---

### 9. Run the Streamlit dashboard

```bash
python -m streamlit run app.py
```

The dashboard will open at:

```text
http://localhost:8501
```

---

## 📌 Key Output Files

| File | Description |
|---|---|
| `data/processed/credit_clean.csv` | Cleaned dataset |
| `database/credit_risk.db` | SQLite database created for SQL analysis |
| `reports/sql_outputs/` | SQL query output files |
| `reports/model_comparison.csv` | Model performance comparison |
| `reports/threshold_metrics.csv` | Threshold tuning metrics |
| `reports/feature_importance.csv` | Feature importance results |
| `reports/test_predictions.csv` | Test-set predictions used for threshold simulation |
| `models/best_model.pkl` | Saved trained model pipeline |
| `models/model_metadata.json` | Metadata containing feature columns and model details |

---

## 💡 Key Analytical Questions Explored

This project explores questions such as:

- Do customers with delayed payments have higher default risk?
- Which credit limit segments are riskier?
- How does default rate vary across age groups?
- How does repayment behavior affect default probability?
- Which features are most important for predicting default?
- How does changing the risk threshold affect false positives and false negatives?

---

## 📊 Business Value

This project demonstrates how machine learning can support credit-risk decision-making.

The dashboard can help a risk team:

- Identify high-risk customer groups
- Understand repayment behavior patterns
- Compare model performance
- Adjust risk thresholds based on business priorities
- Estimate default risk for individual customers
- Support manual review decisions

---

## ⚠️ Limitations

This project is built for learning and portfolio purposes.

It is not a real banking approval system.

Important limitations:

- The dataset is historical and may not represent current customer behavior.
- The model does not include regulatory or compliance checks.
- The model does not perform fairness auditing.
- The predictions should not be used for real lending decisions.
- More validation would be required before using this system in production.

---

## 🔮 Future Improvements

Possible improvements include:

- Add hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Add probability calibration for more reliable risk scores
- Add SHAP-based model explainability
- Add PostgreSQL integration
- Deploy the dashboard on Streamlit Community Cloud
- Add downloadable customer risk reports
- Add fairness analysis across demographic groups
- Add model monitoring for future data drift

---

## 📌 Resume Bullet Points

```text
Loan Default Risk Prediction & Analysis Dashboard | Python · scikit-learn · Pandas · SQL · Streamlit · Plotly

• Built an end-to-end ML pipeline to analyze credit customer behavior and predict next-month default risk using Python, Pandas, and scikit-learn.

• Performed SQL-based risk analysis to identify default patterns across credit limit, age group, education level, marital status, and delayed-payment behavior.

• Trained and compared Logistic Regression, Random Forest, and HistGradientBoostingClassifier models using ROC-AUC, PR-AUC, precision, recall, F1-score, and confusion matrix.

• Added a threshold-tuning simulator to show how different risk cutoffs affect false positives, false negatives, recall, and customers flagged for manual review.

• Developed an interactive Streamlit dashboard with Plotly visuals, model-performance reports, feature-importance analysis, and a real-time customer default-risk predictor.
```

---

## 👤 Author

**Avnish Tewari**

GitHub: [light-uzumaki](https://github.com/light-uzumaki)

---

## 📄 Disclaimer

This project is for educational and portfolio demonstration purposes only.  
It should not be used for actual financial, lending, or credit approval decisions.
