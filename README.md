# Credit-Risk-Analysis-and-Prediction
Predict credit default risk based on borrower financial data. Uses machine learning models to classify whether a loan applicant is likely to default or not. Designed to support risk analysts and lending teams in making data-driven decisions.


## Dataset

- Source: [LendingClub Dataset (Kaggle)](https://www.kaggle.com/datasets/lendingclub/loan-data)
- Target: `loan_status` (Fully Paid = 0, Charged Off = 1)

## Features Used

- Annual Income
- Loan Amount
- FICO Score
- Purpose
- Home Ownership

## Models Applied

- Logistic Regression
- Random Forest
- XGBoost

## Results

- Metrics: ROC AUC, F1 Score, Confusion Matrix
- Plots: Target distribution, Feature correlation, ROC Curve

## How to Run the App

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run preprocessing: `python scripts/preprocess.py`
4. Train models: `python scripts/train_model.py`
5. Launch Streamlit demo: `streamlit run app/app.py`

## Dashboard Screenshot

![dashboard_screenshot](app/dashboard_screenshot.png)
