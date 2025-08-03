# Credit Risk Analysis Prediction Dashboard

## Overview

This project delivers a machine learning-powered solution for assessing the likelihood of loan default based on borrower data. The dashboard enables financial institutions to make data-driven decisions by visualizing risk predictions and key insights.

## Features

- **End-to-End ML Pipeline:** Data ingestion, cleaning, exploratory data analysis (EDA), feature engineering, model training, and deployment.
- **Classification Models:** Trained and compared multiple classifiers, including XGBoost, to optimize performance.
- **High Accuracy:** Achieved 92% accuracy using optimized hyperparameters and robust validation techniques.
- **Interactive Dashboard:** Presents risk scores, feature contributions, and cohort analysis to support decision-making.
- **Deployment:** Web-based Flask application for serving predictions, with Tableau dashboards for interactive reporting.

## Technologies Used

- **Python**: Core programming language
- **Pandas, Scikit-learn, XGBoost**: Data processing & ML modeling
- **SQL**: Data extraction & management
- **Tableau**: Dashboard visualization
- **Flask**: Model deployment and API

## Solution Architecture

1. **Data Collection & Cleaning**: Ingest raw borrower data from SQL databases, clean and preprocess using Pandas.
2. **EDA & Feature Engineering**: Explore data patterns and engineer features for maximum predictive value.
3. **Model Training & Evaluation**: Train classifiers (Logistic Regression, Random Forest, XGBoost, etc.), perform hyperparameter tuning, validate results.
4. **Risk Prediction & Insights**: Deploy best model via Flask API, generate risk scores.
5. **Dashboard Visualization**: Tableau dashboards consume predictions for interactive analysis (e.g., risk segmentation, feature impact).

## Getting Started

1. Clone the repository.
2. Prepare your data in the specified format (see `/data` directory).
3. Run the ML pipeline (`ml_pipeline.ipynb` or `train_model.py`).
4. Start the Flask app for predictions (`app.py`).
5. Connect Tableau to the output database or API for visualization.

## Folder Structure

```
├── data/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
├── app.py
├── requirements.txt
├── tableau_dashboard/
└── README.md
```

## Role & Contributions

- Designed and implemented the complete ML pipeline.
- Performed extensive EDA and feature engineering.
- Trained and optimized multiple classifiers, achieving 92% accuracy.
- Built Flask app for model deployment.
- Developed Tableau dashboards for risk analysis and stakeholder reporting.

---

## Models Applied

- Logistic Regression
- Random Forest
- XGBoost

## Results

- Metrics: ROC AUC, F1 Score, Confusion Matrix
- Plots: Target distribution, Feature correlation, ROC Curve

## Dashboard Screenshot

![dashboard_screenshot](app/dashboard_screenshot.png)

**Contact:** For questions or collaboration, please reach out via GitHub Issues or email.


