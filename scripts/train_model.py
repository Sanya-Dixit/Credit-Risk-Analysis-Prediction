import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import pickle

def load_processed_data():
    X = pd.read_csv('data/processed/processed.csv')
    y = pd.read_csv('data/processed/labels.csv').values.ravel()
    return X, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # Save test split for evaluation notebook
    X_test.to_csv('data/processed/X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        roc = roc_auc_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        print(f'--- {name} ---')
        print('ROC AUC:', roc)
        print('F1 Score:', f1)
        print('Confusion Matrix:\n', cm)
        print(classification_report(y_test, preds))
        results[name] = {'model': model, 'roc_auc': roc, 'f1': f1, 'conf_matrix': cm}

    # Save the best performing model (example: XGBoost)
    best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_model = results[best_model_name]['model']
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

if __name__ == "__main__":
    X, y = load_processed_data()
    train_and_evaluate(X, y)