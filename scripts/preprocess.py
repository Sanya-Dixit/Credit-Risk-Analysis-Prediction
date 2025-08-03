import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_raw_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Keep only relevant loan_status values
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    # Features to use
    num_features = ['annual_inc', 'loan_amnt', 'fico_score']
    cat_features = ['purpose', 'home_ownership']

    # Handle missing values
    for col in num_features:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_features:
        df[col] = df[col].fillna('missing')

    # Split features and label
    X = df[num_features + cat_features]
    y = df['loan_status']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    X_processed = preprocessor.fit_transform(X)

    # Save processed features and label
    processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)
    processed_df.to_csv('data/processed/processed.csv', index=False)
    y.to_csv('data/processed/labels.csv', index=False)
    return processed_df, y

if __name__ == "__main__":
    df = load_raw_data('data/raw/lending_club.csv')
    clean_data(df)