import pandas as pd

def load_data(path):
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)

def save_data(df, path):
    """Saves a pandas DataFrame to CSV."""
    df.to_csv(path, index=False)

def get_feature_importance(model, feature_names):
    """Returns feature importances if supported."""
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    return None