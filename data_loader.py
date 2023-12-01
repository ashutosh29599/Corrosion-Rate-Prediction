import pandas as pd

from data_configurations import features, categorical_features, target


def load_data(filename):
    return pd.read_csv(filename)


def load_features_and_target(df):
    if len(categorical_features) > 0:
        df = pd.get_dummies(df, columns=categorical_features)
    categorical_features_expanded = [c for c in df.columns.values if "_" in c]
    new_features = features + categorical_features_expanded
    x = df[new_features]
    y = df[target]
    return x, y
