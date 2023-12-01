from sklearn.model_selection import train_test_split

from data_configurations import features, categorical_features, target
from data_loader import load_data, store_file


def split_data(x, y):
    return train_test_split(x, y, test_size=0.5, random_state=42)


def process_data_file():
    df = load_data("corrosion data combined.csv")
    all_features = features + categorical_features
    x_original = df[all_features]
    y_original = df[target]
    x_model_train, x_score, y_model_train, y_score = split_data(x_original, y_original)
    store_file(x_model_train, y_model_train, all_features, "model_train/model_train.csv")
    x_score_train, x_test, y_score_train, y_test = split_data(x_score, y_score)
    store_file(x_score_train, y_score_train, all_features, "score_train/score_train.csv")
    store_file(x_test, y_test, all_features, "test/test.csv")


if __name__ == '__main__':
    process_data_file()
