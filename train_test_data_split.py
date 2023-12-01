import pandas as pd
from sklearn.model_selection import train_test_split

features = ["Soil Resistivity (Ohm-m)", "pH", "Moisture Equivalent(%)", "exposure time", "Air pore space (%)",
            "Clay %", "Pipe Diameter (inches)", "Iron and Steel Type"]
target = "Loss Oz/ft2"


def load_data(filename):
    return pd.read_csv(filename)


def split_data(x, y):
    return train_test_split(x, y, test_size=0.5, random_state=42)


def store_file(x, y, filename):
    data = pd.DataFrame(x, columns=features)
    data[target] = y
    data.to_csv(filename, index=False)


def process_data_file():
    df = load_data("corrosion data combined.csv")
    x_original = df[features]
    y_original = df[target]
    x_model_train, x_score, y_model_train, y_score = split_data(x_original, y_original)
    store_file(x_model_train, y_model_train, "model_train/model_train.csv")
    x_score_train, x_test, y_score_train, y_test = split_data(x_score, y_score)
    store_file(x_score_train, y_score_train, "score_train/score_train.csv")
    store_file(x_test, y_test, "test/test.csv")


if __name__ == '__main__':
    process_data_file()
