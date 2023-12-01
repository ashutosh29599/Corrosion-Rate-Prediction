import glob
import logging

import joblib
from sklearn import metrics

from data_loader import load_data, load_features_and_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="logs/model_prediction.log", filemode="a")


def predict_model():
    df = load_data("score_train/score_train.csv")
    x, y = load_features_and_target(df)

    folder_path = "model_train/*.joblib"
    joblib_files = glob.glob(folder_path)

    for joblib_file in joblib_files:
        loaded_model = joblib.load(joblib_file)
        y_pred = loaded_model.predict(x)
        model_name = joblib_file.split(".")[0]

        logging.info("--------------------")
        logging.info(f"{model_name} Performance:")
        logging.info(f"Mean Absolute Error: {metrics.mean_absolute_error(y, y_pred)}")
        logging.info(f"Mean Squared Error: {metrics.mean_squared_error(y, y_pred)}")
        logging.info(f"Root Mean Squared Error: {metrics.mean_squared_error(y, y_pred, squared=False)}")
        logging.info(f"R2 Score:{metrics.r2_score(y, y_pred)}")
        logging.info("--------------------")


if __name__ == '__main__':
    predict_model()