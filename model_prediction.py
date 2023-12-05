import glob
import logging
import os

import joblib
import pandas as pd
from sklearn import metrics

from data_configurations import data_files
from data_loader import load_data, load_features_and_target, store_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="logs/model_prediction.log", filemode="a")


def predict_model(data_file_type, is_store_file=True):
    filename = data_files[data_file_type]
    df = load_data(filename)
    x, y = load_features_and_target(df)

    folder_path = "model_train/*.joblib"
    joblib_files = glob.glob(folder_path)

    metrics_list = []
    for joblib_file in joblib_files:
        loaded_model = joblib.load(joblib_file)
        y_pred = loaded_model.predict(x)
        filename = os.path.basename(joblib_file)
        model_name = filename.split(".")[0]

        if is_store_file:
            features = x.columns.tolist()
            store_file(x, y_pred, features, filename=f"score_train/predictions_{model_name}.csv")

        mae = metrics.mean_absolute_error(y, y_pred)
        mse = metrics.mean_squared_error(y, y_pred)
        rmse = metrics.mean_squared_error(y, y_pred, squared=False)
        r2 = metrics.r2_score(y, y_pred)

        logging.info("--------------------")
        logging.info(f"{model_name} Performance:")
        logging.info(f"Mean Absolute Error: {mae}")
        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"Root Mean Squared Error: {rmse}")
        logging.info(f"R2 Score:{r2}")
        logging.info("--------------------")

        metrics_list.append({"model_name": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"test/{data_file_type}_model_metrics.csv")


if __name__ == '__main__':
    # data_file = "score_train"
    data_type = "test"
    predict_model(data_type, is_store_file=False)
