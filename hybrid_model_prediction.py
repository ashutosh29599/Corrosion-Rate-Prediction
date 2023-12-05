import logging
import sys

import joblib
import numpy as np
from sklearn import metrics

from data_configurations import score_norm_functions, error_functions, models_to_score
from data_loader import load_data, load_features_and_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="logs/hybrid_model_prediction.log", filemode="a")

# score_base_model = "Gradient Boosting Regressor"
# score_base_model = "Linear Regression"
# score_base_model = "Neural Network Regression"
score_base_model = "Random Forest Regression"


def load_score_models(score_norm_function, error_function):
    score_models = {}
    for model in models_to_score:
        filename = f"score_train/score/{score_norm_function} and {error_function}/{model}/{score_base_model}.joblib"
        score_models[model] = joblib.load(filename)

    return score_models


def load_prediction_models():
    prediction_models = {}
    for model in models_to_score:
        filename = f"model_train/{model}.joblib"
        prediction_models[model] = joblib.load(filename)

    return prediction_models


def predict_model():
    df = load_data("test/test.csv")
    x, y = load_features_and_target(df)
    prediction_models = load_prediction_models()

    logging.info(f"--Using Score Base Model: {score_base_model}--")

    for score_norm_function in score_norm_functions:
        for error_function in error_functions:
            y_pred = []
            score_models = load_score_models(score_norm_function, error_function)
            for row in x.itertuples():
                feature_value = np.array([row[1:]])
                min_score = sys.maxsize
                min_score_model = ""
                for score_model_name, score_model in score_models.items():
                    score = score_model.predict(feature_value)
                    if score < min_score:
                        min_score = score
                        min_score_model = score_model_name

                pred_value = prediction_models[min_score_model].predict(feature_value)
                y_pred.append(pred_value)

            y_pred = np.array(y_pred)

            logging.info("--------------------")
            logging.info(f"{score_norm_function}--{error_function} Performance:")
            logging.info(f"Mean Absolute Error: {metrics.mean_absolute_error(y, y_pred)}")
            logging.info(f"Mean Squared Error: {metrics.mean_squared_error(y, y_pred)}")
            logging.info(f"Root Mean Squared Error: {metrics.mean_squared_error(y, y_pred, squared=False)}")
            logging.info(f"R2 Score:{metrics.r2_score(y, y_pred)}")
            logging.info("--------------------")


if __name__ == '__main__':
    predict_model()
