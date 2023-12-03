import logging
import os

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from data_configurations import target, score_var, score_norm_functions, error_functions, models_to_score
from data_loader import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="logs/score_training.log", filemode="a")

models = {
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
}

param_dicts = {
    "Gradient Boosting Regressor": {"params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2],
                                               "max_depth": [3, 4, 5]}, "iters": 10},
}


def save_model(model, filename):
    joblib.dump(model, filename)


def load_features_and_target(df):
    columns = df.columns.tolist()
    columns.remove(target)
    columns.remove(score_var)
    x = df[columns]
    y = df[score_var]
    return x, y


def train_score_models(df, directory, trained_model):
    x, y = load_features_and_target(df)

    for model_name, model in models.items():
        logging.info(f"Training and tuning {directory}/{trained_model} : {model_name}...")

        if model_name in param_dicts:
            param_dist = param_dicts[model_name]["params"]
            grid_search = GridSearchCV(model, param_grid=param_dist, cv=5, scoring='r2')
            grid_search.fit(x.values, y.values)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logging.info(f"Best hyperparameters for {directory}/{trained_model} : {model_name}: {best_params}")
        else:
            best_model = model
            best_model.fit(x.values, y.values)

        os.makedirs(f"{directory}/{trained_model}", exist_ok=True)
        filename = f"{directory}/{trained_model}/{model_name}.joblib"
        save_model(best_model, filename)


def train_all_score_models():
    for score_norm_function in score_norm_functions:
        for error_function in error_functions:
            for model in models_to_score:
                directory = f"score_train/score/{score_norm_function} and {error_function}"
                df = load_data(f"{directory}/score_{model}.csv")
                train_score_models(df, directory, model)


if __name__ == '__main__':
    train_all_score_models()
