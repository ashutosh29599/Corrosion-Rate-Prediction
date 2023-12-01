import logging

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from data_loader import load_data, load_features_and_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="logs/model_training.log", filemode="a")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Neural Network Regression": MLPRegressor(max_iter=2000),
    "Random Forest Regression": RandomForestRegressor(),
    "Ridge Regression": Ridge(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
}

param_dicts = {
    "Decision Tree Regression": {"params": {"max_depth": [None, 10, 20, 40]}, "iters": 4},
    "Neural Network Regression": {"params": {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50, 25)]},
                                  "iters": 4},
    "Random Forest Regression": {"params": {"max_depth": [None, 10, 20], "n_estimators": [50, 100, 200],
                                            "max_features": ["sqrt", "log2"]}, "iters": 10},
    "Ridge Regression": {"params": {"alpha": [0.1, 1.0, 5.0, 10.0, 20.0]}, "iters": 5},
    "Gradient Boosting Regressor": {"params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2],
                                               "max_depth": [3, 4, 5]}, "iters": 10},
}


def save_model(model, filename):
    joblib.dump(model, filename)


def train_all_models():
    df = load_data("model_train/model_train.csv")

    x, y = load_features_and_target(df)

    for model_name, model in models.items():
        logging.info(f"Training and tuning {model_name}...")

        if model_name in param_dicts:
            param_dist = param_dicts[model_name]["params"]
            n_iter = param_dicts[model_name]["iters"]
            randomized_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=5,
                                                   n_jobs=-1,
                                                   random_state=42)
            randomized_search.fit(x.values, y.values)
            best_model = randomized_search.best_estimator_
            best_params = randomized_search.best_params_
            logging.info(f"Best hyperparameters for {model_name}: {best_params}")
        else:
            best_model = model
            best_model.fit(x.values, y.values)

        save_model(best_model, f"model_train/{model_name}.joblib")


if __name__ == '__main__':
    train_all_models()
