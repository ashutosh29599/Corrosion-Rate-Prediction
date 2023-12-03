import os

import numpy as np

from data_configurations import models_to_score, target, score_var
from data_loader import load_data


def min_max_normalization(predictions):
    min_val = min(predictions)
    max_val = max(predictions)
    normalized_predictions = [(x - min_val) / (max_val - min_val) for x in predictions]
    return normalized_predictions


def z_score_normalization(predictions):
    mean_val = np.mean(predictions)
    std_dev = np.std(predictions)
    normalized_predictions = [(x - mean_val) / std_dev for x in predictions]
    return normalized_predictions


def log_transformation(predictions, epsilon=1e-10):
    normalized_predictions = [np.log(x + epsilon) for x in predictions]
    return normalized_predictions


def sigmoid_normalization(predictions):
    normalized_predictions = [1 / (1 + np.exp(-x)) for x in predictions]
    return normalized_predictions


def softmax_normalization(predictions):
    exp_predictions = np.exp(predictions)
    normalized_predictions = exp_predictions / np.sum(exp_predictions)
    return normalized_predictions.tolist()


def absolute_error(true_value, predicted_value):
    return np.abs(true_value - predicted_value)


def squared_error(true_value, predicted_value):
    return (true_value - predicted_value) ** 2


def score(true_values, model_predictions, score_norm_function=softmax_normalization, error_function=squared_error):
    all_scores = []

    for i in range(len(true_values)):
        differences = []
        for predictions in model_predictions:
            differences.append(error_function(true_values[i], predictions[i]))

        prediction_scores = score_norm_function(differences)
        all_scores.append(prediction_scores)

    return all_scores


def build_all_score():
    score_df = load_data("score_train/score_train.csv")
    true_values = score_df[target].values
    model_predictions = []
    predicted_dfs = []
    for model in models_to_score:
        predicted_df = load_data(f"score_train/predictions_{model}.csv")
        predicted_values = predicted_df[target].values
        predicted_dfs.append(predicted_df)
        model_predictions.append(predicted_values)

    score_norm_functions = [min_max_normalization, z_score_normalization, sigmoid_normalization, log_transformation,
                            softmax_normalization]
    error_functions = [absolute_error, squared_error]
    for score_norm_function in score_norm_functions:
        for error_function in error_functions:
            build_score(error_function, model_predictions, predicted_dfs, score_norm_function, true_values)


def build_score(error_function, model_predictions, predicted_dfs, score_norm_function, true_values):
    all_scores = score(true_values, model_predictions, score_norm_function, error_function)
    model_scores = np.transpose(np.array(all_scores))
    for i in range(len(models_to_score)):
        model = models_to_score[i]
        predicted_df = predicted_dfs[i]
        score_values = model_scores[i]
        directory = f"score_train/score/{score_norm_function.__name__} and {error_function.__name__}"
        os.makedirs(directory, exist_ok=True)
        file = f"score_{model}.csv"
        store_score_file(predicted_df, score_values, filename=f"{directory}/{file}")


def store_score_file(df, score_values, filename):
    df[score_var] = score_values
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    build_all_score()
