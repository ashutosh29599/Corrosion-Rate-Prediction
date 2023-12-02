import numpy as np

from data_configurations import models_to_score, target
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
    return normalized_predictions


def absolute_error(true_value, predicted_value):
    return np.abs(true_value - predicted_value)


def squared_error(true_value, predicted_value):
    return (true_value - predicted_value) ** 2


def score(true_values, model_predictions, score_norm_function=softmax_normalization, error_function=squared_error):
    all_scores = []

    for i in range(len(true_values)):
        absolute_diffs = []
        for predictions in model_predictions:
            absolute_diffs.append(error_function(true_values[i], predictions[i]))

        prediction_scores = score_norm_function(absolute_diffs)
        all_scores.append(prediction_scores.tolist())

    return all_scores


def build_score():
    score_df = load_data("score_train/score_train.csv")
    true_values = score_df[target].values
    model_predictions = []
    predicted_dfs = []
    for model in models_to_score:
        predicted_df = load_data(f"score_train/predictions_{model}.csv")
        predicted_values = predicted_df[target].values
        predicted_dfs.append(predicted_df)
        model_predictions.append(predicted_values)

    all_scores = score(true_values, model_predictions)
    model_scores = np.transpose(np.array(all_scores))

    for i in range(len(models_to_score)):
        model = models_to_score[i]
        predicted_df = predicted_dfs[i]
        score_values = model_scores[i]
        store_score_file(predicted_df, score_values, filename=f"score_train/score_{model}.csv")


def store_score_file(df, score_values, filename):
    df["score"] = score_values
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    build_score()
