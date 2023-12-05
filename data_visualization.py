import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_loader import load_data


def plot_ml_model_metrics():
    category_attribute = "model_name"
    metric_attributes = ["R2", "MAE", "MSE", "RMSE"]
    df = load_data("test/test_model_metrics.csv")
    df["model_name"] = df["model_name"].apply(convert_to_abbreviation)

    for metric_attribute in metric_attributes:
        sns.scatterplot(data=df, x=category_attribute, y=metric_attribute, s=100)

        plt.xlabel(category_attribute)
        plt.ylabel(metric_attribute)
        plt.title(f"ML model performance ({metric_attribute}) ")

        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(left=0.1, right=0.83, top=0.9, bottom=0.2)

        plt.savefig(f"plots/ml_model_{metric_attribute}_plot.png")

        plt.show()


def extract_scoring_model(filename):
    filename = os.path.basename(filename)
    model = filename.removeprefix("score_model_").removesuffix(".csv")
    return model


def convert_to_abbreviation(input_string):
    words = input_string.split()
    abbreviation = ''.join([word[0].upper() for word in words])
    return abbreviation


def map_scoring_category(df):
    df["Scoring Category"] = df["score_norm_function"] + "_" + df["error_function"]
    df["Scoring Category"] = df["Scoring Category"].str.replace(r"_(normalization|transformation|error)", "",
                                                                regex=True)
    return df


def plot_metrics(df, metric_attribute, category_attribute, title, store_filename):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=category_attribute, y=metric_attribute, hue='model_name', s=100)
    plt.legend(title='model_name', bbox_to_anchor=(1.05, 1), loc='upper left')

    for i, category in enumerate(df[category_attribute].unique()):
        if i > 0:
            plt.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=1)

    plt.xlabel(category_attribute)
    plt.ylabel(metric_attribute)
    plt.title(title)

    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(left=0.1, right=0.83, top=0.9, bottom=0.2)

    plt.savefig(store_filename)

    plt.show()


def plot_score_metrics():
    folder_path = "test/score_model_*.csv"
    score_model_metrics_files = glob.glob(folder_path)

    category_attribute = "Scoring Category"
    metric_attributes = ["R2", "MAE", "MSE", "RMSE"]
    dfs = []
    for metrics_file in score_model_metrics_files:
        df = load_data(metrics_file)
        df = map_scoring_category(df)
        df["model_name"] = convert_to_abbreviation(extract_scoring_model(metrics_file))
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    for metric_attribute in metric_attributes:
        plot_metrics(combined_df, metric_attribute, category_attribute,
                     f"Hybrid model performance ({metric_attribute}) for different scoring category",
                     f"plots/hybrid_model_{metric_attribute}_plot.png")


if __name__ == '__main__':
    plot_ml_model_metrics()
    plot_score_metrics()
