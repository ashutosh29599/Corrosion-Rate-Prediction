features = ["Soil Resistivity (Ohm-m)", "pH", "Moisture Equivalent(%)", "exposure time", "Air pore space (%)",
            "Clay %", "Pipe Diameter (inches)"]

categorical_features = ["Iron and Steel Type"]

target = "Loss Oz/ft2"

score_var = "score"

models_to_score = ["Neural Network Regression", "Random Forest Regression", "Gradient Boosting Regressor"]

score_norm_functions = ["min_max_normalization", "z_score_normalization", "sigmoid_normalization", "log_transformation",
                        "softmax_normalization"]

error_functions = ["absolute_error", "squared_error"]
