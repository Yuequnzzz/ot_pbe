"""This module contains the Trainer class, which is responsible for training the model."""

import numpy as np
import pandas as pd
import ot
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time
import matplotlib.pyplot as plt

from src.utils.utils import (
    convert_back_pdf,
    get_column_name_id,
)


# from src.model_module.data_processing_for_training import calculate_errors
# from src.model_module.data_processing_for_training import train_test_NN

def calculate_error_metrics(
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        pdf_array: np.ndarray,
        dict_name_id: dict,
):
    """Calculate the error metrics

    Args:
        y_actual (np.ndarray): output actual values, including bins and other features
        y_predicted (np.ndarray): output predicted values, including bins and other features
        pdf_array (np.ndarray): Array with probability density values
        dict_name_id (dict): Dictionary with the column name and the corresponding column ID

    Returns:

    """
    # # get the concentration column
    # # TODO: increase the flexibility here, add params in yaml
    # c_id = dict_name_id["c"]
    # # c_rmse = mean_squared_error(y_actual[:, c_id], y_predicted[:, c_id], squared=False)
    # c_mae = mean_absolute_error(y_actual[:, c_id], y_predicted[:, c_id])
    # c_mape = mean_absolute_percentage_error(y_true=y_actual[:, c_id], y_pred=y_predicted[:, c_id])

    # get the num of crystals column
    num_crystals_id = dict_name_id["mu0"]
    # num_crystals_rmse = mean_squared_error(
    #     y_actual[:, num_crystals_id],
    #     y_predicted[:, num_crystals_id],
    #     squared=False
    # )
    num_crystals_mae = mean_absolute_error(
        y_actual[:, num_crystals_id],
        y_predicted[:, num_crystals_id],
    )
    num_crystals_mape = mean_absolute_percentage_error(
        y_actual[:, num_crystals_id],
        y_predicted[:, num_crystals_id],
    )

    # get the matrix of bins
    bins_ids = [dict_name_id[k] if k.startswith("target_value") else None for k in dict_name_id.keys()]
    bins_ids = [b for b in bins_ids if b is not None]
    bins_actual = y_actual[:, bins_ids]
    bins_predicted = y_predicted[:, bins_ids]

    # # calculate the wasserstein distance between the two pdfs
    # wasserstein_distance_matrix = calculate_wasserstein_distance(
    #     target_true_bins=bins_actual,
    #     target_predicted_bins=bins_predicted,
    #     pdf_array=pdf_array
    # )  # TODO: fix
    # calculate the rmse for bins
    # bin_rmse = mean_squared_error(bins_actual, bins_predicted, squared=False)
    bin_mae = mean_absolute_error(bins_actual, bins_predicted)
    bin_mape = mean_absolute_percentage_error(bins_actual, bins_predicted)

    # TODO: add other error metrics like mu1, mu2,...

    # get the total error
    error_metrics = {
        # "c_rmse": c_rmse,
        # "c_mae": c_mae,
        # "c_mape": c_mape,
        # "num_crystals_rmse": num_crystals_rmse,
        "num_crystals_mae" : num_crystals_mae,
        "num_crystals_mape": num_crystals_mape,
        # "wasserstein_distance_matrix": wasserstein_distance_matrix,
        # "bin_rmse": bin_rmse,
        "bin_mae": bin_mae,
        "bin_mape": bin_mape,
    }

    return error_metrics


# class Trainer:
#     """Trainer class to train the model"""
#
#     def __init__(self, data: pd.DataFrame, nodes_per_layer: int, layers: int, kFoldFlag: bool = False,
#                  n_splits: int = 5, saveFlag: bool = False):
#         """Initialize the Trainer class
#
#         Args:
#             data (pd.DataFrame): Dataframe with the data
#             nodes_per_layer (int): Number of nodes per layer
#             layers (int): Number of layers
#             kFoldFlag (bool, optional): Wether to do kFold crossvalidation. Defaults to False.
#             n_splits (int, optional): Number of splits for kFold crossvalidation. Defaults to 5.
#             saveFlag (bool, optional): Wether to save testing output vectors. Defaults to False.
#         """
#         self.data = data
#         self.nodes_per_layer = nodes_per_layer
#         self.layers = layers
#         self.kFoldFlag = kFoldFlag
#         self.n_splits = n_splits
#         self.saveFlag = saveFlag
#
#     def train(self):
#         """Train the model"""
#         X = self.data.drop(columns=["runID"]).to_numpy()
#         Y = self.data[["c", "mu0", "mu0_rel", "mu3", "mu3_rel", "av_len"]].to_numpy()
#
#         errors, runtimes, mlpr = train_test_NN(X, Y, self.nodes_per_layer, self.layers, self.kFoldFlag, self.n_splits,
#                                                self.saveFlag)
#
#         return errors, runtimes, mlpr


if __name__ == "__main__":
    input = pd.read_csv("../../data/data_cleaning/data_cleaned_input.csv")
    output = pd.read_csv("../../data/data_cleaning/data_cleaned_output.csv")

    # model
    start_time = time.time()
    nodes_per_layer = 400
    layers = 8
    model = MLPRegressor(
        hidden_layer_sizes=([nodes_per_layer] * layers),
        alpha=0
    )
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
    print("X_train shape: ", X_train.shape)


    # TODO: taking out c
    y_train = y_train.drop(columns='c')
    y_test = y_test.drop(columns='c')

    # get the column name and the corresponding column ID
    dict_name_id = get_column_name_id(y_train)

    # train the model
    model.fit(X_train, y_train)
    print("Model trained successfully")
    # predict the output
    y_pred = model.predict(X_test)
    print("Model predicted successfully")

    print("Duration:", time.time()-start_time)

    y_test.to_csv("../../data/data_results/data_output_test.csv", index=False)
    pd.DataFrame(y_pred).to_csv("../../data/data_results/data_output_predicted.csv", index=False)

    # # load the predicted output
    # y_pred = pd.read_csv("../../data/data_results/data_output_predicted.csv")
    # y_pred = y_pred.to_numpy()
    #
    # # load the quantiles
    # quantiles = pd.read_csv("../../data/data_cleaning/data_cleaned_quantiles.csv")
    # quantiles_array = quantiles.to_numpy()
    #
    # # load the intervals
    # target_intervals = pd.read_csv("../../data/data_cleaning/data_cleaned_target_intervals.csv")
    # target_intervals = target_intervals.to_numpy()
    # pdf_array = convert_back_pdf(quantiles_array, dl_array=target_intervals)
    #
    # # calculate the error metrics
    # error_metrics = calculate_error_metrics(y_test.to_numpy(), y_pred, pdf_array, dict_name_id)
    # print(error_metrics)
    #
    # # plot the results
    # # TODO: ATTENTION, take out the first bin loc value
    # cols = [f"target_value_{i}" for i in range(0, 80)]
    # for i in range(100):
    #     # get the ith row and cols in the y_test
    #     y_test_row = y_test.iloc[i][cols]
    #     # todo: modify the column id when taking out certain features
    #     # TODO: ATTENTION, take out the first bin loc value
    #     y_pred_row = y_pred[i, 1:-1]
    #     plt.plot(y_test_row, pdf_array[i], label="Actual")
    #     plt.plot(y_pred_row, pdf_array[i], label="Predicted")
    #     plt.legend()
    #     plt.show()


# todo: 1. check why some intervals are zero (x)
# todo: 2. use pytorch
# todo: 3. try wasserstein distance as the cost function
# todo: 4. use optuna to tune the hyperparameters
# todo: 5. how to constraint the predicted_bin is non-negative
