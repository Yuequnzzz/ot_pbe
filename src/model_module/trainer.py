"""This module contains the Trainer class, which is responsible for training the model."""

import numpy as np
import pandas as pd
import ot
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time


# from src.model_module.data_processing_for_training import calculate_errors
# from src.model_module.data_processing_for_training import train_test_NN


def convert_back_pdf(quantiles_array: np.ndarray) -> np.ndarray:
    """Convert quantiles back to pdf

    Args:
        quantiles_array (np.ndarray): Array with quantiles

    Returns:
        pdf_array (np.ndarray): Array with pdf values

    """
    pdf_array = np.diff(quantiles_array, axis=1)
    # keep the first column of quantiles and stack the pdf values
    pdf_array = np.hstack([quantiles_array[:, :1], pdf_array])

    return pdf_array


def calculate_wasserstein_distance(
        target_true_bins: np.ndarray,
        target_predicted_bins: np.ndarray,
        pdf_array: np.ndarray
) -> np.ndarray:
    """Calculate the Wasserstein distance between two pdfs

    Args:
        target_true_bins (np.ndarray): True bins positions
        target_predicted_bins (np.ndarray): Predicted bins positions
        pdf_array (np.ndarray): Array with pdf values, either true or predicted

    Returns:
        wasserstein_distance_matrix (np.ndarray): Wasserstein distance between the two pdfs
    """
    # calculate the cost matrix
    cost_matrix = ot.dist(target_true_bins, target_predicted_bins)
    # calculate the wasserstein distance
    wasserstein_distance_matrix = ot.emd2(pdf_array, pdf_array, cost_matrix)

    return wasserstein_distance_matrix


def get_column_name_id(data: pd.DataFrame):
    """Get the dictionary with the column name and the corresponding column ID

    Args:
        data (pd.DataFrame): Dataframe with the cleaned data

    Returns:
        dict: Dictionary with the column name and the corresponding column ID
    """
    dict_name_id = {column_name: column_id for column_id, column_name in enumerate(data.columns)}

    return dict_name_id


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
    # get the concentration column
    # TODO: increase the flexibility here, add params in yaml
    c_id = dict_name_id["c"]
    c_rmse = mean_squared_error(y_actual[:, c_id], y_predicted[:, c_id], squared=False)

    # get the num of crystals column
    num_crystals_id = dict_name_id["mu_0"]
    num_crystals_rmse = mean_squared_error(
        y_actual[:, num_crystals_id],
        y_predicted[:, num_crystals_id],
        squared=False
    )

    # get the matrix of bins
    bins_ids = [dict_name_id[k] if k.startswith("target_value") else None for k in dict_name_id.keys()]
    bins_ids = [b for b in bins_ids if b is not None]
    bins_actual = y_actual[:, bins_ids]
    bins_predicted = y_predicted[:, bins_ids]

    # calculate the wasserstein distance between the two pdfs
    wasserstein_distance_matrix = calculate_wasserstein_distance(
        target_true_bins=bins_actual,
        target_predicted_bins=bins_predicted,
        pdf_array=pdf_array
    )

    # TODO: add other error metrics like mu1, mu2,...

    # get the total error
    error_metrics = {
        "c_rmse": c_rmse,
        "num_crystals_rmse": num_crystals_rmse,
        "wasserstein_distance_matrix": wasserstein_distance_matrix
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
    # load quantiles
    quantiles = pd.read_csv("../../data/data_cleaning/data_cleaned_quantiles.csv")
    quantiles_array = quantiles.to_numpy()
    quantiles_diff = np.diff(quantiles_array, axis=1)
    a = np.hstack([quantiles_array[:, :1], quantiles_diff])
    b = 0
