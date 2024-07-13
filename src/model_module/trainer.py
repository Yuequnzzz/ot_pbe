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


def convert_back_pdf(quantiles_array: np.ndarray):
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
):
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
