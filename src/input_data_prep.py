"""This file prepares the input features to the model."""

import pandas as pd
import numpy as np
import yaml


def transform_pdf_to_cdf(prob_values: list):
    """Transform probability density function to cumulative density function

    Args:
        prob_values (list): list of probabilities

    Returns:
         quantile_values (list): list of quantile values in cdf
    """
    quantiles = np.cumsum(prob_values)
    return quantiles


def get_bin_ranges(prob_values: list, bin_ticks: list):
    """ Get the ranges of bin values of which the probabilities are not zero
    This function is useful for the case where the bin values are different
    for different distributions.

    Args:
        prob_values (list): list of probabilities
        bin_ticks (list): list of bin values as the x-tick in pdf

    Returns:
        bin_range (dict): the upper and lower bounds of non-zero bin values
    """
    # Get the locations of first and last non-zero bin values in the pdf
    non_zero_prob = np.where(np.array(prob_values) != 0)
    print(non_zero_prob)
    first_non_zero = non_zero_prob[0][0]
    last_non_zero = non_zero_prob[0][-1]

    # Get the bin values of the first and last non-zero bin values
    bin_range = {
        "lower": bin_ticks[first_non_zero],
        "upper": bin_ticks[last_non_zero]
    }
    # Todo: consider the matrix case
    return bin_range


def sample_quantiles(
        bin_values: list,
        bin_ticks: list,
        bin_range: dict,
        quantiles: list
) -> list:
    """Sample the quantiles from the cdf based on the user-defined bin values

    Args:
        bin_values (list): list of selected bin values
        bin_ticks (dict): the bin values as the x-tick in pdf
        bin_range (dict): the upper and lower bounds of non-zero bin values
        quantiles (list): list of quantile values in cdf

    Returns:
        sampled_quantiles (list): list of sampled quantile values

    """
    # Get the indices of the bin values in the bin ticks
    bin_indices = []
    for bin_val in bin_values:
        bin_indices.append(bin_ticks.index(bin_val))
    # Get the corresponding quantile values
    sampled_quantiles = []
    for idx in bin_indices:
        sampled_quantiles.append(quantiles[idx])

    return sampled_quantiles


def get_target_bins(
        sampled_quantiles: list,
        target_quantiles: list,
        target_bin_ticks: list):
    """Get the target bin values based on the sampled quantiles

    Args:
        sampled_quantiles (list): list of sampled quantile values
        target_quantiles (list): list of target quantile values
        target_bin_ticks (list): list of target bin values as the x-tick in pdf

    Returns:
        target_bins (list): list of target bin values
    """
    target_bins = []
    for quantile in sampled_quantiles:
        # method 1: approximate to the nearest target quantile
        # target_bins.append(target_bin_ticks[np.argmin(np.abs(np.array(target_quantiles) - quantile))])
        # method 2: interpolate the target bin values
        target_bins.append(np.interp(quantile, target_quantiles, target_bin_ticks))
    return target_bins


def select_columns(df: pd.DataFrame, columns: list):
    """Select columns from a dataframe

    Args:
        df (pd.DataFrame): input dataframe
        columns (list): list of column names to be selected

    Returns:
        df (pd.DataFrame): dataframe with selected columns
    """
    return df[columns]


def clean_data(file_path, exp_name):
    """Loads raw simulated data

    Args:
        file_path (str): Path of raw data
        exp_name (str): Name under which training data was saved

    Returns:
        Tuple: Input matrix for PBE solver, PBE solver results
    """
    # Input
    input_mat = pd.read_csv(f"{file_path}/PBEsolver_InputMatrix/{exp_name}.csv")
    print("Input matrix shape: ", input_mat.shape)
    print(type(input_mat))

    # Output
    results = {}
    for runID in input_mat["runID"]:
        try:
            results[runID] = pd.read_csv(f"{file_path}/PBEsolver_outputs/PBEsolver_{exp_name}_runID{int(runID)}.csv")
        except:
            pass
    print("PBE output files found: ", len(results))
    return input_mat, results


if __name__ == '__main__':
    # Todo: read yaml file
    with open("../params/data_params.yaml", "r", encoding="utf-8") as params:
        params = yaml.safe_load(params)
    file_path = 'D:/PycharmProjects/surrogatepbe'
    exp_name = 'InputMat_231207_1605'
    # X, y = load_raw_data(file_path=file_path, exp_name=exp_name)
    prob = [0, 0, 0.1, 0.2, 0.3, 0.4, 0]
    bins = [1, 2, 3, 4, 5, 6, 7]
    bin_ranges = get_bin_ranges(prob, bins)
    print(bin_ranges)
    quantiles = transform_pdf_to_cdf(prob)
    print(quantiles)
    bin_values = [2, 4, 6]
    sampled_quantiles = sample_quantiles(bin_values, bins, bin_ranges, quantiles)
    print(sampled_quantiles)

    clean_data(file_path, exp_name)
