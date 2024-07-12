"""This file prepares the input features to the model."""

import pandas as pd
import numpy as np
import yaml


def transform_pdf_to_cdf(prob_df: pd.DataFrame) -> pd.DataFrame:
    """Transform probability density function to cumulative density function

    Args:
        prob_df (pd.DataFrame): list of probabilities

    Returns:
         quantile_values (pd.DataFrame): list of quantile values in cdf
    """
    quantiles_df = np.cumsum(prob_df, axis=1)
    return quantiles_df


def get_bin_ranges(
        prob_df: pd.DataFrame,
        bin_ticks: list,
        min_prob: float = 0.0,
) -> pd.DataFrame:
    """ Get the ranges of bin values of which the probabilities are
    greater than the minimum probability. This function is useful for
    the case where the bin values are different for different distributions.

    Args:
        prob_df (pd.DataFrame): probabilities in pdf
        bin_ticks (list): list of bin values as the x-tick in pdf
        min_prob (float): the lower bound of probability to use

    Returns:
        bin_range (pd.DataFrame): upper and lower bounds of bin values for different distributions
    """
    prob_to_use = np.where(np.array(prob_df) > min_prob)

    bin_range_df = pd.DataFrame(columns=['min_bin', 'max_bin'])
    for row in np.unique(prob_to_use[0]):
        # get the index of first and last row num
        row_index = np.where(prob_to_use[0] == row)
        print('row index is', row_index)
        row_first_index = row_index[0][0]
        row_last_index = row_index[0][-1]

        column_first_index = prob_to_use[1][row_first_index]
        column_last_index = prob_to_use[1][row_last_index]

        min_bin = bin_ticks[column_first_index]
        max_bin = bin_ticks[column_last_index]

        # Assign the bin values to the dataframe
        bin_range_df.loc[row] = [min_bin, max_bin]

    return bin_range_df


def get_sampled_bins(bin_range_df: pd.DataFrame, num_sample: int) -> tuple:
    """Generate insertion points for the selected bin values

    Args:
        bin_range_df (pd.DataFrame): the upper and lower bounds of non-zero bin values
        num_sample (int): the number of inserted points within certain range

    Returns:
        x_bins (np.array): ndarray with inserted points
        source_bins_df (pd.DataFrame): dataframe with inserted points
    """
    x_bins = np.linspace(bin_range_df['min_bin'], bin_range_df['max_bin'], num=num_sample)
    x_bins = np.transpose(x_bins)
    source_bins_df = pd.DataFrame(x_bins, columns=[f'source_value_{n}' for n in range(num_sample)])

    return x_bins, source_bins_df


def get_sampled_quantiles(
        num_sample: int,
        x_bins: np.array,
        bin_ticks: list,
        quantiles_df: pd.DataFrame,
) -> np.array:
    """Sample the quantiles from the cdf based on the user-defined bin values

    Args:
        num_sample (int): the number of inserted points within certain range
        x_bins (np.array): list of selected bin values
        bin_ticks (list): the bin values as the x-tick in pdf
        quantiles_df (pd.DataFrame): quantile values

    Returns:
        sampled_quantiles (list): list of sampled quantile values

    """
    # interpolate to get the sampled quantiles
    origin_bins_array = np.tile(bin_ticks, (x_bins.shape[0], 1))
    sampled_quantiles_array = np.zeros(x_bins.shape)

    for row_num in range(origin_bins_array.shape[0]):
        sampled_quantiles_array[row_num, :] = np.interp(
            x_bins[row_num], origin_bins_array[row_num],
            quantiles_df.iloc[row_num].values
        )

    return sampled_quantiles_array


def get_target_bins(
        sampled_quantiles_array: np.array,
        bin_ticks: list,
        target_quantiles_df: pd.DataFrame,
        target_bin_ticks: list
) -> tuple:
    """Get the target bin values based on the sampled quantiles

    Args:
        sampled_quantiles_array (np.array): sampled quantile values
        bin_ticks (list): list of bin values as the x-tick in pdf
        target_quantiles_df (pd.DataFrame): list of target quantile values

    Returns:
        target_mapped_bins (np.array): mapped target bin values
        target_bins_df (pd.DataFrame): dataframe with mapped target bin values
    """
    # get the mapped bin values in target distribution
    origin_bins_array = np.tile(bin_ticks, (sampled_quantiles_array.shape[0], 1))
    target_mapped_bins = np.zeros(sampled_quantiles_array.shape)

    for row_num in range(target_mapped_bins.shape[0]):
        target_mapped_bins[row_num, :] = np.interp(sampled_quantiles_array[row_num],
                                                   target_quantiles_df.iloc[row_num].values,
                                                   origin_bins_array[row_num])

    target_bins_df = pd.DataFrame(
        target_mapped_bins,
        columns=[f'target_value_{i}' for i in range(sampled_quantiles_array.shape[1])]
    )

    return target_mapped_bins, target_bins_df


def select_columns(
        df: pd.DataFrame, columns: list
) -> pd.DataFrame:
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
