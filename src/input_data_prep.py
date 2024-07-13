"""This file prepares the input features to the model."""
import logging

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
        bin_ticks: np.ndarray,
        min_prob: float = 0.0,
) -> pd.DataFrame:
    """ Get the ranges of bin values of which the probabilities are
    greater than the minimum probability. This function is useful for
    the case where the bin values are different for different distributions.

    Args:
        prob_df (pd.DataFrame): probabilities in pdf
        bin_ticks (np.ndarray): bin values as the x-tick in pdf
        min_prob (float): the lower bound of probability to use

    Returns:
        bin_range (pd.DataFrame): upper and lower bounds of bin values for different distributions
    """
    prob_to_use = np.where(np.array(prob_df) > min_prob)

    bin_range_df = pd.DataFrame(columns=['min_bin', 'max_bin'])
    for row in np.unique(prob_to_use[0]):
        # get the index of first and last row num
        row_index = np.where(prob_to_use[0] == row)
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
        x_bins: np.array,
        bin_ticks: np.ndarray,
        quantiles_df: pd.DataFrame,
) -> np.array:
    """Sample the quantiles from the cdf based on the user-defined bin values

    Args:
        x_bins (np.array): list of selected bin values
        bin_ticks (np.ndarray): the bin values as the x-tick in pdf
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
        bin_ticks: np.ndarray,
        target_quantiles_df: pd.DataFrame,
) -> tuple:
    """Get the target bin values based on the sampled quantiles

    Args:
        sampled_quantiles_array (np.array): sampled quantile values
        bin_ticks (np.ndarray): bin values as the x-tick in pdf
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


def get_prob_df(
        df: pd.DataFrame,
        prefix: str
) -> pd.DataFrame:
    """Get the probability density function from the dataframe

    Args:
        df (pd.DataFrame): input dataframe
        prefix (str): the prefix of columns to be selected

    Returns:
        prob_df (pd.DataFrame): dataframe with selected columns
    """
    prob_df = df.filter(like=prefix)

    return prob_df


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


def prep_feature_table(
        df: pd.DataFrame,
        feature_columns: list,
        num_sample: int,
        bin_feature: dict
) -> tuple:
    """Prepare the input features for certain dataframe containing raw data

    Args:
        df (pd.DataFrame): input dataframe
        feature_columns (list): list of column names to be selected
        num_sample (int): the number of inserted points within certain range
        bin_feature (dict): include bin_start and the interval length dL

    Returns:
        input_feature_df (pd.DataFrame): input feature dataframe
        sampled_quantiles_array (np.array): sampled quantile values
    """
    # get the probability dataframe
    prob_df = get_prob_df(df, 'pop_bin')
    num_bin = prob_df.shape[1]
    # check if there is null value
    if prob_df.isnull().values.any():
        print(f"Found NaN in prob df")
    # get the quantile dataframe
    quantiles_df = transform_pdf_to_cdf(prob_df)
    if quantiles_df.isnull().values.any():
        print(f"Found NaN in quantile df")

    # get the bin ticks
    bin_start_ticks = np.arange(bin_feature['bin_start'],
                          bin_feature['bin_start']+bin_feature['dL']*(num_bin+1),
                          bin_feature['dL'])
    bin_ticks = (bin_start_ticks[1:] + bin_start_ticks[:-1]) / 2

    # get the bin ranges
    bin_range_df = get_bin_ranges(prob_df, bin_ticks)
    if bin_range_df.isnull().values.any():
        print(f"Found NaN in bin_range df")
    # get the sampled bins
    x_bins, source_bins_df = get_sampled_bins(bin_range_df, num_sample)
    if source_bins_df.isnull().values.any():
        print(f"Found NaN in source bin df")
    # get the sampled quantiles
    sampled_quantiles_array = get_sampled_quantiles(
        x_bins=x_bins,
        bin_ticks=bin_ticks,
        quantiles_df=quantiles_df)
    # get the other features
    other_features = select_columns(df, feature_columns).reset_index(drop=True)
    # combine the other features with the sampled bins
    input_feature_df = pd.concat([other_features, source_bins_df], axis=1)
    if input_feature_df.isnull().values.any():
        print(f"Found NaN in input feature df")

    return input_feature_df, sampled_quantiles_array


def prep_target_table(
        df: pd.DataFrame,
        target_columns: list,
        sampled_quantiles_array: np.array,
        bin_feature: dict
) -> pd.DataFrame:
    """Prepare the target features for certain dataframe containing raw data

    Args:
        df (pd.DataFrame): input dataframe
        target_columns (list): list of column names to be selected
        sampled_quantiles_array (np.array): sampled quantile values
        bin_feature (dict): include bin_start and the interval length dL

    Returns:
        target_feature_df (pd.DataFrame): target feature dataframe

    """
    # get the target probability dataframe
    target_prob_df = get_prob_df(df, 'pop_bin')
    num_bin = target_prob_df.shape[1]
    # get the target quantile dataframe
    target_quantiles_df = transform_pdf_to_cdf(target_prob_df)
    # get the bin ticks
    bin_start_ticks = np.arange(bin_feature['bin_start'],
                                bin_feature['bin_start']+bin_feature['dL']*(num_bin+1),
                                bin_feature['dL'])
    bin_ticks = (bin_start_ticks[1:] + bin_start_ticks[:-1]) / 2
    # get the target bins
    target_mapped_bins, target_bins_df = get_target_bins(
        sampled_quantiles_array=sampled_quantiles_array,
        bin_ticks=bin_ticks,
        target_quantiles_df=target_quantiles_df
    )
    # get the target features
    target_features = select_columns(df, target_columns).reset_index(drop=True)
    # combine the target features with the target bins
    target_feature_df = pd.concat([target_features, target_bins_df], axis=1)

    return target_feature_df


def prep_data_for_model(
        file_path: str,
        exp_name: str,
        num_sample: int,
        source_columns: list,
        target_columns: list,
        bin_feature: dict,
        sample_frac: float,
) -> tuple:
    """Prepare the feature tables for the model

    Args:
        file_path (str): Path of raw data
        exp_name (str): Name under which training data was saved
        num_sample (int): the number of inserted points within certain range
        source_columns (list): list of input column names to be selected
        target_columns (list): list of output column names to be selected
        bin_feature (dict): include bin_start and the interval length dL
        sample_frac (float): the fraction of timepoints in each simulation to sample

    Returns:
        input_df (pd.DataFrame): input feature dataframe
        output_df (pd.DataFrame): target feature dataframe
        input_datasets (dict): dictionary containing input and output arrays
    """
    input_mat = pd.read_csv(f"{file_path}/PBEsolver_InputMatrix/{exp_name}.csv")
    print("Input matrix shape: ", input_mat.shape)

    input_df_list = []
    output_df_list = []
    for runID in input_mat["runID"]:
        try:
            print("Iteration: ", runID)
            # load single simulation output
            target_sub_df = pd.read_csv(f"{file_path}/PBEsolver_outputs/PBEsolver_{exp_name}_runID{int(runID)}.csv")
            # sample the timepoints to avoid overfitting
            target_sub_df = target_sub_df.sample(frac=sample_frac)
            # check if there is Nan
            if target_sub_df.isnull().values.any():
                print(f"Found NaN in target runID{int(runID)}")
            no_timepoints = target_sub_df.shape[0]
            # get the repeated relevant inputs
            relevant_inputs = input_mat.query("runID == @runID")
            repeated_inputs_df = pd.concat([relevant_inputs] * no_timepoints, ignore_index=True)
            # check if there is Nan
            if repeated_inputs_df.isnull().values.any():
                print(f"Found NaN in repeated inputs runID{int(runID)}")
                continue

            # get the input features
            input_features_df, sampled_quantiles = prep_feature_table(
                df=repeated_inputs_df,
                feature_columns=source_columns,
                num_sample=num_sample,
                bin_feature=bin_feature,
            )
            # add time to the input features
            t_vec = np.array(target_sub_df["t"])[..., np.newaxis]
            input_features_df['t'] = t_vec
            # drop runID
            input_features_df = input_features_df.drop(columns=['runID'])

            if input_features_df.isnull().values.any():
                raise ValueError(f"There is empty values in {runID}")

            input_df_list.append(input_features_df)
            print(len(input_df_list))

            # get the target features
            target_features = prep_target_table(
                df=target_sub_df,
                target_columns=target_columns,
                sampled_quantiles_array=sampled_quantiles,
                bin_feature=bin_feature
            )
            output_df_list.append(target_features)

        except Exception as e:  # TODO: check
            print(e)
            raise ValueError("bug!!")

    input_df = pd.concat(input_df_list, axis=0, ignore_index=True)
    output_df = pd.concat(output_df_list, axis=0, ignore_index=True)

    if input_df.isnull().values.any():
        raise ValueError(f"There is empty values in input_df")
    if output_df.isnull().values.any():
        raise ValueError(f"There is empty values in output_df")

    print("Data preparation completed.")
    # TODO: Set the logging

    input_datasets = {
        'X': np.array(input_df),
        'y': np.array(output_df)
    }

    return input_df, output_df, input_datasets


if __name__ == '__main__':
    with open("../params/data_params.yaml", "r", encoding="utf-8") as params:
        data_params = yaml.safe_load(params)
    (
        file_path,
        exp_name,
        num_sample,
        input_columns,
        output_columns,
        bin_start,
        dL,
        sample_frac,
        save_path,
        save_name,
    ) = (
        data_params["file_path"],
        data_params["experiment_name"],
        data_params["num_sample"],
        data_params["input_columns"],
        data_params["output_columns"],
        data_params["bin_start"],
        data_params["dL"],
        data_params["sample_fraction"],
        data_params["save_path"],
        data_params["save_name"],
    )

    bin_feature = {
        'bin_start': bin_start,
        'dL': dL
    }

    input_df, output_df, input_datasets = prep_data_for_model(
        file_path=file_path,
        exp_name=exp_name,
        num_sample=num_sample,
        source_columns=input_columns,
        target_columns=output_columns,
        bin_feature=bin_feature,
        sample_frac=sample_frac
    )
    print(input_df.head())
    print(output_df.head())

    # Save the input and output data
    input_df.to_csv(f"{save_path}/{save_name}_input.csv", index=False)
    output_df.to_csv(f"{save_path}/{save_name}_output.csv", index=False)
