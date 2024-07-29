"""This module prepares the utility functions."""

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d


def convert_back_pdf(
        quantiles_array: np.ndarray,
        dl_array: np.ndarray
) -> np.ndarray:
    """Convert quantiles back to pdf

    Args:
        quantiles_array (np.ndarray): Array with quantiles
        dl_array (float): Array with different intervals

    Returns:
        pdf_array_scaled (np.ndarray): Array with scaled pdf values

    """
    # calculate the difference
    pdf_array = np.diff(quantiles_array, axis=1)
    # consider the interval
    pdf_array_scaled = pdf_array / dl_array
    # # extrapolate the first value
    # first_elements = []
    # for row in range(pdf_array_scaled.shape[0]):
    #     extrapolate_fct = interp1d(
    #         quantiles_array[row, 1:],
    #         pdf_array_scaled[row],
    #         fill_value='extrapolate',
    #         bounds_error=False
    #     )
    #     first_inserted_element = extrapolate_fct(quantiles_array[row, 0])
    #     if first_inserted_element < 0:
    #         first_inserted_element = 0
    #     first_elements.append(first_inserted_element)
    #
    # first_elements_array = np.array(first_elements).reshape(-1, 1)
    # pdf_array_scaled = np.hstack((first_elements_array, pdf_array_scaled))

    return pdf_array_scaled


def get_column_name_id(data: pd.DataFrame):
    """Get the dictionary with the column name and the corresponding column ID

    Args:
        data (pd.DataFrame): Dataframe with the cleaned data

    Returns:
        dict: Dictionary with the column name and the corresponding column ID
    """
    dict_name_id = {column_name: column_id for column_id, column_name in enumerate(data.columns)}

    return dict_name_id


def check_duplicates(arr: np.ndarray):
    """Check for duplicates in the array

    Args:
        arr (np.ndarray): Array to check for duplicates

    Returns:
        alarm (bool): True if duplicates are found, False otherwise
    """
    unique_elements, counts = np.unique(arr, return_counts=True)
    duplicates = unique_elements[counts > 1]
    if len(duplicates) > 1:  # the last few elements should be 1
        alarm = True
        print(f"Array contains duplicates: {duplicates}")
    else:
        alarm = False

    return alarm


def handle_inf_values(y_values: np.ndarray, x_values: np.ndarray):
    """Check and handle infinite values in the data

    Args:
        y_values (np.ndarray): Data to check for infinite values
        x_values (np.ndarray): Data to help handle infinite values

    """
    if np.isinf(y_values).any():
        print("Infinite values found in the data")
        # locate the infinite values
        inf_indices = np.where(np.isinf(y_values))
        inf_rows = inf_indices[0]
        for row in inf_rows:
            inf_cols = inf_indices[1][inf_indices[0] == row]
            non_inf_cols = ~np.isinf(y_values[row])
            # interpolate the infinite values with interpolated values
            # less accurate but feasible
            interp_fct = interp1d(
                x_values[row, non_inf_cols],
                y_values[row, non_inf_cols],
                fill_value='extrapolate',
            )
            y_values[row, inf_cols] = interp_fct(x_values[row, inf_cols])
    else:
        pass

    return y_values





    return
