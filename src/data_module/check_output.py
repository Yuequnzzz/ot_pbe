"""This module checks the quality of cleaned data, as currently it failed to depict the original data"""

from src.data_module.input_data_prep import *
from src.model_module.trainer import convert_back_pdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # import input data
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
        upper_limit,
        lower_limit,
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
        data_params["upper_bound"],
        data_params["lower_bound"],
        data_params["sample_fraction"],
        data_params["save_path"],
        data_params["save_name"],
    )

    bin_feature = {
        'bin_start': bin_start,
        'dL': dL
    }

    # TODO: MODIFY
    num_sample = 80
    runID = 0

    input_df = pd.read_csv(f"{file_path}/PBEsolver_InputMatrix/{exp_name}.csv")
    print("Input matrix shape: \n", input_df.shape)

    output_df = pd.read_csv(f"{file_path}/PBEsolver_outputs/PBEsolver_{exp_name}_runID{int(runID)}.csv")

    # sample the timepoints to avoid overfitting
    output_df = output_df.sample(frac=sample_frac, random_state=42)
    no_timepoints = output_df.shape[0]
    # get the repeated relevant inputs
    relevant_inputs = input_df.query("runID == @runID")
    repeated_inputs_df = pd.concat([relevant_inputs] * no_timepoints, ignore_index=True)

    # get the input features
    input_features_df, sampled_quantiles, sampled_input_intervals = prep_feature_table(
        df=repeated_inputs_df,
        feature_columns=input_columns,
        num_sample=num_sample,
        bin_feature=bin_feature,
        quantile_lower_bound=lower_limit,
        quantile_upper_bound=upper_limit,
    )
    # # append the quantiles
    # quantiles_all.append(sampled_quantiles)
    # add time to the input features
    t_vec = np.array(output_df["t"])[..., np.newaxis]
    input_features_df['t'] = t_vec
    # drop runID
    input_features_df = input_features_df.drop(columns=['runID'])

    if input_features_df.isnull().values.any():
        raise ValueError(f"There is empty values in {runID} input")

    # input_df_list.append(input_features_df)

    # get the target features
    target_features, target_intervals = prep_target_table(
        df=output_df,
        target_columns=output_columns,
        sampled_quantiles_array=sampled_quantiles,
        bin_feature=bin_feature
    )
    # output_df_list.append(target_features)
    # interval_list.append(target_intervals)

    # -----------------------------

    # convert back to pdf
    input_pdfs = convert_back_pdf(sampled_quantiles, dl_array=sampled_input_intervals)
    output_pdfs = convert_back_pdf(sampled_quantiles, dl_array=target_intervals)

    # plot the input results
    input_cols = [f"inipop_bin{i}" for i in range(1000)]
    # TODO: ATTENTION, take out the first bin loc value
    source_cols = [f"source_value_{i}" for i in range(1, num_sample)]

    num_bin = 1000
    bin_start_ticks = np.arange(bin_feature['bin_start'],
                                bin_feature['bin_start'] + bin_feature['dL'] * (num_bin + 1),
                                bin_feature['dL'])
    ini_bin_ticks = (bin_start_ticks[1:] + bin_start_ticks[:-1]) / 2

    # scale the population to pdf
    input_df_prob = get_prob_df(input_df, 'inipop_bin', dL=0.5)[0]
    input_df_cumsum = np.cumsum(input_df_prob * dL, axis=1)

    plt.figure()
    plt.plot(
        input_features_df.iloc[0][source_cols],
        input_pdfs[0],
        label='Fitted'
    )
    plt.plot(
        ini_bin_ticks,
        input_df_prob.iloc[0][input_cols],
        label='Initial'
    )
    plt.legend()
    plt.show()

    # plot the output results
    output_cols = [f"pop_bin{i}" for i in range(1000)]
    # TODO: ATTENTION, take out the first bin loc value
    target_cols = [f"target_value_{i}" for i in range(1, num_sample)]
    output_df_prob = get_prob_df(output_df, 'pop_bin', dL=0.5)[0]
    for row in range(target_features.shape[0]):
        plt.figure()
        plt.plot(
            target_features.iloc[row][target_cols],
            output_pdfs[row],
            label='Fitted'
        )
        plt.plot(
            ini_bin_ticks,
            output_df_prob.iloc[row][output_cols],
            label='Ini_Output'
        )
        plt.legend()
        plt.show()
