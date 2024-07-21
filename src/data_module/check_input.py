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

    input_df = pd.read_csv(f"{file_path}/PBEsolver_InputMatrix/{exp_name}.csv")
    print("Input matrix shape: \n", input_df.shape)

    print("Original input matrix is:\n", input_df)

    input_fe, sampled_quantile, sampled_intervals = prep_feature_table(
        df=input_df,
        feature_columns=input_columns,
        num_sample=num_sample,
        bin_feature=bin_feature,
        quantile_lower_bound=lower_limit,
        quantile_upper_bound=upper_limit,
    )
    print("Featured input is:", input_fe)

    # convert back to pdf
    pdf_values_array = convert_back_pdf(sampled_quantile, dl_array=sampled_intervals)
    fitted_pdf_cumsum = np.cumsum(pdf_values_array*sampled_intervals, axis=1)

    # get the num of inputs
    num_inputs = input_fe.shape[0]

    # plot the results
    plot_cols = [f"source_value_{i}" for i in range(num_sample)]
    ini_cols = [f"inipop_bin{i}" for i in range(1000)]

    num_bin = 1000
    bin_start_ticks = np.arange(bin_feature['bin_start'],
                                bin_feature['bin_start'] + bin_feature['dL'] * (num_bin + 1),
                                bin_feature['dL'])
    ini_bin_ticks = (bin_start_ticks[1:] + bin_start_ticks[:-1]) / 2

    # scale the population to pdf
    input_df_prob = get_prob_df(input_df, 'inipop_bin', dL=0.5)[0]
    input_df_cumsum = np.cumsum(input_df_prob*dL, axis=1)

    for row in range(num_inputs):
        plt.plot(
            input_fe.iloc[row][plot_cols],
            pdf_values_array[row],
            label='Fitted'
        )
        plt.plot(
            ini_bin_ticks,
            input_df_prob.iloc[row][ini_cols],
            label="Initial"
        )
        plt.title(f"Plot {row+1}")
        plt.legend()
        # plt.show()

        # save the plot
        plt.savefig(f"plot_{row + 1}.png")  # Save each plot with a unique filename
        plt.close()


