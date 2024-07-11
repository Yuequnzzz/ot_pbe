import pandas as pd
import numpy as np

# Define the columns
columns = ['time', 'growth rate', 'nucleation rate'] + [f'pop_bin{i}' for i in range(7)]

# Generate random data for three rows
data = {
    'time': np.random.uniform(0, 10, 3),  # random times between 0 and 10
    'growth rate': np.random.uniform(0, 1, 3),  # random growth rates between 0 and 1
    'nucleation rate': np.random.uniform(0, 1, 3),  # random nucleation rates between 0 and 1
}

# Initialize bins
bins = np.random.rand(3, 7)

# Normalize bins so that each row sums to 1
bins = bins / bins.sum(axis=1, keepdims=True)

# Add normalized bins to data dictionary
for i in range(7):
    data[f'pop_bin{i}'] = bins[:, i]

# Create the DataFrame
df = pd.DataFrame(data, columns=columns)

# Display the DataFrame
print(df)

--------------------------

# Select columns with prefix 'pop_bin'
pop_bin_cols = df.filter(like='pop_bin')
pop_bin_cols

quantile_matrix = np.cumsum(pop_bin_cols, axis=1)
quantile_matrix

# for the first case: different selected bin values
# 1. specify the num of inserted points within certain range
prob_to_use = np.where(np.array(pop_bin_cols) > 0)

bin_ticks_test = [1, 2, 3, 4, 5, 6, 7]
for row in np.unique(prob_to_use[0]):
    # get the index of first and last row num
    row_index = np.where(prob_to_use[0]==row)
    print('row index is', row_index)
    row_first_index = row_index[0][0]
    row_last_index = row_index[0][-1]

    column_first_index = prob_to_use[1][row_first_index]
    column_last_index = prob_to_use[1][row_last_index]

    min_bin = bin_ticks_test[column_first_index]
    max_bin = bin_ticks_test[column_last_index]

    pop_bin_cols['min_bin'] = min_bin
    pop_bin_cols['max_bin'] = max_bin

print(pop_bin_cols)

-----------------------------

# generate insertion points
num_sample = 50
x_bins = np.linspace(pop_bin_cols['min_bin'], pop_bin_cols['max_bin'], num=num_sample)
print(x_bins.shape)

# transpose the matrix
x_bins = np.transpose(x_bins)
# transform into df
new_feature_bins = pd.DataFrame(x_bins, columns=[f'source_value_{n}' for n in range(num_sample)])
new_feature_bins

# interpolate to get the sampled quantiles
num_sample = 50
origin_bins_array = np.tile(bin_ticks_test, (3, 1))
sample_quantiles = np.zeros((3, num_sample))

for row_num in range(origin_bins_array.shape[0]):

    sample_quantiles[row_num, :] = np.interp(x_bins[row_num], origin_bins_array[row_num], quantile_matrix.iloc[row_num].values)
sample_quantiles

----------------------------
# generate another distribution as the target dist
import pandas as pd
import numpy as np

# Define the columns
columns = ['time', 'growth rate', 'nucleation rate'] + [f'pop_bin{i}' for i in range(7)]

# Generate random data for three rows
data_target = {
    'time': np.random.uniform(0, 10, 3),  # random times between 0 and 10
    'growth rate': np.random.uniform(0, 1, 3),  # random growth rates between 0 and 1
    'nucleation rate': np.random.uniform(0, 1, 3),  # random nucleation rates between 0 and 1
}

# Initialize bins
bins_target = np.random.rand(3, 7)

# Normalize bins so that each row sums to 1
bins_target = bins_target / bins_target.sum(axis=1, keepdims=True)

# Add normalized bins to data dictionary
for i in range(7):
    data_target[f'pop_bin{i}'] = bins_target[:, i]

# Create the DataFrame
df_target = pd.DataFrame(data_target, columns=columns)

# Display the DataFrame
df_target


---------------------------
# select pop_bin parts and calculate the cumulative sum
target_bin_cols = df_target.filter(like='pop_bin') 
target_bin_cols

target_origin_quantiles = np.cumsum(target_bin_cols, axis=1)
target_origin_quantiles

-----------------------------
# get the mapped bin values in target distribution
num_sample = 50
origin_bins_array = np.tile(bin_ticks_test, (3, 1))
target_mapped_bins = np.zeros((3, num_sample))

for row_num in range(target_mapped_bins.shape[0]):

    target_mapped_bins[row_num, :] = np.interp(sample_quantiles[row_num], target_origin_quantiles.iloc[row_num].values, origin_bins_array[row_num])

target_mapped_bins

--------------------------------
# transform to df
target_feature_bins = pd.DataFrame(target_mapped_bins, columns=[f'target_value_{i}' for i in range(num_sample)])
target_feature_bins