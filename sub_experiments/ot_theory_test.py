import numpy as np
import ot
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Generate synthetic data for demonstration
np.random.seed(0)
num_samples = 100
num_features = 1

# Generate source distribution (X) and target distribution (Y)
X = np.random.normal(loc=0, scale=1, size=(num_samples, num_features))
Y = X + np.random.normal(loc=0, scale=0.5, size=(num_samples, num_features))

# Compute pairwise Euclidean distances between points in X and Y
cost_matrix = ot.dist(X, Y)

# Define initial distribution of source and target data
a = np.ones(num_samples) / num_samples  # Uniform weights for source distribution
b = np.ones(num_samples) / num_samples  # Uniform weights for target distribution

# Compute optimal transport plan between source and target distributions
transport_plan = ot.emd(a, b, cost_matrix)

# Calculate sample weights based on the transportation plan
sample_weights = np.sum(transport_plan, axis=1)  # Use row sums as sample weights

# Perform distribution-to-distribution regression
# Use linear regression as an example, but other models can be used as well
regressor = LinearRegression()
regressor.fit(X, Y, sample_weight=sample_weights)
# regressor.fit(X, Y)

# Predict target distribution based on source distribution
Y_pred = regressor.predict(X)

# Plot the predicted Y against the actual Y
plt.figure(figsize=(8, 6))
plt.scatter(Y, Y_pred, color='blue', label='Predicted vs Actual')
plt.plot(Y, Y, color='red', linestyle='--', label='Perfect prediction')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Predicted vs Actual Y')
plt.legend()
plt.grid(True)
plt.show()

# combine y and y_predict
df = pd.DataFrame({'Actual Y': Y.flatten(), 'Predicted Y': Y_pred.flatten()})
print(df)

# calculate mse
mse = mean_squared_error(Y, Y_pred)
print('the mean squared error is', mse)


# # Calculate Wasserstein distance between predicted target distribution and actual target distribution
# wasserstein_distance = ot.emd2(Y.flatten(), Y_pred.flatten(), cost_matrix)

# print("Wasserstein Distance between actual and predicted distributions:", wasserstein_distance)