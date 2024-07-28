"""This file contains the function to predict the transport cost of a shipment"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import ot


# Generate synthetic data for demonstration
np.random.seed(0)
num_warehouses = 2
num_stores = 3
num_features = 2

# Generate random initial distributions for warehouses
initial_distributions = np.random.randint(10, 30, size=(num_warehouses, num_features))

# Generate random features representing operational conditions
operation_conditions = np.random.rand(num_warehouses, num_features)

# Generate random demand at each store
demand = np.random.rand(num_stores)

# Calculate transportation costs based on Wasserstein distance between initial distributions and demand
transportation_costs = np.zeros((num_warehouses, num_stores))
for i in range(num_warehouses):
    for j in range(num_stores):
        transportation_costs[i, j] = ot.emd2(initial_distributions[i], demand, np.eye(num_features))

# Define initial distribution of goods in warehouses (use synthetic data for demonstration)
warehouse_distributions = initial_distributions

# Compute optimal transport plan
# Define cost matrix using calculated transportation costs
cost_matrix = transportation_costs

# Compute optimal transport plan
transport_plan = ot.emd(np.sum(warehouse_distributions, axis=1), demand, cost_matrix)

# Compute final distribution based on the optimal transport plan
final_distribution = np.dot(transport_plan, demand)

print("Transportation costs based on Wasserstein distance:")
print(transportation_costs)
print("Final distribution of goods at the stores:")
print(final_distribution)

