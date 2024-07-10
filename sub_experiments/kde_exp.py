import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate some sample data
np.random.seed(0)
data = np.random.normal(loc=5, scale=2, size=100)

# Step 2: KDE Estimation
# Use scipy's gaussian_kde function
kde = gaussian_kde(data)

# Evaluate the density function at a grid of points
x_grid = np.linspace(min(data) - 1, max(data) + 1, 1000)
density = kde(x_grid)

# Plot the original data histogram and KDE
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=False, stat="density", label='Histogram')
plt.plot(x_grid, density, label='KDE', linewidth=2)
plt.xlabel('Data values')
plt.ylabel('Density')
plt.legend()
plt.title('Kernel Density Estimation')
plt.show()


# some comments:
# 1. kernel density estimation is the first step. We can sort out the logic/background like follows:
# in PBE problem, we start from a density function, sample from it, which is the step one here. But
# the difference is that, before we  keep record of the density at certain location directly, rather than
# the samples sampled here. So we should go to sample first 

# 2. the logic of kde is to find the pattern of the samples, generate distribution, and give distribution values 
# if given data locations