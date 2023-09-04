import numpy as np
import matplotlib.pyplot as plt

import glob
import os
# Change the working directory to the current file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)

file_number = 20

data = np.empty((file_number, 1520, 1520))

file_list = glob.glob(os.path.join(".", "*.txt"))
for i, file_path in enumerate(file_list[:file_number]):
    arr_2d = np.loadtxt(file_path)
    data[i] = arr_2d

# Define a function to count the number of points within a given radius
def points_within_radius(distances, radius):
    return np.sum(distances <= radius)

dim_fitting_result = np.empty(file_number)
constant_fitting_result = np.empty(file_number)
all_distances = np.array([])

for i in range(file_number):
    # Filter out numbers > 0
    positive_data = np.array([(x, y) for x in range(data[i].shape[0]) for y in range(data[i].shape[1]) if data[i][x, y] > 0])

    # Find the center of the array
    center = np.array([data[i].shape[0] // 2, data[i].shape[1] // 2])

    # Calculate the distances from the center
    distances = np.sqrt(np.sum((positive_data - center)**2, axis=1))
    all_distances = np.append(all_distances, distances)

    # Calculate the density distribution function
    max_radius = int(np.ceil(np.max(distances)))
    density_distribution = [points_within_radius(distances, r) for r in range(max_radius+1-50)]

    log_radius = np.log(np.arange(max_radius+1-50))
    log_density_distribution = np.log(density_distribution)

    nonzero_indices = (log_radius != 0) & (log_density_distribution != 0)
    log_radius_nonzero = log_radius[nonzero_indices]
    log_density_distribution_nonzero = log_density_distribution[nonzero_indices]

    # Fit a linear regression to the data
    slope, intercept = np.polyfit(log_radius_nonzero, log_density_distribution_nonzero, 1)

    dim_fitting_result[i] = slope
    constant_fitting_result[i] = intercept

    '''
    # Calculate the linear regression line values
    linear_regression = slope * log_radius + intercept

    # Plot the density distribution function with the linear regressionx
    plt.plot(log_radius, log_density_distribution, label='Data')
    plt.plot(log_radius, linear_regression, label='Linear Regression', linestyle='--')

    # Add formula to plot
    formula_text = f'y = {slope:.4f} * x + {intercept:.4f}'
    plt.text(0.30, 0.80, formula_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.xlabel('log(Radius)')
    plt.ylabel('log(Number of Clustered Points)')
    plt.title('Density Distribution Function')
    plt.legend()
    '''

print("The results of the fitting dimension:", dim_fitting_result)
avg = np.mean(dim_fitting_result)
sd = np.std(dim_fitting_result)

print(f"Average Dimension: {avg:.4f}")
print(f"Standard Deviation: {sd:.4f}")

iavg = np.mean(constant_fitting_result)

############################ Plot the Figure ############################

# Calculate the density distribution function
max_radius = int(np.ceil(np.max(all_distances)))
density_distribution = [points_within_radius(all_distances, r) for r in range(max_radius+1)]

convolve_num = 10
log_radius = np.log(np.arange(10, max_radius-(convolve_num+118)))

# Define the convolution window
window = np.ones(convolve_num) / convolve_num

edge_distribution = [density_distribution[i+1]-density_distribution[i] for i in range(10, max_radius-(convolve_num+109))]
edge_distribution = np.convolve(edge_distribution, window, mode='valid')

for i in range(len(edge_distribution)):
    edge_distribution[i] = float(edge_distribution[i]) / np.exp(log_radius[i])

log_density_distribution = np.log(edge_distribution)

nonzero_indices = (log_radius != 0) & (log_density_distribution != 0)
log_radius_nonzero = log_radius[nonzero_indices]
log_density_distribution_nonzero = log_density_distribution[nonzero_indices]

slope, intercept = np.polyfit(log_radius_nonzero, log_density_distribution_nonzero, 1)
linear_regression = slope * log_radius + intercept

# Plot the density distribution function with the linear regressionx
plt.plot(log_radius, log_density_distribution, label='Data')
plt.plot(log_radius, linear_regression, label='Linear Regression', linestyle='--')

# Add formula to plot
formula_text = f'y = {slope:.4f} * x + {intercept:.4f}'
plt.text(0.30, 0.80, formula_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.xlabel('log(Radius)')
plt.ylabel('log(Number of Clustered Points)')
plt.title('Density Distribution Function')
plt.legend()
plt.show()