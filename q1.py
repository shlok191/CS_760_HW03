import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Loading in the data values to read in the inputs
data = np.loadtxt('data/D2z.txt', delimiter=' ')

# Separating features as X_train and labels as Y_train
training_data = data[:, 0:2]
training_labels = data[:, 2]

# Defining the range (going beyond to 2.1 since it does not touch the end value) of the test points
# Beginning range: -2, # Ending Range: 2.1, # Step Size: 0.1

# Since feature_value lists should be the same for both features, we can use the same array
# and save space!
feature_values = np.arange(-2, 2.1, 0.1)

# Creating a grid of random test points
testing_data = np.c_[np.ravel(np.meshgrid(feature_values, feature_values)[0]), np.ravel(np.meshgrid(feature_values, feature_values)[1])]
print(testing_data.shape)

# Computing the needed distances from neighbours (1-NN)
euclid_distances = cdist(testing_data, training_data)

# Predicting the labels of the test points based on the closest neighbors labels
predictions = training_labels[np.argmin(euclid_distances, axis=1)].reshape(len(feature_values), len(feature_values))

# Creating a scatter plot of the training set
plt.scatter(training_data[:, 0], training_data[:, 1], c=training_labels, cmap='bwr')
plt.contourf(feature_values, feature_values, predictions, alpha=0.2, cmap='bwr')

plt.title('Nearest Neighbour Implementation with K=1')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')

plt.show()
