import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Defining the sigmoid function to convert our predicted values into range of [0, 1]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Defining the cross_entropy loss function
def cross_entropy_loss(predicted_label, training_label):

    # Calculates the loss function value as dictated in Q6!
    loss = -1 * (training_label * np.log(predicted_label) + (1 - training_label) * np.log(1 - predicted_label))
    
    return np.mean(loss)

# Defining the gradient calculating function
def gradient(training_data, training_labels, predicted_label):

    error = predicted_label - training_labels
    gradient = np.dot(training_data.T, error) / len(training_labels)
    return gradient

# The gradient descent algorithm
def gradient_descent(training_data, training_label, alpha, iterations):
    
    # Initializing the theta_values to zero initially
    theta_values = np.zeros(3000)

    for i in range(iterations):

        # Making a prediction using the in-progress theta values
        predicted_label = sigmoid(np.dot(training_data, theta_values))

        # Calculating the loss function and gradient function values
        loss = cross_entropy_loss(predicted_label, training_label)
        gradient_val = gradient(training_data, training_label, predicted_label)

        # Updating the theta values
        theta_values -= alpha * gradient_val

        # Printing the loss every 100 iterations
        if i % 100 == 0:
            print(f'Iteration {i}: Loss = {loss:.4f}')

    return theta_values

# Finally, defining the prediction function

def predict(training_data, theta_values):

    predicted_labels = []

    for index, row in training_data.iterrows():
        
        predicted_label = sigmoid(np.dot(row, theta_values))
        label_range_fitted = np.where(predicted_label >= 0.5, 1, 0)

        predicted_labels.append(label_range_fitted)
        
    return predicted_labels

# Loading the data
data = pd.read_csv('data/emails.csv')
data = data.drop(['Email No.'], axis=1)

training_data = data.iloc[:, :-1] # Extracting all feature values for all instances
training_labels = data.iloc[:, -1] # Extracting the labels for all instances 

# Initializing arrays to store results for each fold iteration
accuracies = []
precisions = []
recalls = []

for i in range(5):

    # Dividing our global datasets into sections according to fold index
    local_training_data = training_data.iloc[(1000 *i): (1000 * (i+1) + 1), :]
    local_training_labels = training_labels.iloc[1000 *i:1000 * (i+1) + 1]

    local_testing_data =  pd.concat([training_data.iloc[0:(1000*i), :], training_data.iloc[(1000 * (i+1) + 1):, :]])
    local_testing_labels = pd.concat([training_labels.iloc[:(1000*i)], training_labels.iloc[(1000 * (i+1) + 1):]])

    # Training the model using gradient descent
    theta_values = gradient_descent(local_training_data, local_training_labels, 0.01, 1000)
    
    # Evaluating the model on the test data
    predictions = predict(local_testing_data, theta_values)

    # Calculating the evaluation metrics
    accuracy = accuracy_score(local_testing_labels, predictions)
    precision = precision_score(local_testing_labels, predictions)
    recall = recall_score(local_testing_labels, predictions)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

print(accuracies)
print(precisions)
print(recalls)

