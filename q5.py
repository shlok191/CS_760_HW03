from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

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

        # Printing the loss every 500 iterations
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

training_data = data.iloc[:3500, :-1] # Extracting all feature values for all instances
training_labels = data.iloc[:3500, -1] # Extracting the labels for all instances 

testing_data = data.iloc[3500:, :-1] # Extracting all feature values for all instances
testing_labels = data.iloc[3500:, -1] # Extracting the labels for all instances 

# Initializing a kNN model with k=5 (Imported from sklearn)
knn = KNeighborsClassifier(5)
knn.fit(training_data, training_labels)

# Making predictions on the test set using KNN
knn_predictions = knn.predict_proba(testing_data)[:, 1]

# Calculating the false positive rates and true positive rates for KNN
knn_fpr, knn_tpr, thresholds = roc_curve(testing_labels, knn_predictions)

# Calculating the area under the curve (AUC) for KNNs
knn_roc_auc = auc(knn_fpr, knn_tpr)

# Repeating process for logistical regression
theta_values = gradient_descent(training_data, training_labels, 0.01, 3000)

# Evaluate the model on the test data
regression_predictions = predict(testing_data, theta_values)

# Calculating the false positive rates and true positive rates for the regression model
regression_fpr, regression_tpr, thresholds = roc_curve(testing_labels, regression_predictions)

# Calculating the area under the curve (AUC) for regression
regression_roc_auc = auc(regression_fpr, regression_tpr)

# Plotting the ROC curve
plt.plot(knn_fpr, knn_tpr, color='darkorange', lw=2, label='ROC curve for KNN (area = %0.2f)' % knn_roc_auc)
plt.plot(regression_fpr, regression_tpr, color='darkblue', lw=2, label='ROC curve for regression (area = %0.2f)' % regression_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()