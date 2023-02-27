import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Loading the data
data = pd.read_csv('data/emails.csv')
data = data.drop(['Email No.'], axis=1)

training_data = data.iloc[:, :-1] # Extracting all feature values for all instances
training_labels = data.iloc[:, -1] # Extracting the labels for all instances 

# Initializing arrays to store results for each fold iteration
accuracies = []
precisions = []
recalls = []

# Running an iteration for each fold
for i in range(5):

    # Dividing our global datasets into sections according to fold index
    local_training_data = training_data.iloc[(1000 *i): (1000 * (i+1) + 1), :]
    local_training_labels = training_labels.iloc[1000 *i:1000 * (i+1) + 1]

    local_testing_data =  pd.concat([training_data.iloc[0:(1000*i), :], training_data.iloc[(1000 * (i+1) + 1):, :]])
    local_testing_labels = pd.concat([training_labels.iloc[:(1000*i)], training_labels.iloc[(1000 * (i+1) + 1):]])

    # Training the 1-NN model from sklearn's libraries
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(local_training_data, local_training_labels)
    
    # Calculating our prediction values
    predictions = model.predict(local_testing_data)

    # Calculating the needed values to return
    accuracy = accuracy_score(local_testing_labels, predictions)    # Calculating accuracy
    precision = precision_score(local_testing_labels, predictions)  # Calculating precision
    recall = recall_score(local_testing_labels, predictions)        # Calculating recall

    # Append the needed values to the global arrays
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

print(accuracies)
print(precisions)
print(recalls)
