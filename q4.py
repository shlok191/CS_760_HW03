from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Loading the data
data = pd.read_csv('data/emails.csv')
data = data.drop(['Email No.'], axis=1)

training_data = data.iloc[:, :-1] # Extracting all feature values for all instances
training_labels = data.iloc[:, -1] # Extracting the labels for all instances 

# Creating the list of k-values we will be working from!
k_values = [1, 3, 5, 7, 10]

# Initializing arrays to store results for each fold iteration
avg_accuracy = []

for i in k_values:

    # Initializing a kNN model with the given value of k (Imported from sklearn)
    knn = KNeighborsClassifier(i)
    score = cross_val_score(knn, training_data, training_labels, cv=5)
    
    # Calculating the average score from the list of scores for 5-fold validations!
    avg_acc_score = score.mean()
    avg_accuracy.append(avg_acc_score)

# Plotting the needed plot

plt.plot(k_values, avg_accuracy)
plt.xlabel('Value of K')
plt.ylabel('Average accuracy over 5 fold cross-validations')
plt.title('5-Fold Cross-Validation accuracy vs K plot')
plt.show()
