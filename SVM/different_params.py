import pandas as pd
import time
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow import keras
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load Dataset in batches 

# filename = 'cifar-10-batches-py/data_batch_1'

# with open (filename, 'rb') as f:
#     datadict = pickle.load(f, encoding = 'bytes')
#     X = datadict[b'data']
#     Y = datadict[b'labels']
#     X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
#     Y = np.array(Y)

# # # X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.4, random_state=0) 
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 

# Load Comlpete Dataset

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Making sure the sizes of the train and test sets match the sizes mentioned in the CIFAR-10 site
assert X_train.shape == (50000, 32, 32, 3)
assert X_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

class_id_1, class_id_2 = 0, 1

# Creating two groups: class 0 and all other classes
class_1_indices = np.where(y_train == class_id_1)[0]
class_2_indices = np.where(y_train == class_id_2)[0]

# Combine all elements into a single array
X_train_divided = np.concatenate((X_train[class_1_indices], X_train[class_2_indices]), axis=0)
y_train_divided = np.concatenate((y_train[class_1_indices], y_train[class_2_indices]), axis=0)

# Similarly for the test set
class_1_indices_test = np.where(y_test == class_id_1)[0]
class_2_indices_test = np.where(y_test == class_id_2)[0]

X_test_divided = np.concatenate((X_test[class_1_indices_test], X_test[class_2_indices_test]), axis=0)
y_test_divided = np.concatenate((y_test[class_1_indices_test], y_test[class_2_indices_test]), axis=0)

# Making sure the sizes match the selected classes
assert X_train_divided.shape[0] == y_train_divided.shape[0]
assert X_test_divided.shape[0] == y_test_divided.shape[0]

# Normalize data
X_train_divided, X_test_divided = X_train_divided / 255.0, X_test_divided / 255.0

# Reshape data
X_train_divided = X_train_divided.reshape(X_train_divided.shape[0], -1)
X_test_divided = X_test_divided.reshape(X_test_divided.shape[0], -1)
y_train_divided = y_train_divided.reshape(-1,)
y_test_divided = y_test_divided.reshape(-1,)

# Print number of elements for each class 
num_elements_class_1_train = np.sum(y_train_divided == class_id_1)
print("Number of elements of class ",class_id_1, " in the training set:", num_elements_class_1_train)

num_elements_class_2_train = np.sum(y_train_divided == class_id_2)
print("Number of elements of class ",class_id_2, " in the training set:", num_elements_class_2_train)


param_sets = [
    # Different number of neurons 
    {'C': 0.01},
    {'C': 0.1},
    {'C': 1.0},
    {'C': 10.0},
    {'C': 100}
]

param_sets_poly = [
    {'gamma': 'scale', 'degree': 2},
    {'gamma': 'scale', 'degree': 3},
    {'gamma': 'scale', 'degree': 4},
]

# param_sets_poly = [
#     # Different number of neurons 
#     {'gamma': 0.01},
#     {'gamma': 0.1},
#     {'gamma': 1.0},
#     {'gamma': 10.0}
# ]

# Examples for different sets of parameters

# ~~~~~~~ Linear ~~~~~~~~
C_values = [params['C'] for params in param_sets]
accuracy_values_train = []
accuracy_values_test = []
training_time_values = []
for params in param_sets:
    # Linear Kernel
    start_time = time.time()
    linear = SVC(kernel='linear', C=params['C']).fit(X_train_divided, y_train_divided)
    linear_pred_train = linear.predict(X_train_divided)
    linear_pred_test = linear.predict(X_test_divided)
    linear_accuracy_test = accuracy_score(y_test_divided, linear_pred_test)
    linear_accuracy_train = accuracy_score(y_train_divided, linear_pred_train)
    end_time = time.time()
    training_time = end_time - start_time
    accuracy_values_train.append(linear_accuracy_train)
    accuracy_values_test.append(linear_accuracy_test)
    training_time_values.append(training_time)
    print('Accuracy training (Linear Kernel): ', "%.2f" % (linear_accuracy_train))
    print('Accuracy Testing (Linear Kernel): ', "%.2f" % (linear_accuracy_test))

# ~~~~~~~ Sigmoid ~~~~~~~~
C_values = [params['C'] for params in param_sets]
accuracy_values_train = []
accuracy_values_test = []
training_time_values = []
for params in param_sets:
    # Linear Kernel
    start_time = time.time()
    linear = SVC(kernel='sigmoid', C=params['C']).fit(X_train_divided, y_train_divided)
    linear_pred_train = linear.predict(X_train_divided)
    linear_pred_test = linear.predict(X_test_divided)
    linear_accuracy_test = accuracy_score(y_test_divided, linear_pred_test)
    linear_accuracy_train = accuracy_score(y_train_divided, linear_pred_train)
    end_time = time.time()
    training_time = end_time - start_time
    accuracy_values_train.append(linear_accuracy_train)
    accuracy_values_test.append(linear_accuracy_test)
    training_time_values.append(training_time)
    # linear_f1 = f1_score(y_test_divided, linear_pred_train, average='weighted')
    print('Accuracy training (Linear Kernel): ', "%.2f" % (linear_accuracy_train))
    # linear_f1 = f1_score(y_test_divided, linear_pred_test, average='weighted')
    print('Accuracy Testing (Linear Kernel): ', "%.2f" % (linear_accuracy_test))
    # print('F1 (Linear Kernel): ', "%.2f" % (linear_f1*100))

# ~~~~~~~ RBF  ~~~~~~~~
gamma_values = [params['gamma'] for params in param_sets_poly]
accuracy_values_train = []
accuracy_values_test = []
training_time_values = []
for params in param_sets_poly:
    # Linear Kernel
    start_time = time.time()
    linear = SVC(kernel='rbf', gamma = params['gamma']).fit(X_train_divided, y_train_divided)
    linear_pred_train = linear.predict(X_train_divided)
    linear_pred_test = linear.predict(X_test_divided)
    linear_accuracy_test = accuracy_score(y_test_divided, linear_pred_test)
    linear_accuracy_train = accuracy_score(y_train_divided, linear_pred_train)
    end_time = time.time()
    training_time = end_time - start_time
    accuracy_values_train.append(linear_accuracy_train)
    accuracy_values_test.append(linear_accuracy_test)
    training_time_values.append(training_time)
    # linear_f1 = f1_score(y_test_divided, linear_pred_train, average='weighted')
    print('Accuracy training (Linear Kernel): ', "%.2f" % (linear_accuracy_train))
    # linear_f1 = f1_score(y_test_divided, linear_pred_test, average='weighted')
    print('Accuracy Testing (Linear Kernel): ', "%.2f" % (linear_accuracy_test))
    # print('F1 (Linear Kernel): ', "%.2f" % (linear_f1*100))

# Examples for Plotting Training Accuracy and Time for Sigmoid Kernel

# Plot Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(C_values, accuracy_values_test, marker='o', linestyle='-', color='blue', label='Test')
plt.plot(C_values, accuracy_values_train, marker='o', linestyle='-', color='orange', label='Train')
plt.title('Accuracy for different C values (Sigmoid)')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot Training time 
plt.figure(figsize=(10, 5))
plt.plot(C_values, training_time_values, marker='o', linestyle='-')
plt.title('Training Time for different C values')
plt.xlabel('C')
plt.ylabel('Training Time')
plt.legend()
plt.show()