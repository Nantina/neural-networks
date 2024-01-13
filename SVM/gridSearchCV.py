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


# ~~~~~~~ Linear ~~~~~~~~
# Starting point to calculate the training time 
start_time = time.time()
# declare parameters for hyperparameter tuning Linear
parameters_linear = {'C': [0.1, 1, 10, 100], 'kernel':['linear']}
svc = SVC()
grid_search = GridSearchCV(svc, parameters_linear, cv=5, scoring='accuracy')
# Fit the model to the data
grid_search.fit(X_train_divided, y_train_divided)
# Print the best parameters
print("Best parameters: ", grid_search.best_params_)
# Stop measuring training time
end_time = time.time()
# Calculate and print the total training time
training_time = end_time - start_time
print(f"Total training time Linear: {training_time:.2f} seconds")


# ~~~~~~~~~~~~ Polynomial ~~~~~~~~~~~~~~~~~~~~~~
start_time = time.time()
parameters_poly = {'C': [0.1, 1, 10, 100], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.1,1,10]} 
svc = SVC()
grid_search = GridSearchCV(svc, parameters_poly, cv=5, scoring='accuracy')
grid_search.fit(X_train_divided, y_train_divided)
print("Best parameters for Polynomial Kernel: ", grid_search.best_params_)
end_time = time.time()
training_time = end_time - start_time
print(f"Total training time Polynomial: {training_time:.2f} seconds")

# ~~~~~~~~~~~~ RBF ~~~~~~~~~~~~~~~~~~
parameters_rbf = {'C': [0.1, 1, 10, 100], 'kernel':['rbf'], 'gamma':[0.01, 0.1, 1, 10]}
svc = SVC()
grid_search = GridSearchCV(svc, parameters_rbf, cv=5, scoring='accuracy')
grid_search.fit(X_train_divided, y_train_divided)
print("Best parameters for RBF Kernel: ", grid_search.best_params_)
end_time = time.time()
training_time = end_time - start_time
print(f"Total training time RBF: {training_time:.2f} seconds")

# ~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~
start_time = time.time()
parameters_sigmoid = {'C': [0.1, 1, 10, 100], 'kernel':['sigmoid'] ,'gamma':[0.01,0.1,1,10]} 
svc = SVC()
grid_search = GridSearchCV(svc, parameters_sigmoid, cv=5, scoring='accuracy')
grid_search.fit(X_train_divided, y_train_divided)
print("Best parameters for Sigmoid Kernel: ", grid_search.best_params_)
end_time = time.time()
training_time = end_time - start_time
print(f"Total training time Sigmoid: {training_time:.2f} seconds")