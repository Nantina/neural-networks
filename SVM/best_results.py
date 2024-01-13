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


# RBF Kernel
start_time = time.time()
rbf = SVC(C=10).fit(X_train_divided, y_train_divided)
rbf_pred = rbf.predict(X_test_divided)
rbf_accuracy = accuracy_score(y_test_divided, rbf_pred)
rbf_f1 = f1_score(y_test_divided, rbf_pred, average='weighted')
end_time = time.time()
training_time = end_time - start_time
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
print('Training Time: ', training_time)
print('Classification Report\n\n')
print(classification_report(y_test_divided, rbf_pred))
# Confusion Matrix
cm = confusion_matrix(y_test_divided, rbf_pred)
print('Confusion matrix\n\n', cm)
print('\n\n')

# Linear Kernel
start_time = time.time()
linear = SVC(kernel='linear', C = 0.1).fit(X_train_divided, y_train_divided)
linear_pred = linear.predict(X_test_divided)
linear_accuracy = accuracy_score(y_test_divided, linear_pred)
linear_f1 = f1_score(y_test_divided, linear_pred, average='weighted')
end_time = time.time()
training_time = end_time - start_time
print('Accuracy (Linear Kernel): ', "%.2f" % (linear_accuracy*100))
print('F1 (Linear Kernel): ', "%.2f" % (linear_f1*100))
print('Training Time: ', training_time)
print('Classification Report\n\n')
print(classification_report(y_test_divided, linear_pred, zero_division=1))
# Confusion Matrix
cm = confusion_matrix(y_test_divided, linear_pred)
print('Confusion matrix\n\n', cm)
print('\n\n')

# Sigmoid Kernel
start_time = time.time()
sigmoid = SVC(kernel='sigmoid', C=0.1, gamma=0.01).fit(X_train_divided, y_train_divided)
sigmoid_pred = sigmoid.predict(X_test_divided)
sigmoid_accuracy = accuracy_score(y_test_divided, sigmoid_pred)
sigmoid_f1 = f1_score(y_test_divided, sigmoid_pred, average='weighted')
end_time = time.time()
training_time = end_time - start_time
print('Accuracy (Sigmoid Kernel): ', "%.2f" % (sigmoid_accuracy*100))
print('F1 (Sigmoid Kernel): ', "%.2f" % (sigmoid_f1*100))
print('Training Time: ', training_time)
print('Classification Report\n\n')
print(classification_report(y_test_divided, sigmoid_pred))
# Confusion Matrix
cm = confusion_matrix(y_test_divided, sigmoid_pred)
print('Confusion matrix\n\n', cm)
print('\n\n')

# Polynomial Kernel
start_time = time.time()
polynomial = SVC(kernel='poly', C = 0.1, degree=2, gamma=0.01).fit(X_train_divided, y_train_divided)
polynomial_pred = polynomial.predict(X_test_divided)
polynomial_accuracy = accuracy_score(y_test_divided, polynomial_pred)
polynomial_f1 = f1_score(y_test_divided, polynomial_pred, average='weighted')
end_time = time.time()
training_time = end_time - start_time
print('Accuracy (Polynomial Kernel): ', "%.2f" % (polynomial_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (polynomial_f1*100))
print('Training Time: ', training_time)
print('Classification Report\n\n')
print(classification_report(y_test_divided, polynomial_pred))
# Confusion Matrix
cm = confusion_matrix(y_test_divided, polynomial_pred)
print('Confusion matrix\n\n', cm)
print('\n\n')