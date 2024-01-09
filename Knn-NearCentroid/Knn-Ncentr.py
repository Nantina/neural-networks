from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import tensorflow
from tensorflow import keras

# Function used for scaling of the dataset
def feature_scaling(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Knn Classifier function for k neighbors. Fit -> Predict -> Accuracy Score 
def knn_classifier(k, X_train, y_train, X_test, y_test):
    print('Nearest Neighbors Classifier results k = ', k, '\n\n')
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    y_pred_k = knn_k.predict(X_test)
    accuracy_k = accuracy_score(y_test, y_pred_k)
    # Classification report
    print('Classification Report\n\n')
    print(classification_report(y_test, y_pred_k))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_k)
    print('Confusion matrix\n\n', cm)
    print('\n\n')
    return accuracy_k 

# Nearest Centroid Classifier function. Fit -> Predict -> Accuracy Score 
def nearest_centroid_classifier( X_train, y_train, X_test, y_test):
    print('Nearest Centroid Classifier results\n\n')
    ncc = NearestCentroid()
    ncc.fit(X_train, y_train)
    ncc_pred = ncc.predict(X_test)
    accuracy_nearest_centroid = accuracy_score(y_test, ncc_pred)
    # Classification report
    print('Classification Report\n\n')
    print(classification_report(y_test, ncc_pred))
    # Confusion Matrix
    cm = confusion_matrix(y_test, ncc_pred)
    print('Confusion matrix\n\n', cm)
    print('\n\n')
    return accuracy_nearest_centroid 

def main():
    # Loading the data using the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    # Making sure the sizes of the train and test sets match the sizes mentioned in the CIFAR-10 site
    assert X_train.shape == (50000, 32, 32, 3)
    assert X_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    # Reshaping the sets in order to convert them from 4D to 2D 
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Reshaping the y_train set into its Transpose
    y_train = y_train.reshape(-1,)

    # Scaling the data using MinMaxScaler which scales the data between the range [0,1]
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)

    # Classification using the Knn Classifier with 1 and 3 neighbors and the Nearest Centroid Classifier
    accuracy_knn_1 = knn_classifier(1, X_train_scaled, y_train, X_test_scaled, y_test)
    accuracy_knn_3 = knn_classifier(3, X_train_scaled, y_train, X_test_scaled, y_test)
    accuracy_nearest_centroid = nearest_centroid_classifier(X_train, y_train, X_test, y_test)

    # Printing the Accuraccy scores for all classifiers 
    print("Accuracy with 1 neighbor: {0:0.4f}".format(accuracy_knn_1))
    print("Accuracy with 3 neighbors: {0:0.4f}".format(accuracy_knn_3))
    print("Accuracy of nearest centroid: {0:0.4f}".format(accuracy_nearest_centroid))

    # Plotting accuracies 
    names = ['knn_1', 'knn_3', 'near_cent']
    values = [accuracy_knn_1, accuracy_knn_3, accuracy_nearest_centroid]

    plt.scatter(names, values)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.suptitle('Accuracy Plotting')
    plt.show()

if __name__ == "__main__":
    main()







