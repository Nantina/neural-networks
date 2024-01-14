## Description 

In this repository various classification algoritmhs are tested and compared regarding a multiclass dataset. 
Specifically the dataset on which these algorithms are applied is the Cifar-10 dataset which consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. More information regarding the dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

The files containing the Cifar-10 data are stored in the `cifar-10-batches-py` folder. 

## Classification Algorithms 

The algorithms of this repository are mentioned and described below.


### K - Nearest Neighbours and Nearest Centroid 
The `Knn-NearestCentroid` folder contains the file in which the Knn and Nearest Centroid algorithms are applies to the Cifar-10 dataset. The sklearn library is used for both algorithms.

Specifically for the KNN algorithm, the 1 and 3 neighbors are tested.

The results for the Accuracy of the Test set are:
- Knn k=1 neighbor: **0.3539**
- Knn k=3 neighbors: **0.3303**
- Nearest Centroid:  **0.2774**

### Multilayer Perceptron Neural Network

The `Multilayer-Perceptron-NN` folder contains the file in which a multilayer perceptron neural network is implemented from scratch. The network is traing with the Backpropagation algorithm. 

#### Structure 
- One Hidden Layer (RELU activation function)
- Output Layer (Softmax activation function)

#### Results 
The best results were observed with the following hyperparameters - techniques 
- Batch size = 100
- Number of neurons in the hidden layer = 126
- Stochastic Gradient Descent Optimizer with Learning Rate Decay (initial learning rate = 0.001, learning rate decay = 0.01)

The best results are:
- Training Loss: **0.9922**
- Test Accuracy: **0.5021**
- Training Time: **231.41** seconds (37 epochs)


### Support Vector Machines

The `SVM` folder contains three files for classification with Support Vector Machine techniques of 2 of the Cifar-10 classes. 
Firslty, a division is applied in the dataset in order to keep only the 2 first classes (airplane and automobile). In all files, the SVC function from `sklearn.svm` is used and all kernels are tested.

#### Grid Search 
In the `gridSearchCV.py` file, the `GridSearchCV` function is used. It basically scans and calculated the results for the different hyperparameters and kernels and returns the parameters that produce the best results. The comparison is being done in respect to a specified condition. In this Grid Search the `accuracy` scoring is the comparison critirion. 
All kernel types (Linear, RBF, Sigmoid, Polynomial) are tested with different hyperparameters for each. 

#### Different Parameters

In the `different_params.py` file, a similar procedure to the Grid Search is performed. Here the different hyperparameter sets are defined as dictionaries, then the `SVC` function is performed for each set and finally the results for the training accuracy and training time are plotted for comparison. 

#### Best Results

In the `best_results.py` file, the `SVC` function is used in order to print the results (Accuracy, Classification Report and Confusion Matrix) for the parameters that produce the best results that were computed through the Grid Search.


### Radial Basis Function Neural Network

The `RBF-NN` folder contains the file in which a radial basis function neural network is implemented from scratch. The network is traing with the Backpropagation algorithm. 

#### Structure 
- One Hidden Layer (RBF activation function)
- Output Layer (Softmax activation function)

#### Results 
The best results were observed with the following hyperparameters - techniques 
- Number of Centers = 200 (they are computed using the K-means algorithm, 20 for each class)
- Sigma = 4
- Stochastic Gradient Descent Optimizer with Learning Rate Decay (initial learning rate = 0.1, learning rate decay = 0.001)

The best results are:
- Test Accuracy: **0.3862**
- Training Time: **74.22** seconds (100 epochs)