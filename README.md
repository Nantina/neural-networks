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
- Output Layer (Softmaz activation function)

#### Results 
The best results were observed with the following hyperparameters - techniques 
- Batch size = 100
- Number of neurons in the hidden layer = 126
- Stochastic Gradient Descent Optimizer with Learning Rate Decay (initial learning rate = 0.001, learning rate decay = 0.01)

The best results are:
- Training Loss: **0.9922**
- Test Accuracy: **0.5021**
- Training Time: **231.41** seconds (37 epochs)


