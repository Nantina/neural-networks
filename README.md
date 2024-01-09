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

The Knn classifier with 1 neighbor has the best results.



