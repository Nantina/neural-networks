import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tensorflow import keras
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.random.seed(42)

def one_hot_encode(labels, num_classes):
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        label = labels[i]
        one_hot_labels[i][label] = 1
    return one_hot_labels

def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def calculate_loss(pred_output, true_output):
    epsilon = 1e-15
    pred_output_clipped = np.clip(pred_output, epsilon, 1 - epsilon)
    correct_confidences = np.sum(pred_output_clipped * true_output, axis=1)
    loss = -np.log(correct_confidences)
    average_loss = np.mean(loss)
    return average_loss

# Load CIFAR-10 dataset
def load_cifar10():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    # Flatten images and normalize pixel values
    # Normalize data
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train.reshape((X_train.shape[0], -1)) 
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    # One-hot encode labels
    y_train_one_hot_encoded = one_hot_encode(y_train, 10)
    y_test_one_hot_encoded = one_hot_encode(y_test, 10)

    # Apply PCA to retain 90% of the information
    pca = PCA(0.90) 
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("Number of features retained after PCA:", pca.n_components_)
    
    return X_test, X_train_pca, X_test_pca, y_train_one_hot_encoded, y_test_one_hot_encoded, y_train

class NN:
    def __init__(self, X,y, num_centers, num_classes, sigma, lr):
        self.weights = 0.01 * np.random.randn(num_centers, num_classes)
        self.centers = self.initialize_centers(X,y, num_centers)
        self.sigma = sigma
        self.learning_rate = lr

    def initialize_centers(self, X, y, num_centers):
        # Initialize an array to store centers for each class
        class_centers = []

        # Loop through each class 
        for class_label in range(10):
            # Extract data points belonging to the current class
            class_1_indices = np.where(y == class_label)[0]
            class_data = X_train[class_1_indices]
            num_clusters = num_centers // 10

            if initialize_centers == 'kmeans':
                # Apply KMeans to find centers for the current class
                kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
                kmeans.fit(class_data)
                # Append the centers to the list
                class_centers.append(kmeans.cluster_centers_)
            elif initialize_centers == 'random':
                center_indices = np.random.choice(X.shape[0], num_clusters, replace=False)
                rbf_centers = X[center_indices]
                class_centers.append(rbf_centers)

        # Combine the class centers into a single array
        all_centers = np.concatenate(class_centers, axis=0)

        return all_centers
    
    # Activation functions
    def rbf_gaussian_activation(self, X, sigma):
        return np.exp(-cdist(X, self.centers, 'sqeuclidean') / (2 * sigma**2))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Forward Pass
    def forward(self, X):
        rbf_activations = self.rbf_gaussian_activation(X, self.sigma)
        return rbf_activations
    
    def forward_output(self, rbf_activations):
        output = rbf_activations.dot(self.weights)
        predictions = self.softmax(output)
        return predictions

    # Backward Pass
    def backward(self, X, output, rbf_activations, y):
        error = output - y
        grad_weights = rbf_activations.T.dot(error)
        return grad_weights
    
    def updates(self, grad_weights):
        self.weights -= self.learning_rate * grad_weights

if __name__ == "__main__":
    X_test_before_pca, X_train, X_test, y_train, y_test, y_train_no_encoding = load_cifar10()

    # Extract class names
    filename = '../cifar-10-batches-py/batches.meta'
    with open(filename, 'rb') as f:
        meta_dict = pickle.load(f, encoding='bytes')
    class_names = [class_name.decode('utf-8') for class_name in meta_dict[b'label_names']]

    param_sets = [
        # Different number of centers
        # {'number of centers': 10, 'learning_rate': 0.001, 'sigma': 4},
        # {'number of centers': 50, 'learning_rate': 0.001, 'sigma': 4},
        # {'number of centers': 100, 'learning_rate': 0.001, 'sigma': 4},
        # {'number of centers': 200, 'learning_rate': 0.001, 'sigma': 4},
        # {'number of centers': 500, 'learning_rate': 0.001, 'sigma': 4}

        # Different sigmas
        # {'number of centers': 200, 'learning_rate': 0.001, 'sigma': 0.5},
        # {'number of centers': 200, 'learning_rate': 0.001, 'sigma': 1},
        # {'number of centers': 200, 'learning_rate': 0.001, 'sigma': 2},
        # {'number of centers': 200, 'learning_rate': 0.001, 'sigma': 4},
        # {'number of centers': 200, 'learning_rate': 0.001, 'sigma': 5}

        # Different learning rates
        # {'number of centers': 200, 'learning_rate': 0.1, 'sigma': 4, 'learning_rate_decay':0.1},
        # {'number of centers': 200, 'learning_rate': 0.01, 'sigma': 4, 'learning_rate_decay':0.1},
        # {'number of centers': 200, 'learning_rate': 0.1, 'sigma': 4, 'learning_rate_decay':0.01},
        # {'number of centers': 200, 'learning_rate': 0.01, 'sigma': 4, 'learning_rate_decay':0.01},
        # {'number of centers': 200, 'learning_rate': 0.1, 'sigma': 4, 'learning_rate_decay':0.001}
        # {'number of centers': 200, 'learning_rate': 0.01, 'sigma': 4, 'learning_rate_decay':0.001}
        # {'number of centers': 200, 'learning_rate': 0.001, 'sigma': 4},
        # {'number of centers': 200, 'learning_rate': 0.0001, 'sigma': 4}

        # Different way of computing the centers
        {'initialize_centers': 'kmeans', 'number of centers': 200, 'learning_rate': 0.1, 'sigma': 4, 'learning_rate_decay':0.001},
        # {'initialize_centers': 'random', 'number of centers': 200, 'learning_rate': 0.1, 'sigma': 4, 'learning_rate_decay':0.001},
    ]

    # Create lists to store results for each parameter set
    all_loss_values = []
    all_accuracy_values = []
    all_training_times = []
    all_testing_accuracies = []

    for params in param_sets:
        epochs = 100
        start_time = time.time()
        initialize_centers = params['initialize_centers']

        RBF_NN = NN(X_train, y_train_no_encoding, num_centers = params['number of centers'], num_classes= 10, sigma=params['sigma'], lr= params['learning_rate'])

        # Initializing arrays for the plots 
        loss_values = []
        accuracy_values = []        

        for epoch in range(epochs):
            # X_train, y_train = shuffle_data(X_train, y_train)
            rbf_activations = RBF_NN.forward(X_train)
            predictions_output = RBF_NN.forward_output(rbf_activations)

            grad_weights = RBF_NN.backward(X_train, predictions_output, rbf_activations, y_train)
            RBF_NN.updates(grad_weights)

            predictions = np.argmax(predictions_output, axis=1)
            actual = np.argmax(y_train, axis=1)

            loss = calculate_loss(predictions_output, y_train)
            accuracy = np.mean(predictions == actual)

            lr_decay = params['learning_rate_decay']
            RBF_NN.learning_rate = RBF_NN.learning_rate * (1/ (1 + lr_decay * epoch))

            # Print loss and accuracies for each epoch 
            print('Epoch: ', epoch, 'loss: ', loss, 'accuracy: ', accuracy)
            
            # Store the values of accuracy and loss for plotting
            loss_values.append(loss)
            accuracy_values.append(accuracy)

        # Stop measuring training time
        end_time = time.time()

        # Calculate and print the total training time
        training_time = end_time - start_time
        print(f"Total training time: {training_time:.2f} seconds")
        all_training_times.append(training_time)

        # Make one forward pass for the Test set
        rbf_activations= RBF_NN.forward(X_test)
        predictions_output = RBF_NN.forward_output(rbf_activations)

        # Make predictions
        predictions = np.argmax(predictions_output, axis=1)
        actual = np.argmax(y_test, axis=1)

        # Calculate accuracy of test set 
        test_accuracy = np.mean(predictions == actual)
        print(f'Test accuracy: {test_accuracy * 100:.2f}%')

        all_testing_accuracies.append(test_accuracy)
        all_loss_values.append(loss_values)
        all_accuracy_values.append(accuracy_values)

    # Calculate precision, recall, and f1 score for each class
    precision_per_class = precision_score(actual, predictions, average=None)
    recall_per_class = recall_score(actual, predictions, average=None)
    f1_per_class = f1_score(actual, predictions, average=None)

    # Print classification report
    print("Classification Report:")
    print(classification_report(actual, predictions, target_names=class_names))

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    for i, params in enumerate(param_sets):
        plt.plot(range(epochs), all_loss_values[i], label=f'Num Centers: {params["number of centers"]}, LR: {params["learning_rate"]:.4f}, Sigma: {params["sigma"]}, Decay: {params["learning_rate_decay"]:.3f}')
    plt.title('Training Loss Over Epochs for Different Parameter Sets')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Training Accuracy
    plt.figure(figsize=(10, 5))
    for i, params in enumerate(param_sets):
        plt.plot(range(epochs), all_accuracy_values[i], label=f'Num Centers: {params["number of centers"]}, LR: {params["learning_rate"]:.4f}, Sigma: {params["sigma"]}, Decay: {params["learning_rate_decay"]:.3f}')
    plt.title('Training Accuracy Over Epochs for Different Parameter Sets')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot Training Time
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(param_sets)), all_training_times, label='Training Time', color='blue', marker='o')
    plt.xticks(range(len(param_sets)), [f'NC: {params["number of centers"]}, LR: {params["learning_rate"]:.4f}, Sigma: {params["sigma"]}' for params in param_sets], rotation='vertical')
    plt.title('Training Time for Different Parameter Sets')
    plt.xlabel('Parameters')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.show()

    # Plot Testing Accuracy
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(param_sets)), all_testing_accuracies, label='Testing Accuracy', color='green', marker='o')
    plt.xticks(range(len(param_sets)), [f'NC: {params["number of centers"]}, LR: {params["learning_rate"]:.4f}, Sigma: {params["sigma"]}' for params in param_sets], rotation='vertical')
    plt.title('Testing Accuracy for Different Parameter Sets')
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Revert one-hot encoding to predicted classes
    predicted_classes = np.argmax(predictions_output, axis=1)
    y_test_final = np.argmax(y_test, axis=1)

    # Reshape X_test back to its original format
    X_test_original_format = X_test_before_pca * 255.0

    # Visualize 15 random samples
    plt.figure(figsize=(20, 18))  
    plt.subplots_adjust(0, 0, 0.9, 0.9, wspace=0.05, hspace=0.3)  # Adjust hspace for better separation
    samples_to_show = 15  # Display 15 images
    rows = 3 
    for i in range(samples_to_show):
        random_index = np.random.randint(0, len(X_test_original_format))
        plt.subplot(rows, samples_to_show // rows, i + 1)  # Adjust subplot layout
        # Reshape the image to (32, 32, 3)
        reshaped_image = X_test_original_format[random_index].reshape((32, 32, 3))
        plt.imshow(reshaped_image.astype(np.uint8), interpolation='spline16')
        plt.axis('off')
        predicted_class = class_names[predicted_classes[random_index]]
        actual_class = class_names[y_test_final[random_index]]
        plt.title(f'Pred: {predicted_class}\nActual: {actual_class}')
    plt.show()








        
