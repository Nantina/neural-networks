import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow
from tensorflow import keras
import pickle
import os
np.random.seed(42)

def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def get_batches(X, y, batch_size):
    num_batches = len(X) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        yield X[start_idx:end_idx], y[start_idx:end_idx]

def one_hot_encode(labels, num_classes):
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        label = labels[i]
        one_hot_labels[i][label] = 1
    return one_hot_labels

def calculate_loss(pred_output, true_output):
    epsilon = 1e-15
    pred_output_clipped = np.clip(pred_output, epsilon, 1 - epsilon)
    correct_confidences = np.sum(pred_output_clipped * true_output, axis=1)
    loss = -np.log(correct_confidences)
    average_loss = np.mean(loss)
    return average_loss

def calculate_accuracy(pred_output, true_output):
    # Make predictions
    predictions = np.argmax(pred_output, axis=1)
    actual = np.argmax(true_output, axis=1)
    # Calculate accuracy
    accuracy = np.mean(predictions == actual)
    return accuracy

class Layer:
    def __init__(self, in_neurons, out_neurons):
        self.weights = 0.01 * np.random.randn(in_neurons, out_neurons)
        self.biases = np.zeros((1, out_neurons))
    
    def forward_hidden(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.act_output = self.relu(self.output)

    def forward_output(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.act_output = self.softmax(self.output)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(int)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward_output(self, y_train, previous_output):
        # Partial derivative of the loss with respect to the softmax activation function's inputs 
        self.d_loss_softmax_inputs = self.act_output - y_train
        # Partial derivative of the softmax activation function's inputs with respect to the weights 
        self.d_softmax_inputs_weights = previous_output

        #for weights 
        self.cost_weights = self.d_softmax_inputs_weights.T.dot(self.d_loss_softmax_inputs)

        # for biases 
        num_rows_biases, num_cols_biases = self.biases.shape
        num_rows, num_cols = y_train.shape
        self.output_der_biases = np.ones((num_rows_biases,num_rows))
        self.cost_biases = self.output_der_biases.dot(self.d_loss_softmax_inputs)

    def backward_hidden(self, y_train, output_layer, previous_output):
        # Partial derivative of the loss with respect to the softmax activation function's inputs 
        self.d_loss_softmax_inputs = output_layer.d_loss_softmax_inputs
        # Partial derivative of the softmax activation function's inputs with respect to the output layer's inputs  
        self.d_softmax_inputs_inputs = output_layer.weights
        # Partial derivative of the loss with respect to the output layer's inputs 
        self.d_loss_inputs = self.d_loss_softmax_inputs.dot(self.d_softmax_inputs_inputs.T)
        # Partial derivative of the output layer's inputs (=hidden layer's outputs) with respect to the its weights 
        self.without_activation = self.relu_derivative(self.act_output)
        
        # for weights 
        self.cost_weights  = previous_output.T.dot(self.d_loss_inputs*self.without_activation)

        # for biases 
        num_rows_biases, num_cols_biases = self.biases.shape
        num_rows, num_cols = y_train.shape
        self.output_der_biases = np.ones((num_rows_biases,num_rows))
        self.cost_biases = self.output_der_biases.dot(self.d_loss_inputs*self.without_activation)

    def updates(self): 
        self.weights = np.subtract(self.weights, self.cost_weights * learning_rate)
        self.biases = np.subtract(self.biases, self.cost_biases * learning_rate)

# Loading the data using the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Making sure the sizes of the train and test sets match the sizes mentioned in the CIFAR-10 site
assert X_train.shape == (50000, 32, 32, 3)
assert X_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# Extract class names
filename = 'cifar-10-batches-py/batches.meta'
with open(filename, 'rb') as f:
    meta_dict = pickle.load(f, encoding='bytes')
class_names = [class_name.decode('utf-8') for class_name in meta_dict[b'label_names']]

# Normalize data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
y_train = y_train.reshape(-1,)
y_test_reshaped = y_test.reshape(-1,)

# Applying one hot encoding to the output data
y_train_one_hot_encoded = one_hot_encode(y_train, 10)
y_test_one_hot_encoded = one_hot_encode(y_test_reshaped, 10)

# Choose training type (possible types: 
# 1. 'lr_decay' = learning rate decay
# 2. 'normal' = static learning rate)
training_type = 'lr_decay'

# Training Factors
epochs = 60
batch_size = 100
number_of_neurons = 128
training_epochs = 0

if training_type == 'normal':
    learning_rate = 0.0005
elif training_type == 'lr_decay':
    initial_learning_rate = 0.001
    lr_decay = 0.01
    learning_rate = initial_learning_rate

#Initialization and creation of the layers         
hidden_layer = Layer(X_train.shape[1], number_of_neurons)
output_layer = Layer(number_of_neurons, 10) # 10 output classes

# Starting point to calculate the training time 
start_time = time.time()

# Initializing arrays for the plots 
loss_values = []
accuracy_values = []
test_accuracy_values = []

for epoch in range(epochs):
    #counting the training epichs before reaching the threshold
    training_epochs +=1

    # Shuffling the data in the beginning of each epoch 
    #X_train, y_train_one_hot_encoded = shuffle_data(X_train, y_train_one_hot_encoded)

    for X_batch, y_batch in get_batches(X_train, y_train_one_hot_encoded, batch_size):
        # Forward propagation
        hidden_layer.forward_hidden(X_batch)
        output_layer.forward_output(hidden_layer.act_output)

        # Backward propagation
        output_layer.backward_output(y_batch, hidden_layer.act_output)
        hidden_layer.backward_hidden(y_batch, output_layer, X_batch)

        # Update weights and biases
        hidden_layer.updates()
        output_layer.updates()

    # Calculate loss and accuracy for each epoch
    loss = calculate_loss(output_layer.act_output, y_batch)
    accuracy = calculate_accuracy(output_layer.act_output, y_batch)   
    # Store the values of accuracy and loss for plotting
    loss_values.append(loss)
    accuracy_values.append(accuracy) 

    # Check training type to determine the learning rate 
    if training_type == 'lr_decay':
        learning_rate = initial_learning_rate * (1/ (1 + lr_decay * epoch))

    # Testing 
    hidden_layer.forward_hidden(X_test_reshaped)
    output_layer.forward_output(hidden_layer.act_output)
    final_output = output_layer.act_output

    predictions = np.argmax(final_output, axis=1)
    actual = np.argmax(y_test_one_hot_encoded, axis=1)

    # Calculate accuracy of training set 
    test_accuracy = np.mean(predictions == actual)
    # Store the values of test accuracy for plotting
    test_accuracy_values.append(test_accuracy)

    # Print loss and accuracies for each epoch 
    print('Epoch: ', epoch, 'loss: ', loss, 'accuracy: ', accuracy, 'test accuracy: ', test_accuracy)

    # Check threshold for loss
    if loss <= 1.0:
        print("Reached Threshold - Stopping Training.")
        break
    
# Stop measuring training time
end_time = time.time()

# Calculate and print the total training time
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")

# Beginning of testing 
print('Final Testing')

hidden_layer.forward_hidden(X_test_reshaped)
output_layer.forward_output(hidden_layer.act_output)
final_output = output_layer.act_output

# Make predictions
predictions = np.argmax(final_output, axis=1)
actual = np.argmax(y_test_one_hot_encoded, axis=1)

# Calculate accuracy of training set 
test_accuracy = np.mean(predictions == actual)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Calculate precision, recall, and f1 score for each class
precision_per_class = precision_score(actual, predictions, average=None)
recall_per_class = recall_score(actual, predictions, average=None)
f1_per_class = f1_score(actual, predictions, average=None)

# Print classification report
print("Classification Report:")
print(classification_report(actual, predictions, target_names=class_names))

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(range(training_epochs), loss_values, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(training_epochs), test_accuracy_values, label='Test Accuracy')
plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(training_epochs), accuracy_values, label='Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Revert one-hot encoding to predicted classes
predicted_classes = np.argmax(final_output, axis=1)
y_test_final = np.argmax(y_test_one_hot_encoded, axis=1)

# Reshape X_test back to its original format
X_test_original_format = X_test*255.0

# Visualize some random samples to see if the predicted classes match the true ones 
plt.figure(figsize=(20, 6))
plt.subplots_adjust(0, 0, 0.9, 0.9, wspace=0.05, hspace=0.05)
samples_to_show = 10 # how many images to display 
for i in range(samples_to_show):
    random_index = np.random.randint(0, len(X_test_original_format))
    plt.subplot(1, samples_to_show, i + 1)
    plt.imshow(X_test_original_format[random_index].astype(np.uint8), interpolation='spline16')
    plt.axis('off')
    predicted_class = class_names[predicted_classes[random_index]]
    actual_class = class_names[y_test_final[random_index]]
    plt.title(f'Pred: {predicted_class}\nActual: {actual_class}')
plt.show()