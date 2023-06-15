import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0) * 1

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size)
        self.b3 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = relu(self.z3)
        return self.a3

    def backward(self, X, y, output, learning_rate):
        d_loss = output - y
        d_W3 = np.dot(self.a2.T, d_loss)
        d_b3 = np.sum(d_loss, axis=0)
        d_a2 = np.dot(d_loss, self.W3.T)
        d_z2 = d_a2 * relu_derivative(self.z2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * relu_derivative(self.z1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)

        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W3 -= learning_rate * d_W3
        self.b3 -= learning_rate * d_b3

# Loading Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the neural network
nn = NeuralNetwork(4, 5, 3)
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    output = nn.forward(X_train)
    loss = np.mean((output - y_train) ** 2)
    nn.backward(X_train, y_train, output, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}")

# Testing the network
output_test = nn.forward(X_test)
predicted_labels = np.argmax(output_test, axis=1)
real_labels = np.argmax(y_test, axis=1)
accuracy = np.sum(predicted_labels == real_labels) / len(real_labels)
print(f"Test accuracy: {accuracy}")
