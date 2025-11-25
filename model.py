import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu'):
        self.activation_name = activation
        self.architecture = [input_size] + hidden_layers + [output_size]
        self.parameters = {}
        self.layers = {}
        self.L = len(self.architecture)
        self._initialize_parameters()

    def _initialize_parameters(self):
        scale = 0.01
        if self.activation_name in ['relu', 'leaky_relu']:
            scale = np.sqrt(2 / self.architecture[0])
        
        for i in range(1, self.L):
            self.parameters[f"W{i}"] = np.random.randn(self.architecture[i], self.architecture[i-1]) * scale
            self.parameters[f"b{i}"] = np.zeros((self.architecture[i], 1))

    def _activate(self, Z, name):
        if name == 'relu':
            return np.maximum(0, Z)
        elif name == 'leaky_relu':
            return np.where(Z > 0, Z, Z * 0.01)
        elif name == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif name == 'tanh':
            return np.tanh(Z)
        elif name == 'softmax':
            exp_Z = np.exp(Z - np.max(Z))
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        
        return Z
    
    def _derivative(self, Z, name):
        if name == 'relu':
            return (Z > 0).astype(float)
        elif name == 'leaky_relu':
            return np.where(Z > 0, 1, 0.01)
        elif name == 'sigmoid':
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)
        elif name == 'tanh':
            return 1 - np.tanh(Z) ** 2
        
    def forward(self, X):
        self.layers['A0'] = X

        for i in range(1, self.L - 1):
            Z = np.dot(self.parameters[f"W{i}"], self.layers[f"A{i-1}"]) + self.parameters[f"b{i}"]
            self.layers[f"Z{i}"] = Z
            self.layers[f"A{i}"] = self._activate(Z, self.activation_name)

        last_idx = self.L - 1
        Z_out = np.dot(self.parameters[f"W{last_idx}"], self.layers[f"A{last_idx-1}"]) + self.parameters[f"b{last_idx}"]
        self.layers[f"Z{last_idx}"] = Z_out
        self.layers[f"A{last_idx}"] = self._activate(Z_out, 'softmax')

    def backward(self, Y, learning_rate):
        m = Y.shape[1]
        gradients = {}

        last_idx = self.L - 1
        dZ = self.layers[f"A{last_idx}"] - Y

        gradients[f"dW{last_idx}"] = (1/m) * np.dot(dZ, self.layers[f"A{last_idx-1}"].T)
        gradients[f"db{last_idx}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        for i in range(last_idx - 1, 0, -1):
            dA = np.dot(self.parameters[f"W{i+1}"].T, dZ)
            dZ = dA * self._derivative(self.layers[f"Z{i}"], self.activation_name)

            gradients[f"dW{i}"] = (1/m) * np.dot(dZ, self.layers[f"A{i-1}"].T)
            gradients[f"db{i}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        for i in range(1, self.L):
            self.parameters[f"W{i}"] -= learning_rate * gradients[f"dW{i}"]
            self.parameters[f"b{i}"] -= learning_rate * gradients[f"db{i}"]

    def train(self, X, Y, epochs=100, lr=0.01, verbose=True):
        for i in range(epochs):
            self.forward(X)
            self.backward(Y, lr)
            
            if verbose and i % 10 == 0:
                predictions = np.argmax(self.layers[f"A{self.L - 1}"], axis=0)
                labels = np.argmax(Y, axis=0)
                accuracy = np.mean(predictions == labels)
                print(f"Epoch {i}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=0)
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)