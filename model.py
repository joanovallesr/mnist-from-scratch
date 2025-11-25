import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation="relu"):
        self.activation_name = activation
        # Architecture: [784, 128, 64, 10]
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.params = {}
        self.cache = {}
        self.L = len(self.layer_sizes) - 1 # Number of layers (excluding input)
        self._initialize_parameters()

    def _initialize_parameters(self):
        # He Initialization for ReLU, Xavier for others
        np.random.seed(42) # For reproducibility
        for l in range(1, self.L + 1):
            # Choose scale based on activation
            if self.activation_name in ['relu', 'leaky_relu']:
                scale = np.sqrt(2. / self.layer_sizes[l-1])
            else:
                scale = np.sqrt(1. / self.layer_sizes[l-1])
            
            self.params['W' + str(l)] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * scale
            self.params['b' + str(l)] = np.zeros((self.layer_sizes[l], 1))

    # --- Activation Functions ---
    def _activate(self, Z, name):
        if name == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif name == "tanh":
            return np.tanh(Z)
        elif name == "relu":
            return np.maximum(0, Z)
        elif name == "leaky_relu":
            return np.where(Z > 0, Z, Z * 0.01)
        elif name == "softmax":
            # Numerically stable softmax
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return Z

    def _derivative(self, Z, name):
        if name == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)
        elif name == "tanh":
            return 1 - np.power(np.tanh(Z), 2)
        elif name == "relu":
            return (Z > 0).astype(float)
        elif name == "leaky_relu":
            return np.where(Z > 0, 1, 0.01)
        return 1

    def forward(self, X):
        self.cache['A0'] = X
        
        # Hidden Layers
        for l in range(1, self.L):
            Z = np.dot(self.params['W' + str(l)], self.cache['A' + str(l-1)]) + self.params['b' + str(l)]
            self.cache['Z' + str(l)] = Z
            self.cache['A' + str(l)] = self._activate(Z, self.activation_name)
            
        # Output Layer (Softmax)
        Z_out = np.dot(self.params['W' + str(self.L)], self.cache['A' + str(self.L-1)]) + self.params['b' + str(self.L)]
        self.cache['Z' + str(self.L)] = Z_out
        self.cache['A' + str(self.L)] = self._activate(Z_out, "softmax")
        
        return self.cache['A' + str(self.L)]

    def backward(self, Y, learning_rate):
        m = Y.shape[1]
        grads = {}
        
        # Output Layer Gradient (Softmax + Cross Entropy derivative simplifies to A - Y)
        dZ = self.cache['A' + str(self.L)] - Y
        
        grads['dW' + str(self.L)] = (1/m) * np.dot(dZ, self.cache['A' + str(self.L-1)].T)
        grads['db' + str(self.L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Backpropagate through hidden layers
        for l in range(self.L - 1, 0, -1):
            dA = np.dot(self.params['W' + str(l+1)].T, dZ)
            dZ = dA * self._derivative(self.cache['Z' + str(l)], self.activation_name)
            
            grads['dW' + str(l)] = (1/m) * np.dot(dZ, self.cache['A' + str(l-1)].T)
            grads['db' + str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
        # Update Parameters (Gradient Descent)
        for l in range(1, self.L + 1):
            self.params['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.params['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    def train(self, X, Y, epochs=100, lr=0.1, verbose=True):
        for i in range(epochs):
            self.forward(X)
            self.backward(Y, lr)
            
            if verbose and i % 20 == 0:
                predictions = np.argmax(self.cache['A' + str(self.L)], axis=0)
                labels = np.argmax(Y, axis=0)
                acc = np.mean(predictions == labels)
                print(f"Epoch {i}: Accuracy: {acc:.4f}")

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        probs = self.forward(X)
        return np.argmax(probs, axis=0)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)