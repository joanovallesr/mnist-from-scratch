import numpy as np
from model import NeuralNetwork
import os
import urllib.request

def load_mnist_data():
    print("Loading MNIST dataset...")
    path = 'mnist.npz'

    if not os.path.exists(path):
        url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")

    print("Loaing data into memory...")
    with np.load(path) as f:
        X_train, y_train = f['x_train'], f['y_train']
        X_test, y_test = f['x_test'], f['y_test']

    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    X = X_all.reshape(X_all.shape[0], -1).T / 255.0  # Normalize and flatten
    Y = np.eye(10)[y_all].T  # One-hot encode labels

    return X, Y

def main():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    X, Y = load_mnist_data()

    train_size = 10000
    X_train, Y_train = X[:, :train_size], Y[:, :train_size]

    activations = ['relu', 'leaky_relu', 'tanh', 'sigmoid']

    for act in activations:
        print(f"\nTraining with activation function: {act}")
        nn = NeuralNetwork(input_size=784, hidden_layers=[128, 64], output_size=10, activation=act)
        nn.train(X_train, Y_train, epochs=200, lr=0.01)

        save_path = f'models/model_{act}.pkl'
        nn.save_model(save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()