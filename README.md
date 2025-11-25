# MNIST Digit Recognizer (Built from Scratch)

A fully functional Neural Network classifier built entirely from scratch using **Python** and **NumPy**. This project demonstrates the mathematics of Deep Learning—including forward propagation, backpropagation, and gradient descent—without relying on high-level frameworks like TensorFlow or PyTorch for the core logic.

It includes an interactive **Streamlit** web application where users can draw digits and see real-time predictions.

## Features
- **Pure NumPy Implementation:** No black-box ML libraries for the model logic.
- **Customizable Architecture:** Dynamic support for different layer sizes and activations.
- **Multiple Activations:** Implements **ReLU**, **Leaky ReLU**, **Tanh**, and **Sigmoid** with their respective derivatives.
- **Robust Training:** Uses **He Initialization** and **Numerical Stability** fixes (e.g., for Softmax) to ensure high accuracy (~95%).
- **Interactive Demo:** A drawing canvas to test the model in real-time.

## Mathematical Concepts Implemented
This project manually implements the calculus behind neural networks:
* **Linear Transformation:** $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$
* **Activation Functions:** $\text{ReLU}(z) = \max(0, z)$, $\sigma(z) = \frac{1}{1+e^{-z}}$, etc.
* **Softmax Output:** Stable implementation using $e^{z_i - \max(z)}$ to prevent overflow.
* **Backpropagation:** Manually calculated gradients using the Chain Rule:
    $$dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}$$
    $$db^{[l]} = \frac{1}{m} \sum dZ^{[l]}$$
* **Optimization:** Stochastic Gradient Descent (SGD).
