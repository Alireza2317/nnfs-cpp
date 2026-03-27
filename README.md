# Neural Network from Scratch

This repository contains a C++ implementation of a fully-connected neural network built from scratch. The goal of this project is to provide a clear and efficient understanding of the fundamental concepts behind neural networks, without relying on high-level machine learning frameworks.

## Features

The current implementation includes the following key features:

-   **Customizable Network Topology:** Define the number of layers and neurons in each layer to build various network architectures.
-   **Multiple Activation Functions:** Support for common activation functions such as Sigmoid, ReLU, Tanh, and Softmax.
-   **Loss Functions:** Implements Mean Squared Error (MSE) and Cross-Entropy for different types of learning tasks.
-   **Training Capabilities:**
    -   Mini-batch gradient descent for efficient training.
    -   Configurable learning rate with optional decay.
    -   Epoch-based training with verbose output.
-   **Model Evaluation:**
    -   Calculate accuracy for classification tasks.
    -   Generate confusion matrices to assess model performance.
-   **Prediction:**
    -   Predict outputs for single samples or batches of data.
    -   Predict class labels for classification problems.
-   **Serialization:** Save and load trained network parameters to and from binary files, allowing for persistent models.
-   **Eigen Integration:** Leverages the Eigen library for high-performance matrix and vector operations.

## Getting Started

To build and run this project, you will need a C++23 compatible compiler and CMake. The Eigen library is a required dependency.

1. Clone the repository:
```bash
git clone https://github.com/Alireza2317/nn-from-scratch.git
```

You can use the `NeuralNetwork` class as a library in your own C++ projects. Ensure your data is converted into Eigen matrices (`Eigen::MatrixXd` or `Eigen::VectorXd`) before passing it to the network.

Alternatively, to test the provided examples, you can download the MNIST dataset from Kaggle: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. Place the `mnist_train.csv` and `mnist_test.csv` files into a `data` directory at the root of this project.

To build and run the project:

```bash
mkdir build
cd build
cmake ..
make
./main # the name of your executable
```


## Future Enhancements
More features and optimizations are planned for future releases, including additional activation functions, optimizers, and network architectures.