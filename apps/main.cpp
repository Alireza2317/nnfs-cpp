#include "activation/activations.hpp"
#include "nn/network.hpp"
#include "nn/types.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

Eigen::MatrixXd load_csv(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: " + path);
	}

	std::string line;
	size_t row_count = 0;
	size_t col_count = 0;

	// First pass to only count the number of rows and cols
	if (std::getline(file, line)) {
		row_count = 1;

		std::stringstream ss(line);
		std::string cell;
		while (std::getline(ss, cell, ',')) {
			col_count++;
		}
	} else {
		throw std::runtime_error("File " + path + " was probably empty!");
	}

	while (std::getline(file, line)) {
		row_count++;
	}

	// Second pass to actually read the data
	file.clear();
	file.seekg(0);

	Eigen::MatrixXd matrix(row_count, col_count);
	size_t row = 0;

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string cell;
		size_t col = 0;
		while (std::getline(ss, cell, ',')) {
			try {
				matrix(row, col) = std::stod(cell);
			} catch (const std::invalid_argument& e) {
				std::println("Invalid value in cell!");
			}
			col++;
		}
		if (col != col_count) {
			throw std::runtime_error("CSV file has inconsistent number of columns.");
		}
		row++;
	}

	file.close();

	return matrix;
}

Eigen::MatrixXd one_hot_encode(const Eigen::MatrixXd& y, size_t num_classes) {
	const size_t n_samples = y.cols();

	Eigen::MatrixXd y_one_hot = Eigen::MatrixXd::Zero(num_classes, n_samples);

	for (size_t i = 0; i < n_samples; i++) {
		const size_t label = static_cast<size_t>(y(0, i));

		if (label >= 0 and label < num_classes) {
			y_one_hot(label, i) = 1.0;
		}
	}

	return y_one_hot;
}

void mnist() {
	std::println("Loading MNIST train dataset...");
	Eigen::MatrixXd train_dataset = load_csv("../data/mnist_train.csv");
	std::println("-> Loaded dataset with {} samples.", train_dataset.rows());

	// The dataset is currently of shape (num_samples, 1 + num_features)
	// Slice the dataset into features (X) and labels (y)
	// The first col is the label, and the rest are the 28*28 features

	Eigen::MatrixXd X_train = train_dataset.rightCols(train_dataset.cols() - 1).transpose();
	// X_train is of shape (num_features, num_samples)

	Eigen::MatrixXd y_train = one_hot_encode(train_dataset.leftCols(1).transpose(), 10);
	// y_train is of shape (num_outputs, num_samples), after one-hot encoding

	std::println("-> Sliced and reshaped data:");
	std::println("   X_train shape: ({} features, {} samples)", X_train.rows(), X_train.cols());
	std::println("   y_train shape: ({} output, {} samples)", y_train.rows(), y_train.cols());

	// Normalize the data
	X_train /= 255.0;

	std::vector<size_t> topology = {784, 16, 16, 10};
	std::vector<activation::ActivationType> acts = {
		activation::ActivationType::Relu,
		activation::ActivationType::Relu,
		activation::ActivationType::Sigmoid};

	NeuralNetwork nn = NeuralNetwork(topology, acts);

	nn.train(X_train, y_train, 0.5, false, 0.1, 100, 128, true);
}

void xor_dataset() {
	// 1. Create the XOR dataset
	// Input features (4 samples, 2 features each)
	Eigen::MatrixXd X_train(2, 4);
	X_train << 0, 0, 1, 1, // Feature 1
		0, 1, 0, 1;		   // Feature 2

	// Target labels (4 samples, 1 output each)
	Eigen::MatrixXd y_train(1, 4);
	y_train << 0, 1, 1, 0;

	std::println("XOR Dataset:");
	std::cout << "X_train (inputs):\n" << X_train.transpose() << "\n";
	std::cout << "y_train (labels):\n" << y_train.transpose() << "\n";
	std::println("------------------------------------");

	// 2. Define the Network Architecture
	std::vector<size_t> topology = {2, 3, 1}; // 2 inputs -> 3 hidden neurons -> 1 output
	std::vector<activation::ActivationType> activations = {
		activation::ActivationType::Tanh,	// Hidden layer
		activation::ActivationType::Sigmoid // Output layer
	};

	// 3. Create and Train the Neural Network
	NeuralNetwork nn(topology, activations);
	nn.set_seed(42); // Use a fixed seed for reproducible results

	std::println("Training network...");
	nn.train(X_train, y_train, 0.1, true, 0.0, 100, 4, true);
	std::println("------------------------------------");

	// 4. Test the Trained Network
	std::println("Testing results:");
	for (int i = 0; i < X_train.cols(); ++i) {
		Layer input_sample = X_train.col(i);
		Layer predicted_output = nn.predict(input_sample);
		std::println(
			"Input: [{}, {}] -> Predicted: {:.4f} (True Label: {})",
			input_sample(0),
			input_sample(1),
			predicted_output(0),
			y_train(0, i));
	}
}

int main() {
	mnist();
	return 0;
}