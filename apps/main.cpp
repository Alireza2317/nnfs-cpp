#include "activation/activations.hpp"
#include "nn/loss.hpp"
#include "nn/network.hpp"
#include "nn/types.hpp"
#include <Eigen/Dense>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <print>
#include <span>
#include <sstream>
#include <stdexcept>

Matrix load_csv(const char* path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: ");
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
		throw std::runtime_error("File was probably empty!");
	}

	while (std::getline(file, line)) {
		row_count++;
	}

	// Second pass to actually read the data
	file.clear();
	file.seekg(0);

	Matrix matrix(row_count, col_count);
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

Matrix one_hot_encode(const Matrix& y, size_t num_classes) {
	const size_t n_samples = y.cols();

	Matrix y_one_hot = Matrix::Zero(num_classes, n_samples);

	for (size_t i = 0; i < n_samples; i++) {
		const size_t label = static_cast<size_t>(y(0, i));

		if (label >= 0 and label < num_classes) {
			y_one_hot(label, i) = 1.0;
		}
	}

	return y_one_hot;
}

void print_confusion_matrix(const Matrix& confusion_matrix, const std::span<std::string>& labels) {
	size_t num_classes = labels.size();

	// Print header for predicted classes
	std::cout << std::setw(10) << " "; // Empty corner
	for (size_t i = 0; i < num_classes; ++i) {
		std::cout << std::setw(10) << "Pred " + labels[i];
	}
	std::cout << std::endl;

	// Print separator
	std::cout << std::string(10, ' ');
	for (size_t i = 0; i < num_classes; ++i) {
		std::cout << std::string(10, '-');
	}
	std::cout << std::endl;

	// Print rows for actual classes
	for (size_t i = 0; i < num_classes; ++i) {
		std::cout << std::setw(10) << "Actual " + labels[i];
		for (size_t j = 0; j < num_classes; ++j) {
			std::cout << std::setw(10) << static_cast<int>(confusion_matrix(i, j));
		}
		std::cout << std::endl;
	}
}

void mnist() {
	std::println("Loading MNIST train dataset...");
	Matrix train_dataset = load_csv("../data/mnist_train_mini.csv");
	std::println("-> Loaded dataset with {} samples.", train_dataset.rows());

	// The dataset is currently of shape (num_samples, 1 + num_features)
	// Slice the dataset into features (X) and labels (y)
	// The first col is the label, and the rest are the 28*28 features

	Matrix X_train = train_dataset.rightCols(train_dataset.cols() - 1).transpose();
	// X_train is of shape (num_features, num_samples)

	Matrix y_train = one_hot_encode(train_dataset.leftCols(1).transpose(), 10);
	// y_train is of shape (num_outputs, num_samples), after one-hot encoding

	std::println("-> Sliced and reshaped data:");
	std::println("   X_train shape: ({} features, {} samples)", X_train.rows(), X_train.cols());
	std::println("   y_train shape: ({} output, {} samples)", y_train.rows(), y_train.cols());

	// Normalize the data
	X_train /= 255.0;

	size_t topology[] = {784, 16, 16, 10};
	activation::ActivationType acts[] = {
		activation::ActivationType::Relu,
		activation::ActivationType::Relu,
		activation::ActivationType::Softmax};

	NeuralNetwork nn = NeuralNetwork(topology, acts, loss::LossType::CrossEntropy, 42);

	nn.train(X_train, y_train, 0.8, false, 0.1, 1, 128, true);

	std::println("---- Training Finished ----");

	std::println("\nLoading MNIST test dataset...");
	Matrix test_dataset = load_csv("../data/mnist_test.csv");
	std::println("-> Loaded dataset with {} samples.", test_dataset.rows());

	Matrix X_test = test_dataset.rightCols(test_dataset.cols() - 1).transpose();
	Eigen::VectorXd y_test_labels = test_dataset.leftCols(1).col(0);

	X_test /= 255.0;

	const double test_accuracy = nn.accuracy(X_test, y_test_labels) * 100.0;
	std::println("\nTest Accuracy: {:.2f}%", test_accuracy);

	std::array<std::string, 10> labels;
	for (size_t i = 0; i < 10; ++i) {
		labels.at(i) = std::to_string(i);
	}

	std::println("\nConfusion Matrix:");
	Matrix confusion_matrix = nn.calculate_confusion_matrix(X_test, y_test_labels);
	print_confusion_matrix(confusion_matrix, labels);
}

int main() {
	mnist();

	return 0;
}