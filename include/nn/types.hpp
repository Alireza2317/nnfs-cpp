#pragma once
#include <Eigen/Dense>
#include <array>
#include <functional>

/// @brief Shape type for weights and biases (rows, cols).
using Shape = std::array<size_t, 2>;

/// @brief Type representing a vector, used for activations or pre-activations of a layer.
using Vector = Eigen::VectorXd;

/// @brief Type representing a matrix, used for weights and biases and gradients.
using Matrix = Eigen::MatrixXd;

/// @brief A pair of activation function and its derivative.
struct ActivationPair {
	std::function<Vector(const Vector&)> f;
	std::function<Vector(const Vector&)> df;
};

/// @brief Structure to hold gradients for ALL weights and biases.
struct Gradients {
	std::vector<Matrix> dWs;
	std::vector<Vector> dBs;

	/// @brief Default constructor.
	Gradients() = default;

	/// @brief Construct Gradients initialized with zero matrices based on provided shapes.
	/// @param weights_shapes A vector of shapes for the weight matrices.
	/// @param biases_shapes A vector of shapes for the bias vectors.
	explicit Gradients(
		const std::vector<Shape>& weights_shapes, const std::vector<Shape>& biases_shapes) {

		const size_t N_LAYERS = weights_shapes.size();
		dWs.reserve(N_LAYERS);
		dBs.reserve(N_LAYERS);

		for (const Shape& shape : weights_shapes) {
			dWs.emplace_back(Matrix::Zero(shape.at(0), shape.at(1)));
		}

		for (const Shape& shape : biases_shapes) {
			dBs.emplace_back(Vector::Zero(shape.at(0)));
		}
	}

	explicit Gradients(std::vector<Matrix>&& dWs, std::vector<Vector>&& dBs) noexcept {
		this->dWs = std::move(dWs);
		this->dBs = std::move(dBs);
	}
};