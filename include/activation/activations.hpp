#pragma once
#include "nn/types.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <cmath>

namespace activation {

/// @brief Supported activation functions for neural network layers.
enum class ActivationType {
	Sigmoid,
	Relu,
	Tanh,
	None,
};

/// @brief Sigmoid activation function.
inline Layer sigmoid(const Layer& vec) {
	const auto f = [](auto z) { return 1.0 / (1.0 + std::exp(-z)); };
	return vec.unaryExpr(f);
}

/// @brief The derivative of Sigmoid activation function.
inline Layer d_sigmoid(const Layer& vec) {
	const auto f = [](auto z) {
		const auto exp_neg_z = std::exp(-z);
		return exp_neg_z / ((1 + exp_neg_z) * (1 + exp_neg_z));
	};
	return vec.unaryExpr(f);
}

/// @brief ReLU activation function.
inline Layer relu(const Layer& vec) {
	const auto f = [](auto z) { return std::max(z, 0.0); };
	return vec.unaryExpr(f);
}

/// @brief The derivative of ReLU activation function.
inline Layer d_relu(const Layer& vec) {
	const auto f = [](auto z) { return static_cast<decltype(z)>(z > 0); };
	return vec.unaryExpr(f);
}

/// @brief Tanh activation function.
inline Layer tanh(const Layer& vec) {
	const auto f = [](auto z) { return std::tanh(z); };
	return vec.unaryExpr(f);
}

/// @brief The derivative of Tanh activation function.
inline Layer d_tanh(const Layer& vec) {
	const auto f = [](auto z) {
		const auto tanh_z = std::tanh(z);
		return 1 - tanh_z * tanh_z;
	};
	return vec.unaryExpr(f);
}

} // namespace activation