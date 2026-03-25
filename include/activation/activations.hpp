#pragma once
#include "nn/types.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace activation {

/// @brief Supported activation functions for neural network layers.
enum class ActivationType {
	Sigmoid,
	Relu,
	Tanh,
	Softmax,
	None,
};

/// @brief Sigmoid activation function.
inline Vector sigmoid(const Vector& vec) {
	const auto f = [](auto z) { return 1.0 / (1.0 + std::exp(-z)); };
	return vec.unaryExpr(f);
}

/// @brief The derivative of Sigmoid activation function.
inline Vector d_sigmoid(const Vector& vec) {
	const auto f = [](auto z) {
		const auto exp_neg_z = std::exp(-z);
		return exp_neg_z / ((1 + exp_neg_z) * (1 + exp_neg_z));
	};
	return vec.unaryExpr(f);
}

/// @brief ReLU activation function.
inline Vector relu(const Vector& vec) {
	const auto f = [](auto z) { return std::max(z, 0.0); };
	return vec.unaryExpr(f);
}

/// @brief The derivative of ReLU activation function.
inline Vector d_relu(const Vector& vec) {
	const auto f = [](auto z) { return static_cast<decltype(z)>(z > 0); };
	return vec.unaryExpr(f);
}

/// @brief Tanh activation function.
inline Vector tanh(const Vector& vec) {
	const auto f = [](auto z) { return std::tanh(z); };
	return vec.unaryExpr(f);
}

/// @brief The derivative of Tanh activation function.
inline Vector d_tanh(const Vector& vec) {
	const auto f = [](auto z) {
		const auto tanh_z = std::tanh(z);
		return 1 - tanh_z * tanh_z;
	};
	return vec.unaryExpr(f);
}

/// @brief Softmax activation function.

inline Vector softmax(const Vector& vec) {
	const auto f  = [] (auto z) {
		return std::exp(z);
	};

	const Vector exp_vec = vec.unaryExpr(f);
	return exp_vec / exp_vec.sum();
}

/// @brief The derivative of Softmax activation function.

inline Vector d_softmax(const Vector& vec) {
	const Vector sm = softmax(vec);
	return sm.array() * (1.0 - sm.array());
}

} // namespace activation