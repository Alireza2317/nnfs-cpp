#pragma once
#include "nn/types.hpp"
#include <functional>

namespace loss {
enum class LossType { MSE, CrossEntropy };

using LossFn = std::function<double(const Vector&, const Vector&)>;
using LossDFn = std::function<Vector(const Vector&, const Vector&)>;

struct LossPair {
	LossFn f;
	LossDFn df;
};

/// @brief Calculates the Mean Squared Error between predicted and expected outputs.
inline double mse(const Vector& last_layer, const Vector& expected_output) {
	const Vector diff = last_layer.array() - expected_output.array();
	return diff.array().square().mean();
}

/// @brief Calculates the derivative of the Mean Squared Error with respect to the output layer.
inline Vector d_mse(const Vector& last_layer, const Vector& expected_output) {
	return 2 * (last_layer.array() - expected_output.array());
}

/// @brief Calculates the Cross-Entropy loss between predicted and expected outputs.
inline double cross_entropy(const Vector& last_layer, const Vector& expected_output) {
	// Avoiding log(0)
	const double epsilon = 1e-9;
	return -(expected_output.array() * (last_layer.array() + epsilon).log()).sum();
}

/// @brief Calculates the derivative of the Cross-Entropy loss with respect to the output layer.
inline Vector d_cross_entropy(const Vector& last_layer, const Vector& expected_output) {
	// Assuming softmax activation for the output layer!
	return last_layer.array() - expected_output.array();
}

/// @brief Factory function to retrieve the loss function and its derivative.
/// @param loss_type The enum type of the desired loss function.
/// @return A LossPair containing the function and its derivative.
inline LossPair get_loss_pair(const LossType& loss_type) {
	switch (loss_type) {
	case LossType::MSE:
		return LossPair{.f = mse, .df = d_mse};
	case LossType::CrossEntropy:
		return LossPair{.f = cross_entropy, .df = d_cross_entropy};
	}

	// Should be unreachable
	return LossPair{};
}
} // namespace loss
