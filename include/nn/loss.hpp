#pragma once
#include "nn/types.hpp"
#include <functional>

namespace loss {
enum class LossType { MSE };

using LossFn = std::function<double(const Vector&, const Vector&)>;
using LossDFn = std::function<Vector(const Vector&, const Vector&)>;

struct LossPair {
	LossFn f;
	LossDFn df;
};

inline double mse(const Vector& last_layer, const Vector& expected_output) {
	const Vector diff = last_layer.array() - expected_output.array();
	return diff.array().square().mean();
}

inline Vector d_mse(const Vector& last_layer, const Vector& expected_output) {
	return 2 * (last_layer.array() - expected_output.array());
}


inline LossPair get_loss_pair(const LossType& loss_type) {
	switch (loss_type) {
	case LossType::MSE:
		return LossPair{.f = mse, .df = d_mse};
	}
}
} // namespace loss
