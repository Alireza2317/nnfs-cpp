#pragma once
#include "activation/activations.hpp"
#include "nn/types.hpp"
#include <Eigen/Dense>
#include <cstdlib>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <vector>

/// @brief A standard fully-connected neural network.
///
/// This class implements a neural network with customizable topology and activation functions.
class NeuralNetwork {
  public:
	/// @brief Constructs a neural network with per-layer activation types.
	/// @param topology List of layer sizes (input, hidden..., output).
	/// @param activations List of activation types for each layer (excluding input layer).
	explicit NeuralNetwork(
		std::span<const size_t> topology,
		std::span<const activation::ActivationType> activations,
		size_t seed = 0);

	/// @brief Constructs a neural network with a global activation type for all layers.
	/// @param topology List of layer sizes (input, hidden..., output).
	/// @param global_activation Activation type for all layers (excluding input layer).
	explicit NeuralNetwork(
		std::span<const size_t> topology,
		const activation::ActivationType& global_activation,
		size_t seed = 0);

	/// @brief Setting the seed for the random generations.
	void set_seed(const size_t seed = 0) const;

	/// @brief Setting the seed  random generations.
	void set_seed();

	/// @brief Performs a forward pass through the network using the current input layer.
	void feed_forward();

	/// @brief Performs a forward pass through the network using the given input layer.
	void feed_forward(const Layer& input_layer);

	/// @brief Calculates the Mean Squared Error cost of a single sample.
	double cost_MSE_single_sample(const Layer& sample, const Layer& output);

	/// @brief Calculates MSE cost of the given data.
	double cost_MSE(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y);

	/// @brief Trains the neural network using the provided training data.
	/// @param X_train Input data matrix of shape (input_dim, num_samples).
	/// @param y_train Target data matrix of shape (output_dim, num_samples).
	/// @param lr Learning rate for the optimizer (default: 0.1).
	/// @param const_lr If true, keeps the learning rate constant; otherwise, applies decay
	/// (default: false).
	/// @param decay_rate Decay rate for the learning rate if const_lr is false (default: 0.1).
	/// @param n_epochs Number of training epochs (default: 100).
	/// @param batch_size Size of each mini-batch (default: 32).
	/// @param verbose If true, prints training progress (default: false).
	void train(
		const Eigen::MatrixXd& X_train,
		const Eigen::MatrixXd& y_train,
		const double& lr = 0.1,
		const bool& const_lr = false,
		const double& decay_rate = 0.1,
		const size_t& n_epochs = 100,
		const size_t& batch_size = 32,
		const bool& verbose = false);

	/// @brief Calculates the accuracy of the network on the given dataset.
	/// @param features Input data matrix of shape (input_dim, num_samples).
	/// @param labels Vector containing the true class index for each sample.
	/// @note This method is only suitable for classification problems.
	///
	/// @return The fraction of correctly predicted classes.
	double accuracy(const Eigen::MatrixXd& features, const Eigen::VectorXd& labels);

	/// @brief Predicts the output for a given input layer.
	Layer predict(const Layer& X);

	/// @brief Predicts the class index for a given input layer.
	size_t predict_class(const Layer& X);

  private:
	/// @brief Number of network layers (excluding input layer).
	size_t m_N_LAYERS;

	/// @brief The random seed used throughout random generations.
	mutable size_t m_seed = 0;

	/// @brief Activation types for each layer (excluding input layer).
	std::vector<activation::ActivationType> m_activation_types;

	/// @brief Pairs of activation functions and their derivatives for each layer.
	std::vector<ActivationPair> m_activation_f_df_pairs;

	/// @brief Shapes of the weights matrices for each layer.
	std::vector<Shape> m_weights_shapes;

	/// @brief Shapes of the biases vectors for each layer.
	std::vector<Shape> m_biases_shapes;

	/// @brief Weights vectors for each layer.
	std::vector<Eigen::MatrixXd> m_weights;

	/// @brief Biases vectors for each layer.
	std::vector<Eigen::MatrixXd> m_biases;

	/// @brief Topology of the network (layer sizes).
	std::vector<size_t> m_topology;

	/// @brief Input layer activations.
	Layer m_input_layer;

	/// @brief Pre-activation values (z) for each layer.
	std::vector<Layer> m_z_layers;

	/// @brief Activations for each layer.
	std::vector<Layer> m_layers;

	/// @brief Initializes random weights and biases.
	void init_rnd_weights_biases();

	/// @brief Initializes all neuron activations and pre-activations.
	void init_all_neurons();

	/// @brief Sets up activation functions for each layer.
	void setup_activations();

	/// @brief Calculates the shape of all weights and biases.
	///
	/// This will populate the m_weights_shapes, and m_biases_shapes attributes.
	void calculate_weights_biases_shapes();

	/// @brief Update the weights and biases based on the given gradients.
	void update_weights_biases(const Gradients& gradients, const double& lr);

	/// @brief Performs backpropagation for a single sample.
	/// @param sample Input sample vector.
	/// @param output Target output vector.
	/// @param gradients An accumulator to which the new gradients will be ADDED.
	void backpropagate_one(
		const Eigen::MatrixXd& sample, const Eigen::MatrixXd& output, Gradients& out_gradients);

	/// @brief Performs backpropagation for a batch of samples.
	///
	/// All the calculus of backpropagation happens here.
	/// Calculates the derivative of the cost w.r.t. weights and biases.
	/// @param X_train Batch input data.
	/// @param y_train Batch target data.
	/// @param out_gradients A reference to a Gradients object to be filled with the results.
	void backpropagate(
		const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& y_train, Gradients& out_gradients);
};

// --- IMPLEMENTATION ---

inline NeuralNetwork::NeuralNetwork(
	std::span<const size_t> topology,
	std::span<const activation::ActivationType> activations,
	size_t seed)
	: m_topology(topology.begin(), topology.end()),
	  m_activation_types(activations.begin(), activations.end()), m_N_LAYERS(topology.size() - 1) {

	if (topology.size() < 3) {
		throw std::invalid_argument(
			"Neural network must have at least 3 layers (input, at least one hidden, and output).");
	}

	set_seed(seed);

	calculate_weights_biases_shapes();

	init_rnd_weights_biases();

	init_all_neurons();

	setup_activations();
}

inline NeuralNetwork::NeuralNetwork(
	std::span<const size_t> topology,
	const activation::ActivationType& global_activation,
	size_t seed)
	: NeuralNetwork::NeuralNetwork(
		  topology,
		  std::vector<activation::ActivationType>(topology.size() - 1, global_activation),
		  seed) {
}

inline void NeuralNetwork::set_seed(const size_t seed) const {
	if (seed == 0) {
		m_seed = std::random_device{}();
	} else {
		m_seed = seed;
	}
}

inline void NeuralNetwork::calculate_weights_biases_shapes() {
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		m_weights_shapes.push_back(Shape{m_topology.at(i + 1), m_topology.at(i)});
		m_biases_shapes.push_back(Shape{m_topology.at(i + 1), 1});
	}
}

inline void NeuralNetwork::init_rnd_weights_biases() {
	std::srand(m_seed);

	for (const Shape& shape : m_weights_shapes) {
		m_weights.push_back(Eigen::MatrixXd::Random(shape.at(0), shape.at(1)) * 0.05);
	}

	for (const Shape& shape : m_biases_shapes) {
		m_biases.push_back(Eigen::MatrixXd::Random(shape.at(0), shape.at(1)) * 0.01);
	}
}

inline void NeuralNetwork::init_all_neurons() {
	m_input_layer = Eigen::VectorXd::Zero(m_topology.at(0));

	for (const size_t& layer_size : std::span{m_topology.begin() + 1, m_topology.end()}) {
		m_z_layers.push_back(Eigen::VectorXd::Zero(layer_size));
		m_layers.push_back(Eigen::VectorXd::Zero(layer_size));
	}
}

inline void NeuralNetwork::setup_activations() {
	for (const activation::ActivationType& activation_type : m_activation_types) {
		switch (activation_type) {

		case activation::ActivationType::Sigmoid:
			m_activation_f_df_pairs.push_back(
				ActivationPair{.f = activation::sigmoid, .df = activation::d_sigmoid});
			break;
		case activation::ActivationType::Relu:
			m_activation_f_df_pairs.push_back(
				ActivationPair{.f = activation::relu, .df = activation::d_relu});
			break;
		case activation::ActivationType::Tanh:
			m_activation_f_df_pairs.push_back(
				ActivationPair{.f = activation::tanh, .df = activation::d_tanh});
			break;
		case activation::ActivationType::None:
			m_activation_f_df_pairs.push_back(ActivationPair{
				.f = [](const Layer& v) { return v; },
				.df = [](const Layer& v) { return Layer::Ones(v.size()); }});
			break;
		}
	}
}

inline void NeuralNetwork::feed_forward() {
	NeuralNetwork::feed_forward(m_input_layer);
}

inline void NeuralNetwork::feed_forward(const Layer& input_layer) {
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		// Connection of input layer and the first hidden layer
		if (i == 0) {
			m_z_layers.at(i) = (m_weights.at(0) * input_layer) + m_biases.at(0);
		} else {
			m_z_layers.at(i) = (m_weights.at(i) * m_layers.at(i - 1)) + m_biases.at(i);
		}

		m_layers.at(i) = m_activation_f_df_pairs.at(i).f(m_z_layers.at(i));
	}
}

inline void NeuralNetwork::update_weights_biases(const Gradients& gradients, const double& lr) {
	for (auto&& [weights_matrix, dW] : std::ranges::views::zip(m_weights, gradients.dWs)) {
		weights_matrix -= lr * dW;
	}

	for (auto&& [biases_matrix, dB] : std::ranges::views::zip(m_biases, gradients.dBs)) {
		biases_matrix -= lr * dB;
	}
}

inline double NeuralNetwork::cost_MSE_single_sample(const Layer& sample, const Layer& output) {

	feed_forward(sample);

	const Layer& last_layer = m_layers.back();

	const Layer diff = last_layer.array() - output.array();

	return diff.array().square().mean();
}

inline double NeuralNetwork::cost_MSE(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
	double mse = 0.0;

	for (size_t i = 0; i < X.cols(); i++) {
		const auto& sample = X.col(i);
		const auto& true_output = y.col(i);

		mse += cost_MSE_single_sample(sample, true_output);
	}

	// Divide by the number of samples
	mse /= X.cols();

	return mse;
}

inline void NeuralNetwork::backpropagate_one(
	const Eigen::MatrixXd& sample, const Eigen::MatrixXd& output, Gradients& out_gradients) {

	const Layer& input_layer = sample.col(0);
	const Layer& output_layer = output.col(0);

	feed_forward(input_layer);

	// Derivative of the Mean Squared cost w.r.t. the output layer
	const Eigen::MatrixXd d_cost_p_ol = 2 * (m_layers.back().array() - output_layer.array());

	// Error of the output layer
	const Eigen::MatrixXd error_ol =
		m_activation_f_df_pairs.back().df(m_z_layers.back()).array() * d_cost_p_ol.array();

	// Errors of all hidden layers and the output layer, in order.
	std::vector<Eigen::MatrixXd> errors(m_N_LAYERS);
	errors.at(errors.size() - 1) = error_ol;

	for (int i = m_N_LAYERS - 2; i >= 0; i--) {
		errors.at(i) = m_activation_f_df_pairs[i].df(m_z_layers[i]).array() *
					   (m_weights.at(i + 1).transpose() * errors.at(i + 1)).array();
	}

	// Based on the equations of backprpoagation we know that
	// The derivative of cost w.r.t. bias of each layer
	// Is actually equal to the error of that layer.
	const std::vector<Eigen::MatrixXd>& d_cost_p_biases = errors;

	// Adding it to the gradients dBs
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		out_gradients.dBs[i] += d_cost_p_biases[i];
	}

	// Based on the equations of backpropagation
	// The derivative of the cost w.r.t. the weights of the layer l
	// Is the matrix mult of `error of layer l` and `activation of layer (l-1) transposed`.
	std::vector<Eigen::MatrixXd> d_cost_p_weights(m_N_LAYERS);

	d_cost_p_weights.at(0) = errors[0] * m_input_layer.transpose();
	for (size_t i = 1; i < m_N_LAYERS; i++) {
		d_cost_p_weights.at(i) = errors[i] * m_layers[i - 1].transpose();
	}

	// Adding it to the gradients dWs
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		out_gradients.dWs[i] += d_cost_p_weights[i];
	}
}

inline void NeuralNetwork::backpropagate(
	const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& y_train, Gradients& out_gradients) {

	// Reset gradients to zero before accumulating
	for (auto& dW : out_gradients.dWs) {
		dW.setZero();
	}
	for (auto& dB : out_gradients.dBs) {
		dB.setZero();
	}

	const size_t N_SAMPLES = X_train.cols();
	if (N_SAMPLES == 0) {
		return;
	}

	for (size_t i = 0; i < N_SAMPLES; i++) {
		const Eigen::MatrixXd& sample = X_train.col(i);
		const Eigen::MatrixXd& output = y_train.col(i);

		// Accumulate gradients
		backpropagate_one(sample, output, out_gradients);
	}

	// Now gradients holds the sum of all the derivatives
	// The average should be calculated
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		out_gradients.dWs[i] /= N_SAMPLES;
		out_gradients.dBs[i] /= N_SAMPLES;
	}
}

inline void NeuralNetwork::train(
	const Eigen::MatrixXd& X_train,
	const Eigen::MatrixXd& y_train,
	const double& lr,
	const bool& const_lr,
	const double& decay_rate,
	const size_t& n_epochs,
	const size_t& batch_size,
	const bool& verbose) {

	//* Assuming m training samples
	//* X_train.shape = (topology[0], m)
	//* y_train.shape = (topology[-1], m)
	const size_t N_SAMPLES = X_train.cols();

	// Shuffle the training data to avoid biases in the training
	// Shuffling along columns
	std::vector<size_t> shuffled_indices(N_SAMPLES);
	std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
	std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::mt19937{m_seed});

	const size_t num_batches = N_SAMPLES / batch_size;
	const double INITIAL_LR = lr;
	double current_lr = INITIAL_LR;

	Gradients gradients(m_weights_shapes, m_biases_shapes);

	for (size_t epoch = 0; epoch < n_epochs; epoch++) {
		for (size_t batch_i = 0; batch_i < num_batches; batch_i++) {
			const size_t start_col = batch_i * batch_size;
			const size_t current_batch_size =
				(batch_i != num_batches - 1) ? batch_size : (N_SAMPLES - start_col);

			std::vector<size_t> batch_indices(current_batch_size);
			for (size_t i = 0; i < current_batch_size; i++) {
				batch_indices[i] = shuffled_indices[start_col + i];
			}

			const Eigen::MatrixXd batch_x = X_train(Eigen::all, batch_indices);
			const Eigen::MatrixXd batch_y = y_train(Eigen::all, batch_indices);

			backpropagate(batch_x, batch_y, gradients);

			if (not const_lr) {
				current_lr = INITIAL_LR * std::exp(-decay_rate * epoch);
			}

			update_weights_biases(gradients, current_lr);

			if (verbose and batch_i == num_batches - 1) {
				const double train_cost = cost_MSE(X_train, y_train);

				std::println(
					"epoch: {:03}/{:03} ~ training_cost={:.4f}",
					epoch + 1,
					n_epochs,
					train_cost);
			}
		}
	}
}

inline double
NeuralNetwork::accuracy(const Eigen::MatrixXd& features, const Eigen::VectorXd& labels) {
	size_t total_samples = labels.size();
	size_t correct_predictions = 0;

	for (size_t i = 0; i < total_samples; i++) {
		const auto& sample = features.col(i);
		const auto& true_class = labels(i);
		const auto& predicted_class = predict_class(sample);
		if (predicted_class == true_class) {
			correct_predictions++;
		}
	}

	return static_cast<double>(correct_predictions) / total_samples;
}

inline Layer NeuralNetwork::predict(const Layer& X) {
	feed_forward(X);
	return m_layers.back();
}

inline size_t NeuralNetwork::predict_class(const Layer& X) {
	Layer predicted_output = predict(X);

	size_t max_index;
	predicted_output.maxCoeff(&max_index);
	return max_index;
}
