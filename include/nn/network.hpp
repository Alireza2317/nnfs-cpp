#pragma once
#include "activation/activations.hpp"
#include "nn/loss.hpp"
#include "nn/types.hpp"
#include <Eigen/Dense>
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
	/// @param loss The loss function to be used.
	explicit NeuralNetwork(
		std::span<const size_t> topology,
		std::span<const activation::ActivationType> activations,
		loss::LossType loss,
		size_t seed = 0);

	/// @brief Constructs a neural network with a global activation type for all layers.
	/// @param topology List of layer sizes (input, hidden..., output).
	/// @param global_activation Activation type for all layers (excluding input layer).
	/// @param loss The loss function to be used.
	explicit NeuralNetwork(
		std::span<const size_t> topology,
		const activation::ActivationType& global_activation,
		loss::LossType loss,
		size_t seed = 0);

	/// @brief Setting the seed for the random generations.
	void set_seed(const size_t seed = 0) const;

	/// @brief Setting the seed  random generations.
	void set_seed();

	/// @brief Performs a forward pass through the network using the current input layer.
	void feed_forward();

	/// @brief Performs a forward pass through the network using the given input layer.
	void feed_forward(const Vector& input_layer);

	/// @brief Calculates the cost of a single sample.
	double cost_single_sample(const Vector& sample, const Vector& output);

	/// @brief Calculates cost of the given data.
	double cost(const Matrix& X, const Matrix& y);

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
		const Matrix& X_train,
		const Matrix& y_train,
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
	double accuracy(const Matrix& features, const Eigen::VectorXd& labels);

	/// @brief Predicts the output for a given input layer.
	Vector predict(const Vector& X);

	/// @brief Predicts the class index for a given input layer.
	size_t predict_class(const Vector& X);

  private:
	/// @brief Number of network layers (excluding input layer).
	size_t m_N_LAYERS;

	/// @brief The random seed used throughout random generations.
	mutable size_t m_seed = 0;

	/// @brief The loss function used for training.
	loss::LossType m_loss_type;

	/// @brief Pair of loss function and its derivative.
	loss::LossPair m_loss_f_df_pair;

	/// @brief Activation types for each layer (excluding input layer).
	std::vector<activation::ActivationType> m_activation_types;

	/// @brief Pairs of activation functions and their derivatives for each layer.
	std::vector<ActivationPair> m_activation_f_df_pairs;

	/// @brief Shapes of the weights matrices for each layer.
	std::vector<Shape> m_weights_shapes;

	/// @brief Shapes of the biases vectors for each layer.
	std::vector<Shape> m_biases_shapes;

	/// @brief Weights vectors for each layer.
	std::vector<Matrix> m_weights;

	/// @brief Biases vectors for each layer.
	std::vector<Matrix> m_biases;

	/// @brief Topology of the network (layer sizes).
	std::vector<size_t> m_topology;

	/// @brief Input layer activations.
	Vector m_input_layer;

	/// @brief Pre-activation values (z) for each layer.
	std::vector<Vector> m_z_layers;

	/// @brief Activations for each layer.
	std::vector<Vector> m_layers;

	/// @brief Initializes random weights and biases.
	void init_rnd_weights_biases();

	/// @brief Initializes all neuron activations and pre-activations.
	void init_all_neurons();

	/// @brief Sets up activation functions for each layer.
	void setup_activations();

	/// @brief Sets up the loss function.
	void setup_loss();

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
	void backpropagate_one(const Matrix& sample, const Matrix& output, Gradients& out_gradients);

	/// @brief Performs backpropagation for a batch of samples.
	///
	/// All the calculus of backpropagation happens here.
	/// Calculates the derivative of the cost w.r.t. weights and biases.
	/// @param X_train Batch input data.
	/// @param y_train Batch target data.
	/// @param out_gradients A reference to a Gradients object to be filled with the results.
	void backpropagate(const Matrix& X_train, const Matrix& y_train, Gradients& out_gradients);
};
