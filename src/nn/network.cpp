#include "nn/network.hpp"
#include "activation/activations.hpp"
#include "nn/types.hpp"
#include <print>
#include <random>
#include <ranges>

NeuralNetwork::NeuralNetwork(
	std::span<const size_t> topology,
	std::span<const activation::ActivationType> activations,
	loss::LossType loss,
	size_t seed)
	: m_topology(topology.begin(), topology.end()),
	  m_activation_types(activations.begin(), activations.end()), m_N_LAYERS(topology.size() - 1),
	  m_loss_type(loss) {

	if (topology.size() < 3) {
		throw std::invalid_argument(
			"Neural network must have at least 3 layers (input, at least one hidden, and output).");
	}

	set_seed(seed);

	calculate_weights_biases_shapes();

	init_rnd_weights_biases();

	init_all_neurons();

	setup_activations();

	setup_loss();
}

NeuralNetwork::NeuralNetwork(
	std::span<const size_t> topology,
	const activation::ActivationType& global_activation,
	loss::LossType loss,
	size_t seed)
	: NeuralNetwork::NeuralNetwork(
		  topology,
		  std::vector<activation::ActivationType>(topology.size() - 1, global_activation),
		  loss,
		  seed) {
}

void NeuralNetwork::set_seed(const size_t seed) const {
	if (seed == 0) {
		m_seed = std::random_device{}();
	} else {
		m_seed = seed;
	}
}

void NeuralNetwork::calculate_weights_biases_shapes() {
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		m_weights_shapes.push_back(Shape{m_topology.at(i + 1), m_topology.at(i)});
		m_biases_shapes.push_back(Shape{m_topology.at(i + 1), 1});
	}
}

void NeuralNetwork::init_rnd_weights_biases() {
	std::srand(m_seed);

	for (const Shape& shape : m_weights_shapes) {
		m_weights.push_back(Matrix::Random(shape.at(0), shape.at(1)) * 0.05);
	}

	for (const Shape& shape : m_biases_shapes) {
		m_biases.push_back(Matrix::Random(shape.at(0), shape.at(1)) * 0.01);
	}
}

void NeuralNetwork::init_all_neurons() {
	m_input_layer = Eigen::VectorXd::Zero(m_topology.at(0));

	for (const size_t& layer_size : std::span{m_topology.begin() + 1, m_topology.end()}) {
		m_z_layers.push_back(Eigen::VectorXd::Zero(layer_size));
		m_layers.push_back(Eigen::VectorXd::Zero(layer_size));
	}
}

void NeuralNetwork::setup_loss() {
	m_loss_f_df_pair = loss::get_loss_pair(m_loss_type);
}

void NeuralNetwork::setup_activations() {
	for (const activation::ActivationType& activation_type : m_activation_types) {
		m_activation_f_df_pairs.push_back(activation::get_activation_pair(activation_type));
	}
}

void NeuralNetwork::feed_forward() {
	NeuralNetwork::feed_forward(m_input_layer);
}

void NeuralNetwork::feed_forward(const Vector& input_layer) {
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

void NeuralNetwork::update_weights_biases(const Gradients& gradients, const double& lr) {
	for (auto&& [weights_matrix, dW] : std::ranges::views::zip(m_weights, gradients.dWs)) {
		weights_matrix -= lr * dW;
	}

	for (auto&& [biases_matrix, dB] : std::ranges::views::zip(m_biases, gradients.dBs)) {
		biases_matrix -= lr * dB;
	}
}

double NeuralNetwork::cost_single_sample(const Vector& sample, const Vector& output) {

	feed_forward(sample);

	const Vector& last_layer = m_layers.back();

	return m_loss_f_df_pair.f(last_layer, output);
}

double NeuralNetwork::cost(const Matrix& X, const Matrix& y) {
	double total_cost = 0.0;

	for (size_t i = 0; i < X.cols(); i++) {
		const auto& sample = X.col(i);
		const auto& true_output = y.col(i);

		total_cost += cost_single_sample(sample, true_output);
	}

	// Divide by the number of samples
	total_cost /= X.cols();

	return total_cost;
}

void NeuralNetwork::backpropagate_one(
	const Matrix& sample, const Matrix& output, Gradients& out_gradients) {

	const Vector& input_layer = sample.col(0);
	const Vector& output_layer = output.col(0);

	feed_forward(input_layer);

	// Derivative of the cost w.r.t. the output layer
	const Matrix d_cost_p_ol = m_loss_f_df_pair.df(m_layers.back(), output_layer);

	// Error of the output layer
	const Matrix error_ol =
		m_activation_f_df_pairs.back().df(m_z_layers.back()).array() * d_cost_p_ol.array();

	// Errors of all hidden layers and the output layer, in order.
	std::vector<Matrix> errors(m_N_LAYERS);
	errors.at(errors.size() - 1) = error_ol;

	for (int i = m_N_LAYERS - 2; i >= 0; i--) {
		errors.at(i) = m_activation_f_df_pairs[i].df(m_z_layers[i]).array() *
					   (m_weights.at(i + 1).transpose() * errors.at(i + 1)).array();
	}

	// Based on the equations of backprpoagation we know that
	// The derivative of cost w.r.t. bias of each layer
	// Is actually equal to the error of that layer.
	const std::vector<Matrix>& d_cost_p_biases = errors;

	// Adding it to the gradients dBs
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		out_gradients.dBs[i] += d_cost_p_biases[i];
	}

	// Based on the equations of backpropagation
	// The derivative of the cost w.r.t. the weights of the layer l
	// Is the matrix mult of `error of layer l` and `activation of layer (l-1) transposed`.
	std::vector<Matrix> d_cost_p_weights(m_N_LAYERS);

	d_cost_p_weights.at(0) = errors[0] * m_input_layer.transpose();
	for (size_t i = 1; i < m_N_LAYERS; i++) {
		d_cost_p_weights.at(i) = errors[i] * m_layers[i - 1].transpose();
	}

	// Adding it to the gradients dWs
	for (size_t i = 0; i < m_N_LAYERS; i++) {
		out_gradients.dWs[i] += d_cost_p_weights[i];
	}
}

void NeuralNetwork::backpropagate(
	const Matrix& X_train, const Matrix& y_train, Gradients& out_gradients) {

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
		const Matrix& sample = X_train.col(i);
		const Matrix& output = y_train.col(i);

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

void NeuralNetwork::train(
	const Matrix& X_train,
	const Matrix& y_train,
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

			const Matrix batch_x = X_train(Eigen::all, batch_indices);
			const Matrix batch_y = y_train(Eigen::all, batch_indices);

			backpropagate(batch_x, batch_y, gradients);

			if (not const_lr) {
				current_lr = INITIAL_LR * std::exp(-decay_rate * epoch);
			}

			update_weights_biases(gradients, current_lr);

			if (verbose and batch_i == num_batches - 1) {
				const double train_cost = cost(X_train, y_train);

				std::println(
					"epoch: {:03}/{:03} ~ training_cost={:.4f}", epoch + 1, n_epochs, train_cost);
			}
		}
	}
}

double NeuralNetwork::accuracy(const Matrix& features, const Eigen::VectorXd& labels) {
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

Vector NeuralNetwork::predict(const Vector& X) {
	feed_forward(X);
	return m_layers.back();
}

size_t NeuralNetwork::predict_class(const Vector& X) {
	Vector predicted_output = predict(X);

	size_t max_index;
	predicted_output.maxCoeff(&max_index);
	return max_index;
}
