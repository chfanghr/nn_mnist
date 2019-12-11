//
// Created by 方泓睿 on 2019/12/10.
//

#include "network.h"

#include "utils.h"

#include <algorithm>

namespace nn_mnist::network {

Network::Network(size_t n, const std::vector<int32_t> &nums) {
	using utils::math::Rand;

	// Setting the number of layers and number of neurons in each layer.
	network_size_ = n;
	layer_        = nums;

	// Weight
	for (size_t i = 0; i < network_size_ - 1; i++) {
		weight_.emplace_back();
		for (int32_t j = 0; j < layer_[i]; j++) {
			weight_[i].emplace_back();
			for (int32_t k = 0; k < layer_[i + 1]; k++)
				weight_[i][j].push_back(Rand(-1.0, 1.0));
		}
	}

	// Bias
	for (size_t i = 0; i < network_size_; i++) {
		bias_.emplace_back();
		for (int32_t j = 0; j < layer_[i]; j++)
			bias_[i].push_back(Rand(-1.0, 1.0));
	}

	// Activation
	for (size_t i = 0; i < network_size_; i++) {
		activation_.emplace_back();
		for (int32_t j = 0; j < layer_[i]; j++)
			activation_[i].push_back(0.0);
	}

	// Value
	for (size_t i = 0; i < network_size_; i++) {
		value_.emplace_back();
		for (int32_t j = 0; j < layer_[i]; j++)
			value_[i].push_back(0.0);
	}
};

auto Network::SetDataset(const std::vector<std::vector<double >> &dataset, const std::vector<std::vector<double>>
&expected, size_t size) -> void {
	dataset_size_ = size;
	dataset_      = dataset;
	expected_     = expected;
}

auto Network::FeedForward() -> void {
	using utils::math::Sigmoid;

	for (size_t i = 1; i < network_size_; i++) {
		for (int32_t j = 0; j < layer_[i]; j++) {
			double sum{};

			for (int32_t k = 0; k < layer_[i - 1]; k++)
				sum += activation_[i - 1][k] * weight_[i - 1][k][j];

			sum += bias_[i][j];
			value_[i][j]      = sum;
			activation_[i][j] = Sigmoid(value_[i][j]);
		}
	}
}

auto Network::GetScore(int32_t idx) -> double {
	return activation_[network_size_ - 1][idx];
}

auto Network::Input(const std::vector<double> &in) {
	for (size_t i = 0; i < in.size(); i++)
		activation_[0][i] = in[i];
}

auto Network::GradientDescent(double alpha, int32_t n) -> void {
	using utils::Info, utils::math::SigmoidPrime;

	Info("Start of the gradient descent");

	// Vectors storing derivatives of the cost function with respect to biases and with respect to weights.
	std::vector<double>              derivative_B1{}, derivative_B2{};
	std::vector<std::vector<double>> derivative_W0{}, derivative_W1{};

	derivative_B1.resize(layer_[1]);
	derivative_B2.resize(layer_[2]);
	derivative_W0.resize(layer_[0]);
	derivative_W1.resize(layer_[1]);

	std::for_each(derivative_W0.begin(), derivative_W0.end(),
								[&](std::vector<double> &vec) { vec.resize(layer_[1]); });
	std::for_each(derivative_W1.begin(), derivative_W1.end(),
								[&](std::vector<double> &vec) { vec.resize(layer_[2]); });

	for (int32_t z = 0; z < n; z++) {
		int32_t i = z % dataset_size_;
		Input(dataset_[i]);
		FeedForward();

		// Bias_2 derivative
		for (int32_t a = 0; a < layer_[2]; a++)
			derivative_B2[a] = (activation_[2][a] - expected_[i][a])
					* SigmoidPrime(value_[2][a]);

		// Weight_1 derivative
		for (int32_t a = 0; a < layer_[1]; a++)
			for (int32_t b = 0; b < layer_[2]; b++)
				derivative_W1[a][b] = (activation_[2][b] - expected_[i][b])
						* SigmoidPrime(value_[2][b])
						* activation_[1][a];

		// Bias_1 derivative
		for (int32_t a = 0; a < layer_[1]; a++) {
			double       sum{};
			for (int32_t j = 0; j < layer_[2]; j++)
				sum += (activation_[2][j] - expected_[i][j])
						* SigmoidPrime(value_[2][j])
						* weight_[1][a][j]
						* SigmoidPrime(value_[1][a]);
			derivative_B1[a] = sum;
		}

		// Weight_0 derivative
		for (int32_t a = 0; a < layer_[0]; a++) {
			for (int32_t b = 0; b < layer_[1]; b++) {
				double sum{};

				for (int32_t j = 0; j < layer_[2]; j++)
					sum += (activation_[2][j] - expected_[i][j])
							* SigmoidPrime(value_[2][j])
							* weight_[1][b][j]
							* SigmoidPrime(value_[1][b])
							* activation_[0][a];
				derivative_W0[a][b] = sum;
			}
		}

		// Here the actual gradient descent is being applied.
		// Values of weights and biases are being changed by a fraction of its derivatives
		for (int32_t j = 0; j < layer_[2]; j++)
			bias_[2][j] -= alpha * derivative_B2[j];

		for (int32_t j = 0; j < layer_[1]; j++)
			bias_[1][j] -= alpha * derivative_B1[j];

		for (int32_t j = 0; j < layer_[1]; j++)
			for (int32_t k = 0; k < layer_[2]; k++)
				weight_[1][j][k] -= alpha * derivative_W1[j][k];

		for (int32_t j = 0; j < layer_[0]; j++)
			for (int32_t k = 0; k < layer_[1]; k++)
				weight_[0][j][k] -= alpha * derivative_W0[j][k];

		if (!(z % 100) && z)
			Info(std::string("Iteration nr ") + std::to_string(z + 1));
	}
}

auto Network::Evaluate(int32_t n,
											 std::vector<std::vector<double>> &data,
											 const std::vector<int32_t> &exp) -> void {
	using utils::Info;
	Info("Start of evaluation");

	for (int32_t i = 0; i < n; i++) {
		Info(std::string("Dataset nr: ") + std::to_string(i + 1));

		// Loading dataset entry to the neural net.
		Input(data[i]);
		FeedForward();

		// Converting expected vector element to a double vector.
		std::vector<double> x{};
		x.reserve(layer_[2]);

		for (int32_t j = 0; j < layer_[2]; j++)
			x.push_back(exp[i] != j ? 0 : 1);

		for (int32_t j = 0; j < layer_[2]; j++)
			Info(std::to_string(j) + ": " +
					std::to_string(activation_[2][j]) + "     " +
					std::to_string(x[j]));
	}

	// Calculating net's overall performance on the test dataset
	int32_t sum{};

	for (int32_t i = 0; i < data.size(); i++) {
		if ((i % 100) == 0 && i)
			Info(std::string("Evaluation nr ") + std::to_string(i + 1));

		double  max_A{};
		int32_t max_A_i = -1;

		Input(data[i]);
		FeedForward();
		for (int32_t j = 0; j < 10; j++) {
			if (activation_[2][j] > max_A) {
				max_A   = activation_[2][j];
				max_A_i = j;
			}
		}

		if (max_A_i == exp[i])
			sum++;
	}

	double score = (double) sum / data.size();
	Info(std::string("Scored ") + std::to_string(sum) + " out of " + std::to_string(data.size()));
	Info(std::string("Rate ") + std::to_string(score * 100) + "%");
}

}