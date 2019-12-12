//
// Created by 方泓睿 on 2019/12/10.
//

#include <nn_mnist/network.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <utility>
#include <iomanip>

namespace nn_mnist {
const Network::ProgressReporter     Network::default_progress_reporter_      = // NOLINT(cert-err58-cpp)
																				[](size_t, size_t) -> void {};
const Network::PreProgressReporter  Network::default_pre_progress_reporter_  = // NOLINT(cert-err58-cpp)
																				[](const std::string &) {};
const Network::PostProgressReporter Network::default_post_progress_reporter_ = // NOLINT(cert-err58-cpp)
																				[](const std::string &) {};
const Network::Reporter             Network::default_reporter_               = // NOLINT(cert-err58-cpp)
																				[](const std::string &) {};

Network::Network()
		: Network(std::array<size_t, layer_num_>{28 * 28, 30, 10}) {}

Network::Network(const std::array<size_t, layer_num_> &layers)
		: layers_(layers) {
	Init();
}

Network::Network(std::istream &is) : Network() {
	Load(is);
}

Network::Network(const std::string &file_name) : Network() {
	LoadFrom(file_name);
}

auto Network::Reset() -> void {
	weight_ = {};
	value_  = activation_ = bias_ = {};
}

auto Network::Init() -> void {
	// Weight
	for (size_t i = 0; i < layer_num_ - 1; i++) {
		weight_[i].resize(layers_[i]);
		for (size_t j = 0; j < layers_[i]; j++) {
			weight_[i][j].resize(layers_[i + 1]);
			std::generate(
					weight_[i][j].begin(),
					weight_[i][j].end(),
					[]() { return Rand(-1, 1); }
			);
		}
	}

	// Bias
	for (size_t i = 0; i < layer_num_; i++) {
		bias_[i].resize(layers_[i]);
		std::generate(bias_[i].begin(),
									bias_[i].end(),
									[]() { return Rand(-1, 1); });
	}

	// Activation && Value
	for (size_t i = 0; i < layer_num_; i++) {
		activation_[i].resize(layers_[i], 0);
		value_[i].resize(layers_[i], 0);
	}
}

auto Network::Save(std::ostream &os) -> void {
	os << std::fixed << std::setprecision(10);

	auto serialize_1d_vector = [&](const auto &vec) {
		for (const auto &ele:vec)
			os << ele << " ";
		os << std::endl;
	};
	auto serialize_2d_vector = [&](const auto &vec_2d) {
		for (const auto &vec:vec_2d)
			serialize_1d_vector(vec);
	};
	auto serialize_3d_vector = [&](const auto &vec_3d) {
		for (const auto &vec_2d:vec_3d)
			serialize_2d_vector(vec_2d);
	};

	serialize_1d_vector(layers_);
	serialize_3d_vector(weight_);
	serialize_2d_vector(bias_);
	serialize_2d_vector(activation_);
	serialize_2d_vector(value_);
}

auto Network::Load(std::istream &is) -> void {
	Reset();

	auto deserialize_1d_vector = [&](auto &vec) {
		if (is.eof() || is.fail())
			throw std::runtime_error("unexpected I/O error or EOF");
		std::stringstream ss{};
		std::string       line{};
		if (!std::getline(is, line))
			throw std::runtime_error("unexpected I/O error or EOF");
		ss << line;
		for (auto &ele:vec)
			ss >> ele;
	};
	auto deserialize_2d_vector = [&](auto &vec_2d) {
		for (auto &vec:vec_2d)
			deserialize_1d_vector(vec);
	};
	auto deserialize_3d_vector = [&](auto &vec_3d) {
		for (auto &vec_2d:vec_3d)
			deserialize_2d_vector(vec_2d);
	};

	deserialize_1d_vector(layers_);

	Init();

	deserialize_3d_vector(weight_);
	deserialize_2d_vector(bias_);
	deserialize_2d_vector(activation_);
	deserialize_2d_vector(value_);
}

auto Network::FeedForward() -> void {
	for (size_t i = 1; i < layer_num_; i++) {
		for (int32_t j = 0; j < layers_[i]; j++) {
			double sum{};

			for (int32_t k = 0; k < layers_[i - 1]; k++)
				sum += activation_[i - 1][k] * weight_[i - 1][k][j];

			sum += bias_[i][j];
			value_[i][j]      = sum;
			activation_[i][j] = Sigmoid(value_[i][j]);
		}
	}
}

auto Network::Input(const Network::Image &in) -> void {
	std::copy(in.begin(), in.end(), activation_[0].begin());
}

auto Network::Train(const Network::Dataset &training_set,
										double alpha,
										const Network::PreProgressReporter &pre_progress_reporter,
										const Network::ProgressReporter &progress_reporter,
										const Network::PostProgressReporter &post_progress_reporter,
										const Network::Reporter &reporter) -> void {
	reporter("Size of training dataset is " + std::to_string(training_set.size()));
	reporter("Start of training");
	pre_progress_reporter("Training");
	GradientDescent(training_set, alpha, progress_reporter);
	post_progress_reporter("Training");
}

auto Network::Test(const Network::Dataset &testing_set,
									 const Network::PreProgressReporter &pre_progress_reporter,
									 const Network::ProgressReporter &progress_reporter,
									 const Network::PostProgressReporter &post_progress_reporter,
									 const Network::Reporter &reporter) -> std::tuple<size_t, size_t, double> {
	reporter("Size of testing dataset is " + std::to_string(testing_set.size()));
	reporter("Start of testing");

	int32_t sum{};

	pre_progress_reporter("Testing");

	for (size_t i = 0; i < testing_set.size(); i++) {
		progress_reporter(i, testing_set.size());

		double  max{};
		int64_t max_idx = -1;

		Input(testing_set[i].first);
		FeedForward();

		for (size_t j = 0; j < 10; j++)
			if (activation_[2][j] > max) {
				max     = activation_[2][j];
				max_idx = j;
			}

		if (max_idx >= 0 && testing_set[i].second[max_idx])
			sum++;
	}

	post_progress_reporter("Testing");

	double score = (double) sum / testing_set.size() * 100;

	reporter(std::string("Scored ") + std::to_string(sum) + " out of " + std::to_string(testing_set.size()));
	reporter(std::string("Rate ") + std::to_string(score) + "%");

	return std::make_tuple(sum, testing_set.size(), score);
}

auto Network::Evaluate(const Network::Image &in) -> std::array<double, 10> {
	std::array<double, 10> res{};
	Evaluate(in, res);
	return res;
}

auto Network::Evaluate(const Network::Image &in, std::array<double, 10> &res) -> void {
	Input(in);
	FeedForward();

	std::copy(activation_[2].begin(), activation_[2].end(), std::begin(res));
}

auto Network::GradientDescent(const Network::Dataset &training_set,
															double alpha,
															const Network::ProgressReporter &progress_reporter) -> void {

	// Vectors storing derivatives of the cost function with respect to biases and with respect to weights.
	std::vector<double>              derivative_B1{}, derivative_B2{};
	std::vector<std::vector<double>> derivative_W0{}, derivative_W1{};

	derivative_B1.resize(layers_[1]);
	derivative_B2.resize(layers_[2]);
	derivative_W0.resize(layers_[0]);
	derivative_W1.resize(layers_[1]);

	std::for_each(derivative_W0.begin(), derivative_W0.end(),
								[&](auto &vec) { vec.resize(layers_[1]); });
	std::for_each(derivative_W1.begin(), derivative_W1.end(),
								[&](auto &vec) { vec.resize(layers_[2]); });

	for (size_t i = 0; i < training_set.size(); i++) {
		progress_reporter(i, training_set.size());

		Input(training_set[i].first);
		FeedForward();

		// Bias_2 derivative
		for (int32_t a = 0; a < layers_[2]; a++)
			derivative_B2[a] = (activation_[2][a] - training_set[i].second[a])
					* SigmoidPrime(value_[2][a]);

		// Weight_1 derivative
		for (int32_t a = 0; a < layers_[1]; a++)
			for (int32_t b = 0; b < layers_[2]; b++)
				derivative_W1[a][b] = (activation_[2][b] - training_set[i].second[b])
						* SigmoidPrime(value_[2][b])
						* activation_[1][a];

		// Bias_1 derivative
		for (int32_t a = 0; a < layers_[1]; a++) {
			double       sum{};
			for (int32_t j = 0; j < layers_[2]; j++)
				sum += (activation_[2][j] - training_set[i].second[j])
						* SigmoidPrime(value_[2][j])
						* weight_[1][a][j]
						* SigmoidPrime(value_[1][a]);
			derivative_B1[a] = sum;
		}

		// Weight_0 derivative
		for (int32_t a = 0; a < layers_[0]; a++) {
			for (int32_t b = 0; b < layers_[1]; b++) {
				double sum{};

				for (int32_t j = 0; j < layers_[2]; j++)
					sum += (activation_[2][j] - training_set[i].second[j])
							* SigmoidPrime(value_[2][j])
							* weight_[1][b][j]
							* SigmoidPrime(value_[1][b])
							* activation_[0][a];
				derivative_W0[a][b] = sum;
			}
		}

		// Here the actual gradient descent is being applied.
		// Values of weights and biases are being changed by a fraction of its derivatives
		for (int32_t j = 0; j < layers_[2]; j++)
			bias_[2][j] -= alpha * derivative_B2[j];

		for (int32_t j = 0; j < layers_[1]; j++)
			bias_[1][j] -= alpha * derivative_B1[j];

		for (int32_t j = 0; j < layers_[1]; j++)
			for (int32_t k = 0; k < layers_[2]; k++)
				weight_[1][j][k] -= alpha * derivative_W1[j][k];

		for (int32_t j = 0; j < layers_[0]; j++)
			for (int32_t k = 0; k < layers_[1]; k++)
				weight_[0][j][k] -= alpha * derivative_W0[j][k];
	}
}

auto Network::Rand(double min, double max) -> double {
	static std::random_device               rd{};
	static std::mt19937                     gen(rd());
	static std::uniform_real_distribution<> dis(min, max);
	return dis(gen);
}

auto Network::SigmoidPrime(double x) -> double {
	return Sigmoid(x) * (1 - Sigmoid(x));
}

auto Network::Sigmoid(double x) -> double {
	return 1.0 / (1.0 + exp(-x));
}

auto Network::SaveTo(const std::string &file_name) -> void {
	std::fstream file(file_name, std::ios_base::out);
	if (!file)
		throw std::runtime_error("failed to open " + file_name);
	Save(file);
}

auto Network::LoadFrom(const std::string &file_name) -> void {
	std::fstream file(file_name, std::ios_base::in);
	if (!file)
		throw std::runtime_error("failed to open " + file_name);
	Load(file);
}

}