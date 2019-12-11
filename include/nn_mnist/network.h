//
// Created by 方泓睿 on 2019/12/10.
//

#ifndef NN_MNIST_INCLUDE_NN_MNIST_NETWORK_H_
#define NN_MNIST_INCLUDE_NN_MNIST_NETWORK_H_

#include <vector>
#include <cstdint>
#include <array>
#include <tuple>
#include <bitset>
#include <functional>
#include <iostream>
#include <string>
#include <cmath>
#include <random>

namespace nn_mnist {
class Network {
 public:
	static const size_t layer_num_  = 3;
	static const size_t img_width_  = 28;
	static const size_t img_height_ = 28;

 public:
	using Image=std::array<double, img_width_ * img_height_>;
	using Dataset=std::vector<
			std::pair<
					Image,
					std::bitset<10>
			>
	>;
	using ProgressReporter=std::function<void(size_t, size_t)>;
	using PreProgressReporter=std::function<void(const std::string &)>;
	using PostProgressReporter=std::function<void(const std::string &)>;
	using Reporter=std::function<void(const std::string &)>;

 private:
	static const ProgressReporter     default_progress_reporter_;
	static const PreProgressReporter  default_pre_progress_reporter_;
	static const PostProgressReporter default_post_progress_reporter_;
	static const Reporter             default_reporter_;

 public:
	Network();
	explicit Network(const std::array<size_t, layer_num_> &layers);
	explicit Network(std::istream &is);
	explicit Network(const std::string &file_name);

 public:
	auto Reset() -> void;
	auto Init() -> void;

 public:
	auto Train(const Dataset &training_set,
						 double alpha,
						 const PreProgressReporter &pre_progress_reporter = default_pre_progress_reporter_,
						 const ProgressReporter &progress_reporter = default_progress_reporter_,
						 const PostProgressReporter &post_progress_reporter = default_post_progress_reporter_,
						 const Reporter &reporter = default_reporter_
	) -> void;

	auto Test(const Dataset &testing_set,
						const PreProgressReporter &pre_progress_reporter = default_pre_progress_reporter_,
						const ProgressReporter &progress_reporter = default_progress_reporter_,
						const PostProgressReporter &post_progress_reporter = default_post_progress_reporter_,
						const Reporter &reporter = default_reporter_
	) -> std::tuple<size_t, size_t, double>;

	auto Evaluate(const Image &in) -> std::array<double, 10>;
	auto Evaluate(const Image &in, std::array<double, 10> &res) -> void;

 private:
	std::array<size_t, layer_num_> layers_{};

	std::array<std::vector<std::vector<double>>,
						 layer_num_ - 1> weight_{};

	std::array<std::vector<double>, layer_num_> bias_{}, activation_{}, value_{};

 public:
	auto Save(std::ostream &os) -> void;
	auto SaveTo(const std::string &file_name) -> void;
	auto Load(std::istream &is) -> void;
	auto LoadFrom(const std::string &file_name) -> void;

 private:
	auto FeedForward() -> void;

	auto Input(const Image &in) -> void;

	auto GradientDescent(const Dataset &training_set,
											 double alpha,
											 const ProgressReporter &progress_reporter = default_progress_reporter_
	) -> void;

 private:
	static auto Sigmoid(double x) -> double;
	static auto SigmoidPrime(double x) -> double;
	static auto Rand(double min, double max) -> double;
};
}

#endif //NN_MNIST_INCLUDE_NN_MNIST_NETWORK_H_
