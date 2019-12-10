//
// Created by 方泓睿 on 2019/12/10.
//

#ifndef NN_MNIST__NETWORK_H_
#define NN_MNIST__NETWORK_H_

#include <vector>
#include <cstdint>

namespace nn_mnist::network {
class Network {
 private:
	size_t               network_size_{};
	std::vector<int32_t> layer_{};
 private:
	std::vector<std::vector<std::vector<double>>> weight_{};
	std::vector<std::vector<double>>              bias_{}, activation_, value_{};
 private:
	size_t                           dataset_size_{};
	std::vector<std::vector<double>> dataset_{}, expected_{};
 public:
	Network(size_t n, const std::vector<int32_t> &nums);

	auto SetDataset(const std::vector<std::vector<double >> &dataset, const std::vector<std::vector<double>> &expected,
									size_t size) -> void;

	auto FeedForward() -> void;

	auto GetScore(int32_t idx) -> double;

	auto Input(const std::vector<double> &in);

	auto GradientDescent(double alpha, int32_t n) -> void;

	auto Evaluate(int32_t n, std::vector<std::vector<double>> &data, const std::vector<int32_t> &expected) -> void;
};
}
#endif //NN_MNIST__NETWORK_H_
