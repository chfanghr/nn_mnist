//
// Created by 方泓睿 on 2019/12/10.
//

#include "network.h"
#include "settings.h"
#include "utils.h"

#ifndef _WIN32
#include <mnist/mnist_reader.hpp>
#else
#include <mnist/mnist_reader_less.hpp>
#endif

#include <iostream>
#include <iomanip>
#include <cstdint>

auto main() -> int {
	//===============================================================
	// Setup
	nn_mnist::utils::ShowStep("Setup");
	// Vectors storing values from both training and test datasets as doubles from range [0.0, 1.0].
	std::vector<std::vector<double>> training_images{}, test_images{};

	// Vectors storing expected values for each entry from the dataset.
	std::vector<std::vector<double>> training_expected{};
	std::vector<int32_t>             test_expected{};

	std::cout << std::fixed << std::setprecision(10);

	//===============================================================
	// Loading MNIST data.
	nn_mnist::utils::ShowStep("Load dataset");

	nn_mnist::utils::Info(std::string("Data location of mnist: ") + MNIST_DATA_LOCATION);

	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	nn_mnist::utils::PanicIf(dataset.training_images.empty() ||
															 dataset.test_images.empty() ||
															 dataset.test_labels.empty() ||
															 dataset.training_labels.empty(),
													 "Failed to load dataset.");


	//===============================================================
	// Normalizing the dataset.
	// The Original dataset is a vector of grayscale images - integer vectors with values from 0 to 255.
	// I'm converting it to the range [0.0, 1.0].
	nn_mnist::utils::ShowStep("Normalize dataset");

	for (size_t i = 0; i < nn_mnist::kDataSetSize; i++) {
		training_images.emplace_back();
		for (auto j = 0; j < 28 * 28; j++)
			training_images[i].push_back((double) dataset.training_images[i][j] / 255);
	}

	for (size_t i = 0; i < dataset.test_labels.size(); i++) {
		test_images.emplace_back();
		for (auto j = 0; j < 28 * 28; j++)
			test_images[i].push_back((double) dataset.test_images[i][j] / 255);
	}

	//===============================================================
	// Setting the "expected" vectors.
	nn_mnist::utils::ShowStep("Setup expected vectors");

	for (const auto &test_label : dataset.test_labels)
		test_expected.push_back((int) test_label);

	for (int i = 0; i < nn_mnist::kDataSetSize; i++) {
		training_expected.emplace_back();
		for (int j = 0; j < 10; j++)
			if (dataset.training_labels[i] == j)
				training_expected[i].push_back(1.0);
			else
				training_expected[i].push_back(0.0);
	}

	//===============================================================
	// Neural network object initialization.
	nn_mnist::utils::ShowStep("Initialize neural network");

	auto network = nn_mnist::network::Network(3, {28 * 28, 30, 10});
	network.SetDataset(training_images, training_expected, nn_mnist::kDataSetSize);

	//===============================================================
	// Train the network.
	nn_mnist::utils::ShowStep("Train the network");
	network.GradientDescent(nn_mnist::kAlpha, nn_mnist::kDataSetSize);

	//===============================================================
	// Evaluate with the trained network.
	nn_mnist::utils::ShowStep("Evaluate using the trained network");

	network.Evaluate(20, test_images, test_expected);

	return EXIT_SUCCESS;
}