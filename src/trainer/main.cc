//
// Created by 方泓睿 on 2019/12/10.
//

#include "utils/utils.h"
#include "settings.h"

#include <nn_mnist/network.h>

#ifndef _WIN32
#include <mnist/mnist_reader.hpp>
#else
#include <mnist/mnist_reader_less.hpp>
#endif

#include <iomanip>

auto main() -> int {
	//===============================================================
	// Setup.
	ShowStep("Setup");

	nn_mnist::Network::Dataset training_dataset, testing_dataset;

	std::cout << std::fixed << std::setprecision(10);

	//===============================================================
	// Loading MNIST data.

	ShowStep("Load mnist_dataset");

	Info(std::string("Data location of mnist: ") + MNIST_DATA_LOCATION);

	auto mnist_dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	PanicIf(mnist_dataset.training_images.empty() ||
							mnist_dataset.test_images.empty() ||
							mnist_dataset.test_labels.empty() ||
							mnist_dataset.training_labels.empty(),
					"Failed to load mnist_dataset.");

	//===============================================================
	// Normalizing the mnist_dataset.
	// The Original mnist_dataset is a vector of grayscale images - integer vectors with values from 0 to 255.
	// I'm converting it to the range [0.0, 1.0].
	ShowStep("Convert mnist_dataset");

	training_dataset.resize(kTrainingDatasetSize);
	testing_dataset.resize(mnist_dataset.test_images.size());

	for (size_t i = 0; i < kTrainingDatasetSize; i++) {
		for (size_t j = 0; j < 28 * 28; j++)
			training_dataset[i].first[j] = (double) mnist_dataset.training_images[i][j] / 255;
		training_dataset[i].second.reset();
		training_dataset[i].second[mnist_dataset.training_labels[i]] = true;
	}

	for (size_t i = 0; i < mnist_dataset.test_images.size(); i++) {
		for (size_t j = 0; j < 28 * 28; j++)
			testing_dataset[i].first[j] = (double) mnist_dataset.test_images[i][j] / 255;
		testing_dataset[i].second.reset();
		testing_dataset[i].second[mnist_dataset.test_labels[i]] = true;
	}

	//===============================================================
	// Neural network object initialization.
	ShowStep("Initialize neural network");

	auto network = nn_mnist::Network();

	//===============================================================
	// Train the network.
	ShowStep("Train the network");

	network.Train(training_dataset, kAlpha, ShowProgressStart, ShowProgress, ShowProgressEnd, Info);

	//===============================================================
	// Test the network
	ShowStep("Test the network");

	network.Test(testing_dataset, ShowProgressStart, ShowProgress, ShowProgressEnd, Info);

	//===============================================================
	// Save network to file system
	ShowStep("Save network to filesystem");
	network.SaveTo("network");

	//===============================================================
	// Load network from filesystem
	ShowStep("Load network from filesystem");
	network.LoadFrom("network");

	//===============================================================
	// Test the network
	ShowStep("Test the network");

	network.Test(testing_dataset, ShowProgressStart, ShowProgress, ShowProgressEnd, Info);

	return EXIT_SUCCESS;
}