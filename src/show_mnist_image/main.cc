//
// Created by 方泓睿 on 2019/12/12.
//

#include "utils/utils.h"
#ifndef _WIN32
#include <mnist/mnist_reader.hpp>
#else
#include <mnist/mnist_reader_less.hpp>
#endif

#include <getopt.h>

#include <iostream>

static bool kShowAsArray = false;

auto ShowImage(const std::vector<uint8_t> &image, int32_t label) -> void;

auto main(int argc, char **argv) -> int {
	Info(std::string("Data location of mnist: ") + MNIST_DATA_LOCATION);

	auto mnist_dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	PanicIf(mnist_dataset.training_images.empty() ||
							mnist_dataset.test_images.empty() ||
							mnist_dataset.test_labels.empty() ||
							mnist_dataset.training_labels.empty(),
					"Failed to load mnist_dataset.");

	Info("Dataset loaded");

	int c{};

	while ((c = getopt(argc, argv, "at:T:")) != -1) {
		switch (c) {
			case 'a':kShowAsArray = true;
				break;
			case 't': {
				size_t idx = std::stoul(optarg);
				if (idx >= mnist_dataset.training_images.size())
					Panic("Index out of range");
				ShowImage(mnist_dataset.training_images[idx], mnist_dataset.training_labels[idx]);
			}
				break;
			case 'T': {
				size_t idx = std::stoul(optarg);
				if (idx >= mnist_dataset.test_images.size())
					Panic("Index out of range");
				ShowImage(mnist_dataset.test_images[idx], mnist_dataset.test_labels[idx]);
			}
				break;
			case '?': Panic("-T/t requires one argument");
			default: Panic("Unknown option: " + std::to_string(c));
		}
	}

	return EXIT_SUCCESS;
}

auto ProcessPixel(uint8_t pixel) -> char {
	switch (pixel) {
		case 0: return '_';
		case 1 ... 40: return '.';
		case 41 ... 80: return '*';
		case 81 ... 128: return 'x';
		case 129 ... 220: return 'X';
		default: return '#';
	}
}

auto ShowImage(const std::vector<uint8_t> &image, int32_t label) -> void {
	Info("=====================================");
	Info("label=" + std::to_string(label));
	if (kShowAsArray) {
		for (size_t i = 0; i < 28; i++) {
			for (size_t j = 0; j < 28; j++)
				std::cout << (double) image[28 * i + j] / 255 << ", ";
			std::cout << std::endl;
		}
	} else {
		for (size_t i = 0; i < 28; i++) {
			for (size_t j = 0; j < 28 * 2; j++)
				std::cout << ProcessPixel(image[28 * i + j / 2]);
			std::cout << std::endl;
		}
	}
	Info("=====================================");
}