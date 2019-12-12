//
// Created by 方泓睿 on 2019/12/13.
//

#include <nn_mnist/network.h>

#include <iostream>

#ifndef NETWORK_DAT_LOCATION
#define NETWORK_DAT_LOCATION "network.dat"
#endif

nn_mnist::Network::Image kImg = {};

nn_mnist::Network network{};

extern "C" void SetData(size_t idx, double data) __attribute__((used));

extern "C" int32_t Evaluate() __attribute__((used));

extern "C" void ShowImg() __attribute__((used));

extern "C" void Clear() __attribute__((used));

extern "C" void SetData(size_t idx, double data) {
  if (idx < 0 || idx >= kImg.size())
	return;
  kImg[idx] = data ? 1 : 0;
}

extern "C" void ShowImg() {
  for (size_t i = 0; i < 28; i++) {
	for (size_t j = 0; j < 28; j++)
	  std::cout << kImg[i * 28 + j];
	std::cout << std::endl;
  }
  std::cout.flush();
}

extern "C" int32_t Evaluate() {
  auto res = network.Evaluate(kImg);
//  for (size_t i = 0; i < res.size(); i++)
//	std::cout << i << "   " << res[i] << std::endl;
//  std::cout.flush();
  double max{};
  int32_t max_idx = -1;
  for (size_t i = 0; i < res.size(); i++) {
	if (res[i] > max) {
	  max = res[i];
	  max_idx = i;
	}
  }
  return max_idx;
}

extern "C" void Clear() {
  kImg = {};
}

auto main() -> int {
  try {
	std::cout << std::string("Load network.dat from ") + NETWORK_DAT_LOCATION << std::endl;
	network = nn_mnist::Network(NETWORK_DAT_LOCATION);
	std::cout << "network.dat loaded" << std::endl;
	std::cout.flush();
  } catch (const std::exception &exception) {
	std::cout << "Failed to load network.dat: " << exception.what() << std::endl;
	std::cout.flush();
	return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}