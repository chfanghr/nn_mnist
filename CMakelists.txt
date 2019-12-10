CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(nn_mnist)

SET(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(MNIST PATHS third_party/mnist)

IF (NOT MNIST_FOUND)
	MESSAGE(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
ENDIF (NOT MNIST_FOUND)

INCLUDE_DIRECTORIES(${MNIST_INCLUDE_DIR})

ADD_EXECUTABLE(digit_recognition main.cc network.h settings.h utils.h utils.cc network.cc)

TARGET_COMPILE_DEFINITIONS(digit_recognition PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")