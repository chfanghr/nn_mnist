CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(nn_mnist)

SET(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(MNIST PATHS third_party/mnist)

IF (NOT MNIST_FOUND)
	MESSAGE(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
ENDIF (NOT MNIST_FOUND)

INCLUDE_DIRECTORIES(${MNIST_INCLUDE_DIR})
INCLUDE_DIRECTORIES(include/)

ADD_LIBRARY(nn_mnist
			include/nn_mnist/network.h
			src/nn_mnist/network.cc)

ADD_LIBRARY(utils OBJECT
			include/utils/utils.h
			src/utils/utils.cc)

ADD_EXECUTABLE(trainer
			   src/trainer/main.cc
			   src/trainer/settings.h
			   )

TARGET_COMPILE_DEFINITIONS(trainer
						   PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

TARGET_LINK_LIBRARIES(trainer utils nn_mnist)

ADD_EXECUTABLE(digit_recognition_example
			   src/digit_recognition_example/main.cc)

TARGET_LINK_LIBRARIES(digit_recognition_example utils nn_mnist)

TARGET_COMPILE_DEFINITIONS(digit_recognition_example
						   PRIVATE NETWORK_DAT_LOCATION="${CMAKE_SOURCE_DIR}/data/network.dat")

ADD_EXECUTABLE(show_mnist_image
			   src/show_mnist_image/main.cc)

TARGET_COMPILE_DEFINITIONS(show_mnist_image
						   PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

TARGET_LINK_LIBRARIES(show_mnist_image utils nn_mnist)