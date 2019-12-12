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

ADD_EXECUTABLE(demo
			   src/demo/main.cc
			   src/demo/settings.h
			   )

TARGET_COMPILE_DEFINITIONS(demo
						   PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

TARGET_LINK_LIBRARIES(demo utils nn_mnist)

ADD_EXECUTABLE(digit_recognition
			   src/digit_recognition/main.cc)

TARGET_COMPILE_DEFINITIONS(digit_recognition
						   PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

TARGET_LINK_LIBRARIES(digit_recognition nn_mnist)

ADD_EXECUTABLE(show_mnist_image
			   src/show_mnist_image/main.cc)

TARGET_COMPILE_DEFINITIONS(show_mnist_image
						   PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

TARGET_LINK_LIBRARIES(show_mnist_image utils nn_mnist)