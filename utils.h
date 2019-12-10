//
// Created by 方泓睿 on 2019/12/10.
//

#ifndef NN_MNIST__UTILS_H_
#define NN_MNIST__UTILS_H_

#include <string>
#include <cstdlib>
#include <exception>
#include <ostream>

namespace nn_mnist::utils {
auto ShowStep(const std::string &name) -> void;

__attribute__((noreturn))
auto Panic(const std::string &what, int exit_value = EXIT_FAILURE) -> void;

__attribute__((noreturn))
auto Panic(const std::exception &exp, int exit_value = EXIT_FAILURE) -> void;

auto PanicIf(bool condition, const std::string &what = "", int exit_value = EXIT_FAILURE) -> void;

auto Info(const std::string &info) -> void;

namespace color {
class Modifier {
 public:
	enum class Code : int32_t {
		FG_RED     = 31,
		FG_GREEN   = 32,
		FG_BLUE    = 34,
		FG_DEFAULT = 39,
		BG_RED     = 41,
		BG_GREEN   = 42,
		BG_BLUE    = 44,
		BG_DEFAULT = 49
	};
 private:
	Code code_;
 public:
	explicit Modifier(Code code);
	friend std::ostream &operator<<(std::ostream &os, const Modifier &mod);
};

std::ostream &operator<<(std::ostream &os, const Modifier &mod);
}

namespace math {
// Sigmoid function.
auto Sigmoid(double) -> double;

// Derivative of the sigmoid function.
auto SigmoidPrime(double) -> double;

// Function returning a random double value in range [fMin, fMax].
auto Rand(double min, double max) -> double;
}
}

#endif //NN_MNIST__UTILS_H_
