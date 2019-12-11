//
// Created by 方泓睿 on 2019/12/12.
//

#ifndef NN_MNIST_SRC_DEMO_UTILS_H_
#define NN_MNIST_SRC_DEMO_UTILS_H_

#include <string>
#include <cstdlib>

auto ShowStep(const std::string &name) -> void;

__attribute__((noreturn))
auto Panic(const std::string &what, int exit_value = EXIT_FAILURE) -> void;

__attribute__((noreturn))
auto Panic(const std::exception &exp, int exit_value = EXIT_FAILURE) -> void;

auto PanicIf(bool condition, const std::string &what = "", int exit_value = EXIT_FAILURE) -> void;

auto Info(const std::string &info) -> void;

auto ShowProgressStart(const std::string &name) -> void;

auto ShowProgress(size_t now, size_t all) -> void;

auto ShowProgressEnd(const std::string &name) -> void;

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

#endif //NN_MNIST_SRC_DEMO_UTILS_H_
