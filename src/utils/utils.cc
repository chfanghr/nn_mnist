//
// Created by 方泓睿 on 2019/12/10.
//

#include "utils/utils.h"

#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib>

auto ShowStep(const std::string &name) -> void {
	static size_t step = 1;
	std::cout << color::Modifier(color::Modifier::Code::FG_GREEN)
						<< "[STEP " << step << "] " << name << std::endl
						<< color::Modifier(color::Modifier::Code::FG_DEFAULT);
	step++;
}

__attribute__((noreturn))
auto Panic(const std::string &what, int exit_value) -> void {
	std::cout << color::Modifier(color::Modifier::Code::FG_RED)
						<< "[PANIC] " << what << std::endl
						<< color::Modifier(color::Modifier::Code::FG_DEFAULT);
	exit(exit_value);
}

__attribute__((noreturn))
auto Panic(const std::exception &exp, int exit_value) -> void {
	Panic(std::string("exception.what(): ") + exp.what(), exit_value);
}

auto PanicIf(bool condition, const std::string &what, int exit_value) -> void {
	if (condition)
		Panic(what, exit_value);
}

auto Info(const std::string &info) -> void {
	std::cout << color::Modifier(color::Modifier::Code::FG_BLUE)
						<< "[INFO] " << info << std::endl
						<< color::Modifier(color::Modifier::Code::FG_DEFAULT);
}

auto ShowProgressStart(const std::string &name) -> void {
	std::cout << color::Modifier(color::Modifier::Code::BG_GREEN)
						<< color::Modifier(color::Modifier::Code::FG_RED)
						<< "[PROGRESS] " << name << "   0%";
}

auto ShowProgress(size_t now, size_t all) -> void {
	int32_t progress     = (double) now / all * 100;
	auto    progress_str = std::to_string(progress) + "%";
	while (progress_str.size() < 4)
		progress_str = " " + progress_str; // NOLINT(performance-inefficient-string-concatenation)
	std::cout << "\b\b\b\b" << progress_str;
	std::cout.flush();
}

auto ShowProgressEnd(const std::string &) -> void {
	ShowProgress(1, 1);
	std::cout << color::Modifier(color::Modifier::Code::BG_DEFAULT)
						<< color::Modifier(color::Modifier::Code::FG_DEFAULT)
						<< std::endl;
}

namespace color {
Modifier::Modifier(Modifier::Code code) : code_(code) {}
std::ostream &operator<<(std::ostream &os, const Modifier &mod) {
	return os << "\033[" << (int32_t) mod.code_ << "m";
}
}