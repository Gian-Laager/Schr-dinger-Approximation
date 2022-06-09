#include <chrono>
#include <iostream>

constexpr int steps = 5;

int main() {

    system("/usr/lib/go/bin/go build -o /tmp/GoLand/___1go_build_SchroedingerApproximation SchroedingerApproximation #gosetup");

    double timeSum = 0.0;

    for (int i = 0; i < steps; i++) {
        auto before = std::chrono::high_resolution_clock::now();
        system("/tmp/GoLand/___1go_build_SchroedingerApproximation");
        auto after = std::chrono::high_resolution_clock::now();
        timeSum += (after - before).count() / 1e9;
    }

    std::cout << "average: " << timeSum / static_cast<double>(steps) << std::endl;
}