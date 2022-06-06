#include "pch.h"
#include "potential.h"
#include "constants.h"
#include <sstream>
#include <fstream>
#include <atomic>
#include <future>
#include <numeric>
#include <gmpxx.h>

int add(int a, int b)
{
    return a + b;
}

std::complex<double> integrate(const std::function<std::complex<double>(double)> f, double a, double b, size_t n)
{
    std::complex<double> result = f(a) / 2.0 + f(b) / 2.0;

    for (int k = 1; k < n; k++)
    {
        result += f(a + static_cast<double>(k) * (b - a) / static_cast<double>(n));
    }

    return (b - a) / static_cast<double>(n) * result;
}

std::complex<double>
integrateTrapezoidal(std::function<std::complex<double>(double)> f, double a, double b, size_t n)
{
    std::complex<double> result = 0.0;
    auto nd = static_cast<double>(n);

    std::vector<std::future<std::complex<double>>> futures(n);

    for (size_t k = 0; k < n; k++)
    {
        auto kd = static_cast<double>(k);
        double a_ = a + kd * (b - a) / nd;
        double b_ = a + (kd + 1) * (b - a) / nd;

        result += (b_ - a_) * (f(a_) + f(b_)) / 2.0;
    }
    return result;
}

class IntegrateTrapezoidal
{
public:

    std::function<std::complex<double>(double)> f;
    double a;
    double b;
    size_t n;

    IntegrateTrapezoidal(std::function<std::complex<double>(double)> f,
                         double a,
                         double b,
                         size_t n) : f(std::move(f)),
                                     a(a),
                                     b(b),
                                     n(n)
    {

    }


    std::complex<double> eval()
    {
        std::complex<double> result = 0.0;
        auto nd = static_cast<double>(n);

        std::vector<std::future<std::complex<double>>> futures;

        constexpr size_t num_threads = 16;

        for (size_t k = 0; k < num_threads; k++)
        {
            size_t k_start = k * n / num_threads;
            size_t k_end = (k + 1) * n / num_threads;
            futures.push_back(std::async(std::launch::async,
                                         [this, nd, k_start, k_end]() {
                                             std::complex<double> result = 0.0;
                                             for (size_t k = k_start; k < k_end; k++)
                                             {
                                                 auto kd = static_cast<double>(k);
                                                 double a_ = a + kd * (b - a) / nd;
                                                 double b_ = a + (kd + 1) * (b - a) / nd;

                                                 result += (b_ - a_) * (f(a_) + f(b_)) / 2.0;
                                             }
                                             return result;
                                         }));
        }

        std::vector<std::complex<double>> vals;

        for (auto& f: futures)
        {
            f.wait();
            vals.push_back(f.get());
        }

        return std::accumulate(vals.begin(), vals.end(), 0.0 + 0.0i);
    }
};

class waveFunc
{
public:
    double mass;
    double energy;
    double cPlus;
    double cMinus;
    std::function<double(double)> v;
    size_t integrateSteps;

    std::complex<double> funcToIntegrate(double x)
    {
        return std::sqrt(std::complex<double>(2 * mass * (v(x) - energy))) / std::numbers::h_bar;
    }

    std::complex<double> operator()(double x)
    {
        std::complex<double> superPosCplus = std::exp(
                IntegrateTrapezoidal([this](double x_) { return funcToIntegrate(x_); }, 0, x,
                                     integrateSteps).eval());
        std::complex<double> superPosCminus = std::exp(
                -IntegrateTrapezoidal([this](double x_) { return funcToIntegrate(x_); }, 0, x,
                                      integrateSteps).eval());
        auto phase = IntegrateTrapezoidal([this](double x_) { return funcToIntegrate(x_); }, 0, x,
                                          integrateSteps).eval();

        return (cPlus * superPosCplus + cMinus * superPosCminus) / std::sqrt(funcToIntegrate(x));
    }
};

#ifdef SA_LIB
namespace sa
{
#endif

int main()
{
    StepPotential stepPotential;
    stepPotential.yStart = 0;
    stepPotential.edges.push_back({-1, 1});
    stepPotential.edges.push_back({0, 0});
    stepPotential.edges.push_back({1, 1});
    stepPotential.edges.push_back({2, 0});

    double mass = 4.0 * 2.18e-8;
    double energy = (1000 + 0.5) * std::numbers::pi * std::numbers::h_bar;
    double c0 = 3;
    double theta = std::numbers::pi / 2;
    double cPlus = 0.5 * c0 * cos(theta - std::numbers::pi / 4);
    double cMinus = -0.5 * c0 * sin(theta - std::numbers::pi / 4);

    std::stringstream output{};

    auto potential = [](double x) { return x * x; };

    auto waveFunction = waveFunc{mass, energy, cPlus, cMinus, potential, 1000};

    for (double x = -3; x <= 3; x += 0.0001)
    {
        auto waveFuncAtX = waveFunction(x);
        double real = waveFuncAtX.real();
        double imag = waveFuncAtX.imag();
        if (std::isnan(real) || std::isinf(real))
        {
            std::cerr << "real is nan" << std::endl;
            continue;
        }
        if (std::isnan(imag) || std::isinf(imag))
        {
            std::cerr << "imag is nan" << std::endl;
            continue;
        }
        output << x << " " << real << " " << imag << std::endl;
    }

//    output << std::endl;
//
//    for (double x = -10; x <= 10; x += 0.01)
//    {
//        output << x << " " << stepPotential(x) << std::endl;
//    }

    std::ofstream file;
    file.open("../data.txt");
    file << output.str();
    file.close();

    return 0;
}

#ifdef SA_LIB
}
#endif