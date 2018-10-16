#ifndef RANDLIB_H
#define RANDLIB_H
#include <random>
#include <utility>
#include <memory>
#include <chrono>
class RandLib {
public:
    static int binomial_rand();
    static int binomial_rand(double p);
    static double normal_rand();
    static int uniform_rand();
    static int uniform_rand(int L, int R);
private:
    static std::unique_ptr<std::default_random_engine> engine;
    static std::unique_ptr<std::uniform_int_distribution<int> > u;
    static std::unique_ptr<std::binomial_distribution<int> > b;
    static std::unique_ptr<std::normal_distribution<double> > n;
};

#endif // RANDLIB_H
