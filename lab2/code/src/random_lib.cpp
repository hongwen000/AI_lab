#include "random_lib.h"

std::unique_ptr<std::default_random_engine> RandLib::engine;
std::unique_ptr<std::uniform_int_distribution<int> > RandLib::u;
std::unique_ptr<std::binomial_distribution<int> > RandLib::b;
std::unique_ptr<std::normal_distribution<double> > RandLib::n;
int RandLib::binomial_rand()
{
    if(!engine) {
        unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
        engine = std::make_unique<std::default_random_engine>(seed);
    }
    if(!b) {
        b = std::make_unique<std::binomial_distribution<int> >(1, 0.6);
    }
    return (*b)(*engine);

}

int RandLib::binomial_rand(double p)
{
    if(!engine) {
        unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
        engine = std::make_unique<std::default_random_engine>(seed);
    }
    std::binomial_distribution<> d(1, p);
    return d(*engine);
}

double RandLib::normal_rand()
{
    if(!engine) {
        unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
        engine = std::make_unique<std::default_random_engine>(seed);
    }
    if(!n) {
        n = std::make_unique<std::normal_distribution<double> >(2.1, 1.5);
    }
    return (*n)(*engine);
}

int RandLib::uniform_rand()
{
    if(!engine) {
        unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
        engine = std::make_unique<std::default_random_engine> (seed);
    }
    if(!u) {
        u = std::make_unique<std::uniform_int_distribution<int> >(0, 1000);
    }
    return (*u)(*engine);
}

int RandLib::uniform_rand(int L, int R)
{
    if(!engine) {
        unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
        engine = std::make_unique<std::default_random_engine> (seed);
    }
    std::uniform_int_distribution<> dis(L,R);
    return dis(*engine);
}
