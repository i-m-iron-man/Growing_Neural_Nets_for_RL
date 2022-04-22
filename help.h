//#pragma once
#include <random>
#include <chrono>
#include <math.h>
using namespace std;
class Help
{
public:
    random_device rd{};
    mt19937 gen{rd()};
    void xavier(vector<double> &v, short);
    void rand_value_normal(double &v , double &mean, double &std_dev);
    void rand_value_0_1(double &v);
    void rand_value_minus_1_1(double &v);
};