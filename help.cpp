#include <random>
#include <chrono>
#include <math.h>
#include "help.h"
using namespace std;

void Help::rand_value_normal(double &v, double &mean, double &std_dev)
{
    normal_distribution<> d{mean,std_dev};
    v= d(gen);
}
void Help::rand_value_0_1(double &v)
{
   v = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
 
}

void Help::rand_value_minus_1_1(double &v)
{
    double LO = -1.0;
    double HI = 1.0;
    v = LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO)));
 
}

void Help::xavier(vector<double> &v, short in)
{
    normal_distribution<> d{0,0.7};
    for(short i=0; i<in; i++)
    {
         v.push_back(d(gen)/(in));
    }
}
