#pragma once
#include <iostream>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/random.hpp>

#include "Neuron.h"
class Neuron;

using namespace std;

class Bond
{
    public:
        Bond(shared_ptr<Neuron> in_node, shared_ptr<Neuron> out_node);
        void Update_Wt();
        void Update_dWt(double &reward);
        void Update_GA(vector<double> &GA, double &lr);
        void info();
    
    
        shared_ptr<Neuron> in_node;
        shared_ptr<Neuron> out_node;
        double weigh;
        double d_weigh;

        vector<double> GA;
        double lr;
        double in_node_output;
        double out_node_output;
        

};