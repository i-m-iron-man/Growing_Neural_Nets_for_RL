#pragma once
#include <iostream>
#include <vector>

#include "Bond.h"
class Bond;

using namespace std;

class Neuron
{
    public:
        Neuron(size_t, char type);
        void activate(char mode);
        void reset();
        void info();
        
        char type; // 'i' = input 'o' = output 'n' = network
        double input;
        double output;
        
        bool forward_pass_visited; 
        
        size_t id;
        vector<shared_ptr<Bond>> in_connections;
        vector<shared_ptr<Bond>> out_connections;


};
