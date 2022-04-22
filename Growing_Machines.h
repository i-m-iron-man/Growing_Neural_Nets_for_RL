#include <iostream>
#include <vector>

#include "Net.h"
using namespace std;

class Growing_Machines
{
    public:
        Growing_Machines(double evolution_lr, double decay_rate, 
                        double sigma, size_t population_size, 
                        size_t state_size, size_t action_size);

        void update_mean();
        void initialize_population();
        void info();
        void initialize_infer_net();

        vector<shared_ptr<Net>> population;
        shared_ptr<Net> mean_net; //this will be used as mean for the population
        shared_ptr<Net> infer_net; //this will be used to check the progress
        size_t id_counter;
        double evolution_lr;
        double decay_rate;
        double sigma;
        double update_step;
};