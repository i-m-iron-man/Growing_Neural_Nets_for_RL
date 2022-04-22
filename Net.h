
#include <iostream>
#include <vector>

#include "Bond.h"
#include "Neuron.h"

using namespace std;


class Net
{
    public:
        Net(size_t state_size, size_t action_size, size_t id);
        //void Info();

        //network topology related functions
        void connect(shared_ptr<Neuron> in_neuron, shared_ptr<Neuron> out_neuron);//connects 2 neurons
        void connect(shared_ptr<Neuron> neuron, vector<shared_ptr<Neuron>> output_neuron_list); // connects one input neuron to a vector of output neurons
        void connect(vector<shared_ptr<Neuron>> input_neuron_list, shared_ptr<Neuron> neuron); // connects one input neuron to a vector of output neurons
        void connect(vector<shared_ptr<Neuron>> input_neuron_list, vector<shared_ptr<Neuron>> output_neuron_list); // connects one input neuron to a vector of output neurons
        
        //function approx commands 
        void forward(vector<double> &input, vector<double> &result);
        double f_recurse(shared_ptr<Neuron> n);
        void info(char choice);
        void update_wts(double &reward);
        void reset();
        void get_wt_reward();
        

        


    
    size_t id_counter;
    size_t id;
    vector<shared_ptr<Neuron>> Network_Neuron_List;
    vector<shared_ptr<Neuron>> Input_Neuron_List;
    vector<shared_ptr<Neuron>> Output_Neuron_List;
    vector<shared_ptr<Bond>> Bond_List;

    double total_reward;
    

};