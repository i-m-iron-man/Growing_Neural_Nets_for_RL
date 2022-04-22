#include <iostream>

#include "Bond.h"
#include "help.h"

using namespace std;
Help h;

Bond::Bond(shared_ptr<Neuron> in_node, shared_ptr<Neuron> out_node)
{
    this->in_node = in_node;
    this->out_node = out_node;
    this->in_node_output = 0.0;
    this->out_node_output = 0.0;
    h.rand_value_minus_1_1(this->weigh);
    h.rand_value_minus_1_1(this->d_weigh);
    h.rand_value_0_1(this->lr);
    for(size_t i=0; i<5; i++)
    {
        double random_value;
        double mean = 0.0;
        double std_dev = 0.2;
        h.rand_value_normal(random_value, mean, std_dev);
        GA.push_back(random_value);
    }
}

void Bond::Update_Wt()
{
    this->weigh += this->lr*(this->d_weigh);
}

void Bond::Update_dWt(double &reward)
{
    //will have to redo later
    this->d_weigh = this->GA[0]*(this->in_node->output)*(this->out_node->output);
    this->d_weigh += this->GA[1]*(this->in_node->output);
    this->d_weigh += this->GA[2]*(this->out_node->output);
    this->d_weigh += this->GA[3];
    //this->d_weigh += this->GA[4]*(reward);

}

void Bond::Update_GA(vector<double> &GA, double &lr)
{
    this->GA = GA;
    this->lr = lr;
}

void Bond::info()
{
    cout<<"hi I am a bond connecting nodes with id:"<<this->in_node->id<<" and id:"<<this->out_node->id<<endl;
    cout<<"wt:"<<this->weigh<<endl;
    cout<<"dwt:"<<this->d_weigh<<endl;
    cout<<"learning rate:"<<this->lr<<endl;
    cout<<"GA coeffs:"<<this->GA[0]<<" "<<this->GA[1]<<" "<<this->GA[2]<<" "<<this->GA[3]<<" "<<this->GA[4]<<" "<<endl;
    cout<<endl;
}
