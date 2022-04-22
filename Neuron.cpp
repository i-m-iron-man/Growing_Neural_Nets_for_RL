#include <iostream>
#include <math.h>

#include "Neuron.h"

using namespace std;

Neuron::Neuron(size_t id, char type)
{
    this->id = id;
    this->type = type;
    this->input = 0.0;
    this->output = 0.0;

    if(type != 'i')
        this->forward_pass_visited = false;
    else
        this->forward_pass_visited = true;
    
}

void Neuron::reset()
{
    this->input = 0.0;
    this->output = 0.0;
}

void Neuron::activate(char mode)
{
    this->output = tanh(this->input);
    //this->output = this->input;
}

void Neuron::info()
{
    cout<<endl;
    cout<<"Hi I am neuron with id: "<<this->id<<" type: "<<this->type<<endl;
    cout<<"output: "<<this->output<<endl;
    cout<<"input: "<<this->input<<endl;
    cout<<"incoming connections info"<<endl;
    for(size_t i=0; i<this->in_connections.size(); i++)
    {
        cout<<"id: "<<this->in_connections[i]->in_node->id<<endl;
        //cout<<"weight: "<<this->in_connections[i]->weigh<<endl;
        cout<<"output: "<<this->in_connections[i]->in_node->output<<endl;
    }
    cout<<endl;
}
//double Neuron::differ(char ) 