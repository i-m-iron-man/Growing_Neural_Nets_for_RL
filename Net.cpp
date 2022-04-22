#include <iostream>

#include "Net.h"
#include <assert.h>

//#include "Bond.h"
//#include "Neuron.h"

using namespace std;

void Net::connect(shared_ptr<Neuron> in_neuron, shared_ptr<Neuron> out_neuron)//connects 2 neurons
{
    //step1: create a bond in the Net's bond list with in_node and out_node
    this->Bond_List.push_back(make_shared<Bond>(in_neuron , out_neuron));

    //step2: add the ptr to the latest bond to the output list of in_node 
    in_neuron->out_connections.push_back(this->Bond_List.back());

    //step3: add the ptr to the latest bond to the input list of out_node
    out_neuron->in_connections.push_back(this->Bond_List.back());
}


void Net::connect(shared_ptr<Neuron> neuron, vector<shared_ptr<Neuron>> output_neuron_list) // connects one input neuron to a vector of output neurons
{
    //connect neuron to all output neurons
    for (size_t i=0; i< output_neuron_list.size(); i++)
        this->connect(neuron,output_neuron_list[i]);

}


void Net::connect(vector<shared_ptr<Neuron>> input_neuron_list, shared_ptr<Neuron> neuron) // connects one input neuron to a vector of output neurons
{
    //connect neuron to all input neurons
    for (size_t i=0; i< input_neuron_list.size(); i++)
        this->connect(input_neuron_list[i], neuron);

}


void Net::connect(vector<shared_ptr<Neuron>> input_neuron_list, vector<shared_ptr<Neuron>> output_neuron_list) // connects one input neuron to a vector of output neurons
{
    //connect all input neuronsto output neurons
    for(size_t i=0; i< output_neuron_list.size(); i++)
        for (size_t j=0; j< input_neuron_list.size(); j++)
            this->connect(input_neuron_list[j], output_neuron_list[i]);

}



Net::Net(size_t state_size, size_t action_size, size_t id)
{
    this->id = id;// this is the network's id in the population
    this->id_counter = 0; // init the id counter to zero

    for (size_t i=0; i<state_size; i++)
    {
        // populate the input neuron vector
        this->Input_Neuron_List.push_back(make_shared<Neuron>(this->id_counter,'i'));
        this->id_counter++;
    }

    for (size_t i=0; i<action_size; i++)
    {
        // populate the output neuron vector
        this->Output_Neuron_List.push_back(make_shared<Neuron>(this->id_counter,'o'));
        this->id_counter++;
    }

    for (size_t i=0; i<128/*(state_size + action_size)*/; i++)
    {
        // populate the network neuron vector with state_size+action_size neurons
        this->Network_Neuron_List.push_back(make_shared<Neuron>(this->id_counter,'n'));
        this->id_counter++;

        //form the beginning connections: input_neurons-> latest network_neuron -> output neuron
        this->connect(this->Input_Neuron_List, this->Network_Neuron_List.back());
        this->connect(this->Network_Neuron_List.back(), this->Output_Neuron_List);
    }
    //this->connect(this->Network_Neuron_List[0],this->Network_Neuron_List[0]);
    for (size_t i=0; i<256; i++)
    {
        this->Network_Neuron_List.push_back(make_shared<Neuron>(this->id_counter,'n'));
        this->id_counter++;

        for (size_t j=0; j<128; j++)
            this->connect(this->Network_Neuron_List[j], this->Network_Neuron_List.back());

        this->connect(this->Network_Neuron_List.back(), this->Output_Neuron_List);
    }

    this->reset(); // total reward =0.0

};


double Net::f_recurse(shared_ptr<Neuron> n)
{
    if(n->forward_pass_visited == false)
    {
        if(n->type != 'o')
            n->forward_pass_visited = true;
        
        n->input = 0.0; //initialize each input to 0
        for(size_t i=0; i<n->in_connections.size(); i++)
        {
            double in_node_output = this->f_recurse(n->in_connections[i]->in_node);
            n->input += in_node_output*n->in_connections[i]->weigh;
            n->in_connections[i]->in_node_output = in_node_output;            
        }
        n->activate('i');
        for(size_t i=0; i<n->in_connections.size(); i++)
            n->in_connections[i]->out_node_output = n->output;
        return n->output;
    }
    else return n->output;
}


void Net::forward(vector<double> &input, vector<double> &result)
{
    assert(input.size() == this->Input_Neuron_List.size());
    assert(result.size() == this->Output_Neuron_List.size());
    for(size_t i=0; i<this->Input_Neuron_List.size(); i++)
    {
        this->Input_Neuron_List[i]->output = input[i];
    }
    for(size_t i=0; i<this->Output_Neuron_List.size(); i++)
    {
        result[i] = this->f_recurse(this->Output_Neuron_List[i]);
    }
    for(size_t i=0; i<this->Network_Neuron_List.size(); i++)
    {
        this->Network_Neuron_List[i]->forward_pass_visited=false;
    }
}

void Net::update_wts(double &reward)
{
    this->total_reward += reward;
    for(size_t i=0; i<this->Bond_List.size(); i++)
    {
        this->Bond_List[i]->Update_dWt(reward);
        this->Bond_List[i]->Update_Wt();   
    }
}

void Net::get_wt_reward()
{
    double wt_reward = 0.0;
    for(size_t i=0; i<this->Bond_List.size(); i++)
    {
        wt_reward += (this->Bond_List[i]->weigh)*(this->Bond_List[i]->weigh);
        wt_reward += (this->Bond_List[i]->lr)*(this->Bond_List[i]->lr);
        for(size_t j=0; j<this->Bond_List[i]->GA.size(); j++)
        {
            wt_reward += (this->Bond_List[i]->GA[j])*(this->Bond_List[i]->GA[j]);
        }
    }
    wt_reward *= 0.02;
    this->total_reward -= wt_reward;
}

void Net::reset()
{
    this->total_reward = 0.0;
    for(size_t i=0; i<this->Network_Neuron_List.size();i++)
        this->Network_Neuron_List[i]->reset(); //set neuron input and output to 0
    for(size_t i=0; i<this->Output_Neuron_List.size();i++)
        this->Output_Neuron_List[i]->reset(); //set neuron input and output to 0
    for(size_t i=0; i<this->Input_Neuron_List.size();i++)
        this->Input_Neuron_List[i]->reset(); //set neuron input and output to 0
}
void Net::info(char choice)
{
    if(choice=='a')//all
    {
        cout<<"neuron info"<<endl;
        for( size_t i=0; i<this->Output_Neuron_List.size(); i++)
        {
            this->Output_Neuron_List[i]->info();
        }
        for( size_t i=0; i<this->Network_Neuron_List.size(); i++)
        {
            this->Network_Neuron_List[i]->info();
        }
        cout<<endl;
        cout<<"bonds info"<<endl;
        for(size_t i=0; i<this->Bond_List.size(); i++)
        {
            this->Bond_List[i]->info();
        }
    }
    else if(choice == 'b') //only bonds
    {
        cout<<"bonds info"<<endl;
        for(size_t i=0; i<this->Bond_List.size(); i++)
        {
            this->Bond_List[i]->info();
        }

    }
    else if (choice == 'n') //only neurons
    {
        cout<<"neuron info"<<endl;
        for( size_t i=0; i<this->Output_Neuron_List.size(); i++)
        {
            this->Output_Neuron_List[i]->info();
        }
        for( size_t i=0; i<this->Network_Neuron_List.size(); i++)
        {
            this->Network_Neuron_List[i]->info();
        }
        cout<<endl;

    }
    else{
        cout<<"wrong options"<<endl;
    }

}
