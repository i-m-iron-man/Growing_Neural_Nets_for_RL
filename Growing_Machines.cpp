#include <iostream>
#include <assert.h> 
#include <math.h>

#include "Growing_Machines.h"
#include "help.h"

using namespace std;
Help h2;

Growing_Machines::Growing_Machines(double evolution_lr, double decay_rate, 
                        double sigma, size_t population_size, 
                        size_t state_size, size_t action_size)
{
    this->evolution_lr = evolution_lr;
    this->decay_rate = decay_rate;
    this->sigma = sigma;
    this->update_step = evolution_lr/(population_size*sigma);
    if (population_size%2 == 0)
    {
        this->id_counter = 0;
        for(size_t i=0; i<population_size; i++)
        {
            cout<<"hi a new network is born with id:"<<i<<endl;
            this->population.push_back(make_shared<Net>(state_size, action_size,i));
            id_counter++;
        }
        this->mean_net = make_shared<Net>(state_size, action_size,id_counter+1);
        this->infer_net = make_shared<Net>(state_size, action_size,id_counter+2);
    }
    else{
        cout<<"please enter even number of population strength"<<endl;
    }

}


void Growing_Machines::initialize_population()
{
    assert(this->mean_net->Bond_List.size() == this->population[0]->Bond_List.size());
    vector<double> GA_plus = {0.0, 0.0, 0.0, 0.0, 0.0};
    vector<double> GA_minus = {0.0, 0.0, 0.0, 0.0, 0.0};
    double lr_plus;
    double lr_minus;
    double w_plus;
    double w_minus;

    double lr_mean = 0.0;
    double lr_std_dev = 0.05;
    double lr_noise;
    
    double GA_mean = 0.0;
    double GA_std_dev = 0.5;
    double GA_noise;

    double w_mean = 0.0;
    double w_std_dev = 0.5;
    double w_noise;


    for(size_t i = 0; i<this->mean_net->Bond_List.size(); i++)
    {
        for(size_t j=0; j<this->population.size()-1; j+=2)
        {
            for(size_t k=0; k<GA_plus.size(); k++)
            { 
                h2.rand_value_normal(GA_noise, GA_mean, GA_std_dev);
                GA_noise *= this->sigma;// ganoise = sigma*ga_noise
                GA_plus[k] = this->mean_net->Bond_List[i]->GA[k] + GA_noise;
                GA_minus[k] = this->mean_net->Bond_List[i]->GA[k] - GA_noise;
            }

            h2.rand_value_normal(lr_noise , lr_mean, lr_std_dev);
            lr_noise *= this->sigma;
            lr_plus =  this->mean_net->Bond_List[i]->lr + lr_noise;
            lr_minus = this->mean_net->Bond_List[i]->lr - lr_noise;

            this->population[j]->Bond_List[i]->Update_GA(GA_plus, lr_plus);
            this->population[j+1]->Bond_List[i]->Update_GA(GA_minus, lr_minus);

            h2.rand_value_normal(w_noise, w_mean, w_std_dev);
            w_noise *= this->sigma;
            w_plus = this->mean_net->Bond_List[i]->weigh + w_noise;
            w_minus = this->mean_net->Bond_List[i]->weigh - w_noise;
            this->population[j]->Bond_List[i]->weigh = w_plus;
            this->population[j+1]->Bond_List[i]->weigh = w_minus;

        }

    }

}

void Growing_Machines::info()
{
    cout<<"roll call all citizens:"<<endl;
    for(size_t i=0; i<this->population.size(); i++)
    {
        cout<<"hi i am network number:"<<this->population[i]->id<<endl;
        this->population[i]->info('b');
        cout<<endl<<endl;
    }
}


void Growing_Machines::update_mean()
{
    //1) subtract mean and divide by std dev for all total rewards of the population
    double mean = 0.0;
    double std_dev = 0.0;
    double var = 0.0;
    
    //finding mean
    for(size_t i=0; i<this->population.size(); i++)
    {
        mean += this->population[i]->total_reward;
    }
    mean = mean/this->population.size();

    //finding std dev
    for(size_t i=0; i<this->population.size(); i++)
    {
        var += (this->population[i]->total_reward - mean)*(this->population[i]->total_reward - mean);
    }
    std_dev = sqrt(var/this->population.size());

    //updating total rewards for each citize to total_reward-mean/std_dev

    for(size_t i=0; i<this->population.size(); i++)
    {
        this->population[i]->total_reward = (this->population[i]->total_reward-mean)/std_dev;
    }

    //update GA_list of mean
    //update lr for each bond of mean
    //update wt for mean

    for(size_t i=0; i<mean_net->Bond_List.size(); i++)
    {
        double wt_update = 0.0;
        double lr_update = 0.0;
        vector<double> GA_update = {0.0, 0.0, 0.0, 0.0, 0.0};
        for(size_t j=0; j<this->population.size(); j++)
        {
            double importance = (this->update_step)*(this->population[j]->total_reward);
            wt_update += importance*(this->population[j]->Bond_List[i]->weigh);
            lr_update += importance*(this->population[j]->Bond_List[i]->lr);
            for(size_t k=0; k<GA_update.size(); k++)
            {
                GA_update[k] += importance*(this->population[j]->Bond_List[i]->GA[k]);
            }

        }
        mean_net->Bond_List[i]->weigh += wt_update;
        mean_net->Bond_List[i]->lr += lr_update;
        for(size_t j=0; j<GA_update.size(); j++)
        {
            mean_net->Bond_List[i]->GA[j] += GA_update[j];
        }
    }

    if(this->evolution_lr > 0.001)
        this->evolution_lr *= this->decay_rate;
    if(this->sigma >0.01)
        this->sigma *= this->decay_rate;
    
    this->update_step = this->evolution_lr/(this->population.size()*this->sigma);
    

}


void Growing_Machines::initialize_infer_net()
{
    for(size_t i=0; i<this->infer_net->Bond_List.size(); i++)
    {
        this->infer_net->Bond_List[i]->weigh = this->mean_net->Bond_List[i]->weigh;
        this->infer_net->Bond_List[i]->Update_GA(this->mean_net->Bond_List[i]->GA,this->mean_net->Bond_List[i]->lr);
    }
}