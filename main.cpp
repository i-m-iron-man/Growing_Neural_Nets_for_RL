#include <iostream>
#include <vector>
#include <boost/circular_buffer.hpp>
#include "Growing_Machines.h"
//#include "Net.h"
#include "env.h"
#include <cstdio>
#include <thread>

#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

using namespace std;

void learn(shared_ptr<Net> net, const char** argv)
{
    Env env(argv);
    
    int state_size = env.state_size;
    
    vector<double> state;
    vector<double> new_state;

    for(size_t  i=0; i<state_size; i++)
    {
        state.push_back(0.0);
        new_state.push_back(0.0);
    }

    int action_size = env.action_size;
    vector<double> action;
    for(size_t i=0; i<action_size; i++)
    {
        action.push_back(0.0);
    }

    mjtNum max_epoch_time = 10.0;

    //reset the total reward in the net to 0
    net->reset();
    env.reset(state);
    env.set_epoch_time(); //begins the epoch time in obj net

    double reward = 0.0;

    while(env.d->time - env.epoch_start_time < max_epoch_time)
    {
        net->forward(state,action);
        env.take_action(action);
        //update new_state
        env.get_state(new_state);
        //get reward
        env.get_reward(state,new_state,action,reward);
        //update state
        state = new_state;
        //update network wts
        net->update_wts(reward);
    };
    net->get_wt_reward();//adds neg reward to total reward for higher wts


}

void heb_meta_learning(const char** argv, int population_size, int evolution_epochs, double evolution_lr, double decay_rate, double sigma)
{
    Env env(argv);
    int state_size = env.state_size;
    int action_size = env.action_size;
    
    Growing_Machines gm(evolution_lr, decay_rate, sigma, population_size, state_size, action_size);// many nets collected

    for(size_t epoch=0; epoch<evolution_epochs; epoch++)
    {
        gm.initialize_population();// initialize w,ga_list,lr for each bond for each citizen
        cout<<"*******EPOCH:"<<epoch<<"***********"<<endl;
        vector<thread> ts;
        for(size_t i=0; i<gm.population.size(); i++)
        {
            ts.push_back(thread(learn,gm.population[i],argv));
        }
        for(size_t i=0; i<gm.population.size(); i++)
        {
            ts[i].join();
        }
        cout<<endl<<endl;
        gm.update_mean();

        // infer the total reward ever 10 epochs
        //if(epoch%10==0)
        //{
            cout<<"############  inferring progress  #############"<<endl;
            gm.initialize_infer_net();

            learn(gm.infer_net,argv);
            cout<<"total reward:"<<gm.infer_net->total_reward<<endl;
            cout<<endl;
        //}
    }

}



int main(int argc, const char** argv)
{
   int population_size = 200;
   int evolution_epochs = 50;
   double evolution_lr = 0.2;
   double decay_rate = 0.995;
   double sigma = 0.1;


   heb_meta_learning(argv, population_size, evolution_epochs, evolution_lr, decay_rate, sigma);



    return 0;
}
