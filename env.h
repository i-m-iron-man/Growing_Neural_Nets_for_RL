#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include "mujoco.h"
#include "glfw3.h"
#include <vector>

using namespace std;


class Env
{
    public:

        mjData* d;
        mjModel* m;
        mjvCamera cam;                      // abstract camera
        mjvOption opt;                      // visualization options
        mjvScene scn;                       // abstract scene
        mjrContext con;                     // custom GPU context
        GLFWwindow* window;                  // render window
        mjtNum epoch_start_time;              //when did the epoch start
        mjtNum render_time;               //keeps track of when to render

        vector<mjtNum> initial_qpos;
        vector<mjtNum> initial_qvel;
        
        int state_size;
        int action_size;


        Env(const char** argv);
        ~Env();

        void reset(vector<double> &sensordata);//get new sensor reading as well in the vector

        void get_state(vector<double> &sensordata);// float is used here coz weights of model will be float and implicit typr covension in libtorch is dubious
        void take_action(vector<double> &action);
        void set_epoch_time();
        void set_render_time();
        void render(/*mjtNum &render_time*/);
        void get_reward(vector<double> &state, vector<double> &new_state, vector<double> &action, double &reward);


};
