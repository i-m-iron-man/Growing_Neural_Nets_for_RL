# Growing_Neural_Nets_for_RL
The project is about exploring constructive or growing neural networks for topology optimization in a reinforcement learning setting.
## Step 1) Implementation of Neural Networks (NN) in C++ 
A simple and intiutive implementation of NN is needed so that their topology can be augmented later while learning and thier complete functionality is exposed.<br/>
Net.cpp & Net.h contain implementation of a NN, where one can create an NN as a object having member functions such as adding neurons, connecting neurons forward pass etc. It relies on:<br/>
        1. Class Neuron (defined in `Neuron.h` & `Neuron.cpp`) to represent a single neuron and it's functionality like activating it's input to output.<br/>
        2. Class Bond (defined in Neuron.h & Neuron.cpp) to represent a single bond or connection, connecticting 2 neurons and it's functionality like scaling the input             signal by a scalar.<br/>
### status: Done

## Step 2) Implementation of an environment
An environment class in C++, similar to an OpenAI Gym environment is needed so that our agent can operate in it.
The opensource and fast physics simulation library [MuJoCo](https://mujoco.org/) is used for the task.<br/>
`Env.h` and `Env.cpp` contain the implementation of the ["Half-Cheetah"](https://gym.openai.com/envs/HalfCheetah-v2/) environment.<br/>
 The XML file for the environment is imported from the `model` sub-folder.
### status: Done

## Step 3) Implementation of a Hebbian-Learning  Algorithm.
Hebbian learning is needed here for two main reasons:<br/>
1. If the architecture of network changes mid-execution during a back-prop algorithm then the learning may be adversly-effected due to catastrophic forgetting.</br>
2. In future, growth may occur such that it leads to formation of cyclic connections making the NN a general RNN. Again, Since the toplology of the RNN will keep            augmenting standard RNN-backprop algos like n-BPPT, RTRL may not be effective.</br>
A recent awesome method introduces in the work [Meta-Learning through Hebbian Plasticity in Random Networks](https://github.com/enajx/HebbianMetaLearning.git) is      used for this purpose. `Growing_Machines.cpp` & `Growing_Machines.h` contain a C++ implementation for the approach. Also, multi-threading is used to speed up the         training process.
### status: Tuning of hyper-parameters remaining

## Step 4) Implementation of some emperical Growing Rules.
### status: In Progress 
