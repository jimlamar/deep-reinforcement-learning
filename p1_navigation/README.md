[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 


After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

### Student Solutions
This project use a Fixed Target DQN algorithm with Replay Buffer, and Softmax Update. 
The model is linear neural network with 3 hidden layers (512, 1024, 128 nodes for the respective layers. It uses a RELU function at the output of layer. The model is trained with ADAM optimizer using MSE loss criteria. 

The model.pt weight file gnerated, used the following hyperparameter, run over 500 episodes.

#Model hidden layer nodes
FC1 = 512
FC2 = 1024
FC3 = 128

#Replay Buffer
BUFFER_SIZE = 100000     # Replay buffer size, 300 steps per episode  
BATCH_SIZE = 32         # Batch size  

#Learning Parameters
GAMMA = 0.995           # Discount gamma factor  
TAU = .001              # Soft update of target parameters
LR = 0.0001              # learning rate 
REPLAY_INTERVAL = 4     # How often replay implemented
EPS_START = .998         # Starting  epsilon, for epsilon-greedy action selection
EPS_END = 0.03          # Minimum value of epsilon
EPS_DECAY = 0.980       # Factor per episode decreasing epsilon

Training Progress
Episode: 10	Avg: 0.3	 EPS: 0.82	 Std: 1.10
Episode: 20	Avg: -0.1	 EPS: 0.67	 Std: 1.53
Episode: 30	Avg: 0.2	 EPS: 0.55	 Std: 1.76
Episode: 40	Avg: 0.5	 EPS: 0.45	 Std: 1.72
Episode: 50	Avg: 0.9	 EPS: 0.36	 Std: 1.92
Episode: 60	Avg: 1.4	 EPS: 0.30	 Std: 2.39
Episode: 70	Avg: 1.7	 EPS: 0.24	 Std: 2.57
Episode: 80	Avg: 2.1	 EPS: 0.20	 Std: 2.74
Episode: 90	Avg: 2.4	 EPS: 0.16	 Std: 2.90
Episode: 100	Avg: 2.4	 EPS: 0.13	 Std: 2.79
Episode: 110	Avg: 3.0	 EPS: 0.11	 Std: 3.34
Episode: 120	Avg: 3.5	 EPS: 0.09	 Std: 3.29
Episode: 130	Avg: 3.8	 EPS: 0.07	 Std: 3.19
Episode: 140	Avg: 4.4	 EPS: 0.06	 Std: 3.40
Episode: 150	Avg: 4.6	 EPS: 0.05	 Std: 3.48
Episode: 160	Avg: 4.7	 EPS: 0.04	 Std: 3.60
Episode: 170	Avg: 4.9	 EPS: 0.03	 Std: 3.80
Episode: 180	Avg: 4.9	 EPS: 0.03	 Std: 3.80
Episode: 190	Avg: 5.2	 EPS: 0.03	 Std: 4.21
Episode: 200	Avg: 6.1	 EPS: 0.03	 Std: 4.63
Episode: 210	Avg: 6.2	 EPS: 0.03	 Std: 4.52
Episode: 220	Avg: 6.9	 EPS: 0.03	 Std: 4.79
Episode: 230	Avg: 7.6	 EPS: 0.03	 Std: 4.95
Episode: 240	Avg: 8.2	 EPS: 0.03	 Std: 5.10
Episode: 250	Avg: 8.8	 EPS: 0.03	 Std: 5.19
Episode: 260	Avg: 9.4	 EPS: 0.03	 Std: 5.02
Episode: 270	Avg: 10.0	 EPS: 0.03	 Std: 4.78
Episode: 280	Avg: 10.5	 EPS: 0.03	 Std: 4.51
Episode: 290	Avg: 10.8	 EPS: 0.03	 Std: 4.04
Episode: 300	Avg: 10.9	 EPS: 0.03	 Std: 4.08
Episode: 310	Avg: 11.1	 EPS: 0.03	 Std: 3.94
Episode: 320	Avg: 11.1	 EPS: 0.03	 Std: 3.80
Episode: 330	Avg: 11.2	 EPS: 0.03	 Std: 3.85
Episode: 340	Avg: 11.3	 EPS: 0.03	 Std: 3.81
Episode: 350	Avg: 11.5	 EPS: 0.03	 Std: 3.56


Episode: 360	Avg: 11.6	 EPS: 0.03	 Std: 3.70
Episode: 370	Avg: 11.7	 EPS: 0.03	 Std: 3.84
Episode: 380	Avg: 12.1	 EPS: 0.03	 Std: 3.99
Episode: 390	Avg: 12.3	 EPS: 0.03	 Std: 4.13
Episode: 400	Avg: 12.3	 EPS: 0.03	 Std: 4.25
Episode: 410	Avg: 12.8	 EPS: 0.03	 Std: 4.35
Episode: 420	Avg: 12.7	 EPS: 0.03	 Std: 4.60
Episode: 430	Avg: 13.1	 EPS: 0.03	 Std: 4.61
Episode: 440	Avg: 13.1	 EPS: 0.03	 Std: 4.62
Episode: 450	Avg: 13.3	 EPS: 0.03	 Std: 4.67
Episode: 460	Avg: 13.6	 EPS: 0.03	 Std: 4.64
Episode: 470	Avg: 13.9	 EPS: 0.03	 Std: 4.53
Episode: 480	Avg: 13.9	 EPS: 0.03	 Std: 4.41
Episode: 490	Avg: 14.1	 EPS: 0.03	 Std: 4.46
Episode: 500	Avg: 14.1	 EPS: 0.03	 Std: 4.23


