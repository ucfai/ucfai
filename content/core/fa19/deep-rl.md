---
title: "Learning by Doing, This Time With Neural Networks"
linktitle: "Learning by Doing, This Time With Neural Networks"

date: "2019-11-13T17:30:00"
lastmod: "2019-11-13T17:30:00"

draft: false
toc: true
type: docs

weight: 1

menu:
  core_fa19:
    parent: Fall 2019
    weight: 1

authors: ["ahkerrigan", ]

urls:
  youtube: ""
  slides:  ""
  github:  "https://github.com/ucfai/core/blob/master/fa19/2019-11-13-deep-rl/2019-11-13-deep-rl.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-fa19-deep-rl"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/fa19/2019-11-13-deep-rl/2019-11-13-deep-rl.ipynb"

categories: ["fa19"]
tags: ["machine learning", "deep learning", "reinforcement learning", "DoTA", ]
description: >-
  It's easy enough to navigate a 16x16 maze with tables and some dynamic programming, but how exactly do we extend that to play video games with millions of pixels as input, or board games like Go with more states than particals in the observable universe? The answer, as it often is, is deep reinforcement learning.
---
```


```

**What does it formally mean for an agent to explore?**

**Why does an agent need to explore?**

**What are some ways we can allow for exploration?**

**What exactly is the limitation for using a traditional table for reinforcement learning for something like Doom?**

**If you could only replace state space or action space with a nueral network, which would make more sense to replace?**

Let's start with imports


```
import gym
import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
```

Next, we will write a function that can take some batch of data and return a one-hot encoded tensor. One-hot that instead of the integer representing some Nth option, we create a vector of 0s equal to the number of options, and the Nth entry is a 1.

For example, instead of inputting "5" into the network to represent state #5 out of 16 possible states, I input the vector [0, 0, 0, 0, 1, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

```
def one_hot(ids, nb_digits):
    """
    ids: (list, ndarray) shape:[batch_size]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    batch_size = len(ids)
    ids = torch.LongTensor(ids).view(batch_size, 1)
    out_tensor = Variable(torch.FloatTensor(batch_size, nb_digits))
    out_tensor.data.zero_()
    out_tensor.data.scatter_(dim=1, index=ids, value=1.)
    return out_tensor
```

Next, just for simplicity, we are going to write a funtion that will uniformly set the weights of any Neural Network layer to a uniform value. We do this because you can imagine that when trying to learn something through trial and error, you don't want to start with any assumptions.

Note: This is simply a possible intuition of why starting with uniform weights is the better option. There is no proof that it is in fact better, experiments have simply shown it leads to better results.

```
def uniform_linear_layer(linear_layer):
    linear_layer.weight.data.uniform_()
    linear_layer.bias.data.fill_(-0.02)
```

Now, let's create the frozen lake enviroment. As mentioned in the slides, frozen lake is simply a NxN grid in which an agent wants to go from the top left square to the bottom right square without falling in any holes. Occasionally, wind will cause you to move in random directions.


```
lake = gym.make('FrozenLake-v0')
```

We can go ahead and see what this enviroment looks like 

```
lake.reset()
lake.render()
```

    
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG



Let's take a step and see what it looks like. 

0 - Left

1- Down

2 - Right

3 - Up

**Note - If you (or even I) end up somewhere different than where it says we should be, it's because this enviroment is stochastic, meaning that it occasionally randomly places you somehwere you didn't want to go. This forces the model to deal with randomness**

**Why might stochastic be better than deterministic when training agents?**


```
s1, r, d, _ = lake.step(2)
lake.render()
```

      (Right)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG


As you can see, we died, so we went back to where we started. We can show this by looking at the done value

```
s1, r, d, _ = lake.step(1)
lake.render()
```

      (Down)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG


As you can see, we died, so we went back to where we started. We can show this by looking at the done value

```
print(r)
print(d)
```

    0.0
    False


For this enviroment, the only time a reward other than 0 is recieved is when you complete the goal, in which you recieve one. We are going to reset regardless of whether the randomness put us where we wanted to go or not. Now, let's build our agent.

```
class Agent(nn.Module):
  
    """
    Observation Space - How big is the state that the agent needs to observe?
    In this case, the only thing that changes about the lake is the position of the agent.
    Therefore, the observation space is 1
    
    Action Space - Similar to the O-Space, we can move up, down, left, and right 
    Because we need to measure the Q-value of every action, the action space in this 
    case will be 4
    """
    def __init__(self, observation_space_size, action_space_size):
        super(Agent, self).__init__()
        self.observation_space_size = observation_space_size
        self.hidden_size = observation_space_size
        
        # What is the difference between observation and state space?
         
        """
        Let's build the nueral network. In RL, you'll find that large networks 
        are largely unessesary. Oftentimes, you can get away with just 1 or 2 hidden layers
        The reason should be intuitive. What makes something a cat or a dog has many, many variables
        But "wich direction should I walk on a 2D grid" has a lot fewer.
        
        As you can see, the output layer is our action space size. This will be a table
        of our possible actions, each with a q-value
        """

        ## Create a simple 3 layer network using the observation space size as the input
        ## And the action space size as the output

        ### BEGIN SOLUTION
        self.l1 = nn.Linear(in_features=observation_space_size, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space_size)
        uniform_linear_layer(self.l1)
        uniform_linear_layer(self.l2)
        ### END SOLUTION
        
        # Why might a nueral network for deep RL be relatively smaller than what you might expect in something like image classification
    
    # Forward feed of our network
    def forward(self, state):
        obs_emb = one_hot([int(state)], self.observation_space_size)
        out1 = F.sigmoid(self.l1(obs_emb))
        return self.l2(out1).view((-1)) # 1 x ACTION_SPACE_SIZE == 1 x 4  =>  4
```

Now, let's write the trainer that will actually train the agent to navigate the lake.

For this cell, everything inside of train has been jumbled and switched around. Carefully review the steps to the deep-rl process and rearrange them in the correct order.

```
class Trainer:
    def __init__(self):
        self.agent = Agent(lake.observation_space.n, lake.action_space.n)
        self.optimizer = optim.Adam(params=self.agent.parameters())
        self.success = []
        self.jList = []
        self.running_success = []
    
    def train(self, epoch):
      
      # Let's start by resetting our enviroment
      # We don't want to just wander back and forth forever when the simulation starts
      # Therefore, we use a j value that stops our agent from taking more than 200 
      # actions in a simulation
        for i in range(epoch):
            s = lake.reset()
            j = 0

            """
            # Rearrange these in the correct order
                self.optimizer.zero_grad()
                s = s1
                target_q = r + 0.99 * torch.max(self.agent(s1).detach()) 
                self.optimizer.step()
                if d == True: break
                a = self.choose_action(s)
                j += 1
                loss = F.smooth_l1_loss(self.agent(s)[a], target_q)
                s1, r, d, _ = lake.step(a)
                loss.backward()
                if d == True and r == 0: r = -1
            """
            while j < 200:
                
                ### BEGIN SOLUTION
                # perform chosen action
                a = self.choose_action(s)
                # Every action gives us back the new state (s1), the reward, whether we are done, and some metadata that isn't important here

                s1, r, d, _ = lake.step(a)
                if d == True and r == 0: r = -1
                
                # calculate target and loss`
                # Now, we forward feed the NEW STATE, and find the MAX Q-value that the network thinks we acan get in the future
                # 0.99 here is our gamma. Usually, this should be a variable, but lets not worry about it right now
                # Does this seem strange? Aren't we training the network by observing what the network itself believes is true?
                # Kind of, but in the end the Bellman equation is recursive, so this is our only way of getting that info
                target_q = r + 0.99 * torch.max(self.agent(s1).detach()) # detach from the computing flow
                loss = F.smooth_l1_loss(self.agent(s)[a], target_q)
                
                # update model to optimize Q
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # update state
                s = s1
                j += 1
                if d == True: break
                ### END SOLUTION
            # append results onto report lists
            if d == True and r > 0:
                self.success.append(1)
            else:
                self.success.append(0)
            self.jList.append(j)
            if i % 100 == 0:
              print("last 100 epoches success rate: " + str(sum(self.success[-100:])) + "%")
              self.running_success.append(sum(self.success[-100:]))

    def choose_action(self, s):
      
      # 0.1 is our epsilon
      # Normally, we want some fancy way to degrade this (over time, we should be taking fewer random actions)
      # We will cover this a little more, but for this really, really simple example, we can just use a set epsilon
        if (np.random.rand(1) < 0.1): 
            return lake.action_space.sample()
      # Now, if we don't want to act randomly, we are going to feed forward the network
      # Then, we take the action that has the highest Q-value (max index)
        else:
            agent_out = self.agent(s).detach()
            _, max_index = torch.max(agent_out, 0)
            return int(max_index.data.numpy())
```

```
t = Trainer()
t.train(5000)
```

    /usr/local/lib/python3.6/dist-packages/torch/nn/functional.py:1351: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")


    last 100 epoches success rate: 0%
    last 100 epoches success rate: 0%
    last 100 epoches success rate: 1%
    last 100 epoches success rate: 1%
    last 100 epoches success rate: 5%
    last 100 epoches success rate: 14%
    last 100 epoches success rate: 18%
    last 100 epoches success rate: 15%
    last 100 epoches success rate: 23%
    last 100 epoches success rate: 7%
    last 100 epoches success rate: 12%
    last 100 epoches success rate: 25%
    last 100 epoches success rate: 25%
    last 100 epoches success rate: 35%
    last 100 epoches success rate: 36%
    last 100 epoches success rate: 41%
    last 100 epoches success rate: 39%
    last 100 epoches success rate: 40%
    last 100 epoches success rate: 40%
    last 100 epoches success rate: 35%
    last 100 epoches success rate: 45%
    last 100 epoches success rate: 31%
    last 100 epoches success rate: 40%
    last 100 epoches success rate: 37%
    last 100 epoches success rate: 41%
    last 100 epoches success rate: 44%
    last 100 epoches success rate: 28%
    last 100 epoches success rate: 40%
    last 100 epoches success rate: 28%
    last 100 epoches success rate: 32%
    last 100 epoches success rate: 39%
    last 100 epoches success rate: 38%
    last 100 epoches success rate: 34%
    last 100 epoches success rate: 34%
    last 100 epoches success rate: 44%
    last 100 epoches success rate: 39%
    last 100 epoches success rate: 33%
    last 100 epoches success rate: 36%
    last 100 epoches success rate: 37%
    last 100 epoches success rate: 38%
    last 100 epoches success rate: 38%
    last 100 epoches success rate: 47%
    last 100 epoches success rate: 36%
    last 100 epoches success rate: 37%
    last 100 epoches success rate: 30%
    last 100 epoches success rate: 34%
    last 100 epoches success rate: 39%
    last 100 epoches success rate: 41%
    last 100 epoches success rate: 27%
    last 100 epoches success rate: 44%


```
plt.plot(t.success)
```




    [<matplotlib.lines.Line2D at 0x7f1a9d0700b8>]




<img src="data:image/png;base64,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
AABJRU5ErkJggg==
">

```
plt.plot(t.jList)
```




    [<matplotlib.lines.Line2D at 0x7f1a9d042eb8>]




<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXcAAAD4CAYAAAAXUaZHAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0
dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3dd3wcxdnA8d8jyb3JRS5yk4uMbWzc
hG1woRiDC72FEkKNQyCYkpdgUkgCKQ6kkfJCCJCQvMSQEEIzNaaDccMFVyz3LrlI7rbKvH/cnnx3
ujtd29u9vef7+fhj3d7e7syWZ2dnZmfFGINSSilvyXE6AUoppVJPg7tSSnmQBnellPIgDe5KKeVB
GtyVUsqD8pxOAECHDh1MUVGR08lQSqmMsmjRot3GmIJw37kiuBcVFbFw4UKnk6GUUhlFRDZF+k6r
ZZRSyoM0uCullAdpcFdKKQ/S4K6UUh6kwV0ppTyoweAuIk+LSJmILA+Y1k5E3hGRtdb/ba3pIiK/
E5FSEVkmIsPtTLxSSqnwYim5/xWYFDJtBjDHGFMMzLE+A0wGiq1/04DHUpNMpZRS8Wiwn7sx5kMR
KQqZfBFwpvX3M8D7wH3W9L8Z3zjCn4lIvoh0McbsSFWC7fTlrgNUHqmiML8ZP3plBTkCb63YxS8u
G0x+88Z8+59Lue2sPizfVsk9E/vxjb8vok9BSyYP7szdzy8lL0eYPqGYxz9Yx4iebencuinLtlZy
4dBCtlccoU9BS55fsIXmTXK5ZFhXHnt/Ha/dMZZl2yrJb9aIsgPHqDxSxYVDCrnnn0v47pQBrC07
SL9OrfjBS8t5d3UZb9w5jgFdWgPwn8VbOXdgZ95bU8b2iiPcOKYXjXJPXK/Xlx/kD++V8u1zT2L5
tko+Ld3N+UMKObWoXVC+15UfpGz/MU7r055PS3fTJb8ZrZvmMX/DXiYP7hJ1m723poziji3p1rY5
AJv2HOIXb67mutFFdGzdhD4FLdlReYSV2/dzrLqW0b3b07xxLg+/uYapp3RhRM+2ABw4WsW7q8u4
aGjXeuuI9l2gzXsOs2nvIYrat+C5BZs5cryW4k4tubKkOy9+vpULhxby8pLtHD5WzQ1jerFg415W
79jPh2t3M7hrGyYM6EjHVk156LWViMDgrm04vU8HBhb6tndtreGFz7dy6bCu5OYILyzaSt+OLQEY
1qMts+ZvZsPuQ4wr7sDALq3rtt/nm/dx77+W8uevldC7wDf/toojfLy2nHYtmtC+ZWOa5uUysLA1
c9ftoWPrJrRqksePX1vJnROK2V5xhDNP6liXhntfWMa08b05qXMrfv7GKnZUHOWXVwzhl2+vYUCX
Vry5fCfnn1IIQLe2zdi89zDHqmq5bEQ3qmpqmb1sB5cO74qIALBo0z7+s3grE/p3YuGmvdx7Xn/m
b9hLfvNG9OvUikPHqnlrxU7mrtvDNaN6sK78EJdZv1++rZIvdx2gML8Zo3u3r9sXc9ft4Z2Vu1iw
cS+1xjB7+rigfbVh9yEWb95H66aNOGdgJwCOVdfwypLtTBnchb9+upFlWyt47NoR5OQIG3Yf4rH3
S7l8RHf2HjrGC4u28uevlXC0qpZrnvyMK0u6c/XIHny8djc/mb2SW8b1pvJIFeed3IlubZvz6H/X
8uy8TZx1Ukd+dulgcnOk3vHz5Efrad+yMZcM68bybZWs2F5J344tGdHTd74cOlbNOyt3ccGQQv69
aCuXDu9KXsD5tnRLBTkiDO7Whtpaw+WPf0pR+xbcNLYXg7q2qZuv8kgVH35ZTuc2TWndtBEndW4V
9bhOlMQynrsV3F8zxgyyPlcYY/KtvwXYZ4zJF5HXgJnGmI+t7+YA9xlj6j2hJCLT8JXu6dGjx4hN
myL2xU+bohmznU4CAK2a5nHgaHXd5xaNczl0vKbu88aZU1m0aS+XPTaXccUd+GjtbgC+PbEfd0wo
rpsvUn42zpwa9Nk/38aZU+v+HtI9n6VbKlj8g4m0bdE4YlqLZsymVZM8vvjxeWHXuXHmVEb97L/s
2n8MgJKebRnesy1PfLg+KC23/+NzZi/bwWt3jA06ERr6Llw+cgRqAw5r/za69Yw+PP7Bunp5DdS5
dVN27j9aLw8Azy/YzH3//oL7JvWnT0ELpv19Ud08Sx6YyNAH36n7PKxHPos3V/D5DyYy/KF36i1r
6INvU3G4qt56Gtpn/1ywhe/8exkAG34+hV73vx6Ux2geungQa3cd4G9zN/G3m0Yyvp/vwcbQdb51
13jO++2Hdeu95/klvLh4W9A8T99Qwtn9OwX9NvC4Cl3mm3eNo3/n1mG/9/9u5hurefyDdUHH/88v
HczVI3uE3S6PXH4Kn5Tu5qUl2wFY+9PJFH/vjaB5mjXKZdVDk4J+/9DFg7hudM96ywt3HgSmz78d
Lh/RjRcWbeX7Uwdwy7jeYX/vP1bCbZub/7qAOavLwn4XLxFZZIwpCfdd0g2qVik97jd+GGOeMMaU
GGNKCgrCPj2btQIDOxAU2P0OHvNN27z3cN20PYeOpywNS7dUAFBd2/CuPXCsOur3/sAOvhJr+YFj
9ebZWekLqEer6uc12nfhhCbZv412H6y/3nrrCgnsgfzBeN/h4+wP2UdVNcEr3brvCADVNbVRlxWv
fYfD7+PA4yCSysPHKbP2xaEo+yx0O4fbJqHHKMDLS7bVm+Z3JMwxHMq/fwKXHW077Tt8nO0VkfcX
wJEwx0xlhG3YEP922Gbt20j7AqKne1vFkYTWH69Eg/suEekCYP3vvwxtA7oHzNfNmqaU8rg7n1vC
wo17nU6GsiQa3F8Brrf+vh54OWD616xeM6OBSrfWt39SuptdUUppmWhd+UGWba1wOhkZYfm2yrSs
J9teYtnQXZwbVNUYXl26HWMMxhheWbo94h1WoL0pvDNOhwYbVEVkFr7G0w4ishX4ITAT+KeI3Axs
Aq60Zn8dmAKUAoeBG21Ic0pc++Q8OrZqwvzvneN0UlLmo7W7+Wjt7qTq8JyS7nf5nv/7j1O6PAlp
n6vfXKfsYBK4fP7u3bUYA03ycjheU8v0WYvZNLFfg79bvfMAANW1DV8I3CCW3jJXR/hqQph5DXB7
solKl7Iwdb+qYVv2Hmb3wWMM69E2od9HC3yhQTJQtpWCG2LX9TCWxUq0HRWnj9aWM644fe1u/u1W
caSqrn0hnliQ5nJIwvQJ1QznxIE27uH3uOR/P03/ipOUKSdlNJFiaibfKVz31PyYGly9IpUXxmg0
uGeoTD6ZExUpzzsrj7J48760piVWy7amp24/0YBRWnYwod/NWbWLqhjqqWNV44Urr8u44mUdKn7+
UyFNhYDUCpPmZE7t8Y+8x/Hq9NaDhrYRhGbJv1++/rfUvoQm1THwnF9/kNDvXl6ynS5tmtWbHulw
tCt0O3FNyJRzTkvuGcgYw3vWQxDpOM7mb9hLZYL9skM1fDLGn6N0B3YvSEVM3LKv4b71dnK6sO/0
+huiwT0D/WfxNv766ca0rKuqppYr/zSXG/46Py3r80qzaTpO/MBVxPikuatLneGy0FBvGCfyIxlS
KarBPQPtqDzRP9/uGFJjPe65eLO9/edjOV2cDvv+9R84Ws3nDtXxJxPM0t3dNB6R0nbUhQ2tsXS/
dMOW1uCe4dJ5vtoZ0NxwMsRq1vzNzJq/Jeo8bi0hp/R4CVvSTly4bfa7d0ujJ8GBAyeGETlcQYO7
itneg5Gf0Nu1/yhrdx1ocBkNBz2httbwaenuBhst083p9UcTS2+ZwHnsyks62z9Cs+wfz8ep9Qd9
l75kRKTBXaXEqJ/NYeJvPqw3/cMvy+Ne1v/N28Q1T87jzeU7U5E0lUYPvroy7PR0BLuzfvl+fD9I
sATuhsAdCw3uylbRRlmMZONuXy+MdI2eZwe3NrodOp7CsV/CZDHRfRbtzqO0rOE7wnRya5VbKA3u
Ku3CBb5Y6k7dXNWZrqcOAyXSQOof8z3aAF92NLw2tMRo6zzn1/XvCGNdblQZEqQTpcE9w7m9FLGu
PL4nIMPlx+VZdIUNuw/FNX8q6sZ3VWb2qKqLN1dQXRP/5SFTCiL6hKqKKtlC3J8+WJ98GpL9vRvO
tAwWafst3OTskA/J7tdZ8zfTplmjtK8/XYUVLbmrqJameHz4SCdETCMRJrtuG8tTkYYfcCsvXO8M
yR8TlUdS8+S1G9uHNLirqO6YtTjly0x34HN7oHW7dG2/eEvCgnsuUmNmvhv02Q2HnAb3DJdpb4eJ
JJaTIZGeN+kSGmQSuWVfsT2+ESTdEthSYV35oZjfkeu0TCksaHDPcOFeVOxV3/pH/buI1Tv3p239
dgfTqb9r+O1Qqepi6bb4dPEfP2H2F658I2fG0uCuXCEwcMbzyrpd+2N/g46dfc+3h9S5urF0539N
nN3SVQJ36s4l2d4y6To2NLirtGrowBbqnzyZ0Fsm1e9kjSSZRuFXl25PbJ1xrjLcHZZtrwTMwK5Q
6UqyBvcstmnPoYwcC71s/9G6Xg7xnNzJ9pZxYWE8YXaWHj9aG/+QE4nw0v6wg/Zzz1L7Dh3njEfe
5ysl3Z1OStxG/mwOrZvmsexH58U0vzNjftsvXSVAN1YxgXPpiuUtaG7YZFpyz1IHrcfPP1m32+GU
WOKMVPuzqCE5UDrGrLHjmmFHIM7AGpm00uCu0sqY6KWacGO0RJtfz+/EaGBMnP94dPs21OCu0mrn
/qNZEZCPZmBbRqocS1Pe3XwcaW8ZlZVe/HxrxO/CNZBmQm+ZUF55uAzcX0JV4WV9cC9z8VOP6eDG
15TFU7JZtSN9DzG5jZ1j5ajUq6qppexA+uJN1gf3kT+bw4GjqRk8SCUvnnHRl2+r5OE318SwTOuP
JGOhl0JpPBfQVFQj2FGIcGpsmUTXed8Lyxj50zlU1aSn2irrgzvAoWOZMaZFtjleUxu1eiPeseKV
Spdw18M3rNdGJjKGfCI0uGc5N/Zh9pfyHn5zDcMfeid1nf9cmFel7KIPMaF1l26QLXugttakbAxx
uxw9nhl3sm4+ZqKmLU2FDA3uWc5tPSHCHfeZ2FsmkkfeXsNj769zOhlRXfPkvKDPbtp+npAJY8uI
yN0iskJElovILBFpKiK9RGSeiJSKyPMi0jhViVXKSakocL1l1bsmnIaARGjQdZZ/88d7J5au3ZZw
cBeRrsB0oMQYMwjIBa4CfgH8xhjTF9gH3JyKhKr4ZHsXz7A0GKoUqzxcxZAfvx3TvOlu30q2QTUP
aCYieUBzYAdwNvCC9f0zwMVJrkMlYEcGvZm+qqaWIwH1vPGM5x4LNzYaJ8qJ0rpbt59TyfI/aCfA
3sPhe3NFS9uhY+kZFynh4G6M2Qb8EtiML6hXAouACmOMP/Vbga7JJlJ527VPzmNtma9bo1sDiXIn
Jy52ya6y7EDsL5hJRjLVMm2Bi4BeQCHQApgUx++nichCEVlYXp6e8Z8j0bpLZ83fsLfub90XKlbp
OlRqag3HqsP3IIr0PoFwU9N9bCfTW+YcYIMxphxARF4ExgD5IpJnld67AdvC/dgY8wTwBEBJSYme
0ioiL/WWSbYuwYk7G1dtPwfc9uwi3lqxq+5zQ7vg+QWb7U1QjJKpc98MjBaR5uJ7ZnwCsBJ4D7jc
mud64OXkkmg/L1YFpOsRZzu4NZi4NFkJSce48F4RGNhDVYcZKOntKPOnUzJ17vPwNZx+DnxhLesJ
4D7gHhEpBdoDT6UgnbZyazBJ1OxlO7j88blOJyMhdlxod7mp55DHjrVsd+5vPox53nQXIpN6iMkY
80PghyGT1wMjk1muSs5/V7mj5JAqyZ4T6Rpf3Ku8eGdrJ4M7RrrQsWXIvMLUWyuiPwgTy0uj9YRV
6oTVO2MfOtqE/O9WGtwz0LKtlUkvw81VUZ6+8KQwb8nsQy+Mp5TKY3jJlorULQx3BH4dWwZ33EKl
khsOrESFa+hzOj9FM2az8PvnpGRZ68sPJfzbft9/g+MBVUwvRHmjVSqlJojasxedODZ2H/T1U581
3x29YiLRkjvOB49Mkc3bacvew04nISiwA7y0OGwv45iku7eMm+8U47Vl7xEAKg7HN6ZMuguRGtw9
6OUl2xP6XdGM2VG/f39NWdDnL3cdSGg9mcprd3iZ7tWliR3ndoulzSsdNLhnqUTqtUP7+y7YuDfC
nKmVqqDqjlMu87i1DWRlCt+fm454nO7jT4O7ctSanfGV/hu6u/CqcPlOtoT4SenupH7vCi4pJQda
sd0dL23X4K4cNXdd/QDjwvPVc0Tg9S92pHV92SJdA4M1xNPB/f8+20TRjNkcOOru15o5IZMCqFuS
6pZ0pJurj5VsumrEydPB/emPNwANX0nd0gDifsHbya7Npuer8iLtLaPSIpEAuvvgcZZvS/4BqkA/
enVl0Odw6Vq0aV9K1uW1a0Yy19Z0l2e0/JR+GtxVXALHrcm0ErbGl2Ce2B4euGrUhhlZMhU0uCtX
SVf3ShW7TLuIh/qffy11OgkUzZjNoePhX/gxy6bx3zW4q7gEPtloR6HppQQfwFIqkhcWpWeYBrfJ
iuDugTs328TbmGz3gFPxFBLvfG5JXMv+TxKP62czN58/qUqai7OYMG8H9xgjhZsPXqXskO6qFj3F
0s/bwV01SOI8ywOrZewKEG6t4nVrulRms2sQNw3uZH6DkVPsuOMR0VJerJLZ/l455L2SDzt4O7jH
ePAbA09+tN7etLiIG4avjUSAp6yHz1R0qX7BRCSpKPzYVfWpBYHIvB3cLbEcnD+Zvcr+hLjEewFD
9ybzdG62nVhey6+2NXlbVgR3PYjri7eu/cTvUpwQ5YhYT4mXl2xLyfmjx01kdm0bbwd3PaAyTqIX
HRWfWAP2Q69lzx2t13g7uMfoWHX4J8e8zK2DpWloT5+9hxoemtatx4nfkQhPfSoN7gBsrzjqdBIc
oyXl7CQCc9ftaXC+WpcH9017UtM5wMmLmF1nYNYE9/1Hqxj24Ns6dgnw5EcbuPLxuUCSB7XLT3wV
XSx7r9ZofbndDtt095E1wX3plgr2Ha7i0f+udTopjtu89zDbK7P3bkXFzu0ldy9YV37QluVmSXDX
A1SpRBiTmhu0Kx6fy7z1DVcDOcWmUXcdlSXBXaVK4B26HeeDVgGkR6ybOZV10TPfXJ2yZXmJdoVM
SvStpwHFPRZsTM1bl+zgucMkhrhdo9UytrNrE2dJcFdKBdKQ7R5acleu8Kt3vqz7O9sKdVmWXcC3
j/XO1l46KmSK2P2yiWzi9gdcVGTGEFM9k9v3sJ7PkSUV3EUkX0ReEJHVIrJKRE4TkXYi8o6IrLX+
b5uqxCZODwA76FbNcHGMmqrsY9cFKtmS+6PAm8aY/sAQYBUwA5hjjCkG5lifHRGuYGLXLVA20i2p
4qHHS3olHNxFpA0wHngKwBhz3BhTAVwEPGPN9gxwcbKJtFs2B3wtlKmo9ACxnRvr3HsB5cBfRGSx
iDwpIi2ATsaYHdY8O4FO4X4sItNEZKGILCwvL08iGcnTervEePHBj2yS7t2nh0t4buwtkwcMBx4z
xgwDDhFSBWN8LW5h96kx5gljTIkxpqSgoCCJZMTGX28418VPyTkhmeNKT1bvS2XBZ/HmipgGK4tH
VY07j0I3tFMkE9y3AluNMfOszy/gC/a7RKQLgPV/WYTf2y5w+1bV1AJQE6a4mc3VMkpFk+qukDf8
ZX7qFqaiSji4G2N2AltE5CRr0gRgJfAKcL017Xrg5aRSqFzrk9LdTidBpYEbSqEqfnlJ/v4O4FkR
aQysB27Ed8H4p4jcDGwCrkxyHQnT8njDkjlv313t2E2ZSgEnzg+9TqRPUsHdGLMEKAnz1YRklquU
G2VjYUGDsf30ZR020t4yKhZeOkqMHvWep8Gd7G5QPXC02ukkKAd8um4PB481vO9TPcREJpxpP3hp
edLLcMOlU4M72T0wUiwnuPKe2ct2NDwTvruVVJ4fzoe8hv39s01JL8MNjdBZEdwb2tBu2BFKuZWe
H/GL54Jo10vqPR3c7dpoSinldp4O7oH1hRrnlYqfltoT44bt5ung7ifijo2tlFLpkhXBvSFaqlcq
snkbdDymTKTBHS3VKxWNjv6ZmTI6uB88Vs2Dr67kaFVN1Pk0eCul0umRt9Y4nYTMDu5/eLeUpz/Z
wLPzNof9PtbeMloto5TymowO7tXWML61Sd43/vG90lQkRynVEL2LTpuMDu7xiFY6/zTFLxBQSimn
eSa4HzxWzcNvruZ4dW3Y77XeXSnlRm58zZ6r/PadL/nf99fxwqKtTidFKaUc55ngfswqsVfXasld
pd68DXudToInHK+p5fdz1jqdjKzgmeDup0Fc2WHmG6udToJn/OqdL51OQlZI9jV7rhFYb7X30HGe
W7BZx5ZRKaUBXtnBrvdJeCa4+xljmPHvZby9cpfTSVFKKcd4rloGwr+AIrDknuq3yyillNt4Mrgr
pVS280xwj1ZrtbXiCM8v2JK2tCilVKzsag/M6Dr3cJUr4abd+JcFdidFKaVcJaNL7vsOH6/7W1+p
p5RSJ2R0cH/x820ArNqxv26atpUqpVSGB3e/4zXhn0pVSim3s6vOwRPBXSmlMpVdlQ0a3JVSykFa
co8isDHVoPXuSqnMoUP+NkA7yyilMpFdhVFPBPfA4QSMMQ0Gei3ZK6W8LungLiK5IrJYRF6zPvcS
kXkiUioiz4tI4+STqZRSKh6pKLnfCawK+PwL4DfGmL7APuDmFKwjZvowk1Iqk7iyzl1EugFTgSet
zwKcDbxgzfIMcHEy64iXMUarXZRSWS/Zkvtvge8A/qeI2gMVxhj/mLtbga7hfigi00RkoYgsLC8v
TyoRIhLXgPfrdx9Man1KKeV2CQd3ETkfKDPGLErk98aYJ4wxJcaYkoKCgkSTkZDDx2vSuj6llIrE
rqrkZEaFHANcKCJTgKZAa+BRIF9E8qzSezdgW/LJjC705Rta7a6UynYJl9yNMfcbY7oZY4qAq4B3
jTHXAu8Bl1uzXQ+8nHQqU0zr5JVSXmdHP/f7gHtEpBRfHfxTNqwjiIiEvEbP7jUqpZS7peRlHcaY
94H3rb/XAyNTsdzE0uLUmpVSKn46tkwD4tlAGv+VUl7nmeDuZzR0K6WU94I7NNxbJrR3jVJKOcaN
T6i60bZ9R7TeXSmV9TwT3P2l9WfmbmpwXo39Simv80xwV0opdYJngruOBqmUUid4IrjHG9a1Tl4p
5RbxDHoYD08E93jNWbXL6SQopZStPBHc4y2Ib913xJZ0KKWUW3giuIN9j/AqpVQm8kxwV0opdYJ3
grsW3ZVSGciV71B1C4G4Kt61s4xSyus8EdyVUkoF805wj+PWRgcOU0p5nXeCu1JKqTqeCO4G+57y
UkopO+mbmFJIx6FRSnmdJ4J7/GPLaJ27UsodtCtkFPFuHA3tSimv80RwByg7cDTmebXkrpTyujyn
E5AKx6treWP5zpjnf/2L2OdVSik76ZC/URyvrg36bLTiRSmVIbTOXSmlVMw8Edy1Z6NSSgXzRHAP
bR/9bP1eZxKilFIu4Y3g7nQClFIqQfqEahTatVEppYJ5IrjXaGxXSqkgngjuH35Z7nQSlFIqMTb1
CEk4uItIdxF5T0RWisgKEbnTmt5ORN4RkbXW/21Tl1yllPIWN9a5VwPfNsYMBEYDt4vIQGAGMMcY
UwzMsT4rpZQKw65a5YSDuzFmhzHmc+vvA8AqoCtwEfCMNdszwMXJJlIppbzKjSX3OiJSBAwD5gGd
jDE7rK92Ap1SsQ6llPIi1w4/ICItgX8Ddxlj9gd+Z3x9FMPedYjINBFZKCILy8u1QVQppVIpqeAu
Io3wBfZnjTEvWpN3iUgX6/suQFm43xpjnjDGlBhjSgoKCpJJhlJKqRDJ9JYR4ClglTHm1wFfvQJc
b/19PfBy4slTSimViGTGcx8DXAd8ISJLrGnfBWYC/xSRm4FNwJXJJVEppbzLrvHcEw7uxpiPidzQ
OyHR5SqllEqeJ55QVUopFUyDu1JKOci1XSGVUkolztUPMSmllEqMltyVUsqDxG2jQiqllHIvDe5K
KeVBGtyVUspBWueulFIqZhrclVLKQXYNP6DB3SPG9G3vdBKUUi6iwd0jcuyquFNK2Urr3FVUxq4X
MSqlMpIGd48wtr1mVymViTS4e0SHlk2cToJSKgE6toyKaEL/jvzk4kFOJ0NF8MxNI51OgnIxrXMP
o3FuRic/Za4o6Uarpo2cToaK4Ix+qXlHcF6ONpqr2Gl0TMK9553kdBJUhrjrnOKkl1HrcKt5h5aN
HV1/oi4Z1tXpJDhCg3sSvjG+t9NJUCnSs31zW5c/PgWl99o0x/bmjXODPp93cuf0JiBFHjh/oNNJ
iEofYgrD6R4i4Ybq/P3Vw5xICQBPXV9SN+WmMb0cSEf2eOiik+Oaf1j3/LDTH778lHrTwtXBDu8R
/vd2qg65mjTSatAG3Tkh+Tu0VMnoveXGvt0XDClM6Hcje7VLet0TBnSq+7tl04Tffa4aMOnkzlw9
skdcv4k0ZveVJd25fES3oGmtw7Sf5OZI2qtF8ptlTzvOyCLf+Zds4+a1o+M7LuyU2cHd6QTE6b5J
/eP+TSz1hbEekFeEBBGn5TfP3OBh1wsWvj91AN+dUv84EYQXvzmGX10xpMFlPP7V4XGvt7BN03q9
ev78tZIIc9vjtTvGpnV9gUTg0auG8t97zuC/94wH4LTeCQzpkUBQ0t4yYdSkuxIyglh3zuRBkess
w5XWAKbFUK/fJK/+bgytLwU4q3/HoM9NG6Vu919ZEv+F45axzlYdBXY+ads8+VJx1/xmSS/jlnG9
I6alR/vmXBbDBXp0AkHpvEGd61X9dGrdNO7lJGNAl9ZpXV8gEbhoaFf6FLSkb8dWbJw5lVnTRqds
+d+fOiDid00b1T9XUyGjg7uTnr1lVFLNIP+69TQe/+qIus+PhKl7BejfuRUPnD8wYrXN/ZP7h+1q
d9Wp3etNC6zGunZUD16fPi7OVEf2gzQ0WuU3b8Q1o4Jve/t2bJnw8p66/tS6v5+L40QOdzG/+5x+
Cacj1DkB1WvhPHrV0KjfJ9JAd+95J9UrdKa7TSs01X+4puH2q/6dW/HLGO5mGmowT9XYTPFusd4F
LfhKmHM1FTS4J6BVkzzG9LyawvIAAA6bSURBVO0Q144c3iO/bv6e7ZtzalE7JgWU5Nu2CF9aExFu
GtsrYn3rN87oE7aKIC83p14QCjxZv3lmH3oXJB4YQyXSzz7eNpMlD5xL07zgUk7PdsEnbaznaKfW
TejcxlcyHd27Xdylp9DV3JlEV8cWIXdZOTlS/2IesMKLhqa+a1/zxnm2tGEVtEr8yenzT2m4/apX
hxb12izCGVfcIez0v9/sq4pK57h7LZucaA+77cy+tjVUa3CPINpBGXoOxHJcBAbgwPn/cM0wXrp9
TL35H71qaFAdZCInXmi6oi3jf87tx6czzq4XaACuG92z7u9UPgmbilgSGlRjPUfzcnLo37kV353S
n9850sMJHrR63HwnSltMF+sCFFfsCZg5ljabn15i7dOQHSIIs6ePZVBXX3VJbshDVL+/ehgzLx0c
T8ocE+nY90+Pp+Tu3yeJStd1RIN7BLlRdnbglRdiK7W2aJJXt8wWAb8//5RChobpJnfR0K4M6tom
1uTGJFoJ4VtnF1OY34wVD06q913gAzj+XiKtUtAbJxUlxdB9Eav2LRsjIkwb34eOreqfrP6Adv/k
MI2bEQ6NnDjPpq+dVgQEHw9+/otstHkCzf/uhLo6/8AY/JuvRK/CAbh2lO/iLWHSf3Jhm7oSdOAT
st+e2I8LhhTaVqUQq1hjcrMId2aJPBg29/4Jcf/GCRkd3L89MXX1nIF+85XodXjPf8NXP5ubIzx4
0cn857bTefG20yOWAAd2ac2vrhhC93bN+O6U/jx5vX29EH52yWDGFXcICnp/vfFUnr6hhIkDo9fl
+v30kkG8fff4us/tAwYlE3x9vF8OuNv42SW+0tvfbx7J0zecyFuk3jBj+raneePclNTp9urQImwA
huhPhT5xXf19MKJnWwD6FLTg+WmnMX1CMTeN7cXUwV3qzRsYVN6409d24Q+ugT1aSqxlxuuXVwzh
3vNO4hvje/M/5/aL2CYTaNbXR/PA+QNjKmyEew6iddNGQQ/8+PePv+NCYMk9FdUYUwYHdzDIScHw
CvdM7Me1o3pw3snBx/rdE/sxPaAP+r+/eRq/v3pY3REYqeT+0XfOqps/0ENh7mBvPaOPq7pnZ3Rw
b2dTv98LTimkRZPIdbA927eo+/trpxXRu6Alw3u05cIIfdynTyimoFWTupJilzbx96qI9aC5ZlQP
/n7zqKBp44oLOLt/J3JzhGKrATJaV75rR/WkX6dWEb+/zspz4DoD1+MXqVT37C2jWfngJBqH6eXj
17ugRdjpTUJ6+IgI3zijT93nliF3RUC99XzzzD519e2B/MF5+oRiWjTJ456J/WiUm8Mt44IDYZOA
ev+WTfLq9fLokn9i2S988/Sw+WhI+5ZNuP2svuTkCN86uzjoAhtJj/bNuSnGHkgPXBC+Afymsb3o
1Dp4Xf6gHlj6jbUraGg1n//C179zK+6ZmPrhO6ZPKOanlwymRePgOx3//vQb0bMdFwwprLubDte7
DKB7u+ZsnDmVET2D20ACqyr9uuY3JZbrU0N3YamS0U+6XDGiO9/7z/Kkl9O/cyvOGdCJi4YW8sGX
5eTl5vDXG0fyzKcbefLjDQC8eNvpXPq/n8a13LfvHs+rS7fHXGJ+5VtjuPAPn8S8/G+d1Teu9AA8
fcOpvLpsO4VWcHv8qyNoFuHAfm7aaMoPHAuaFnhOP3btcFqHedDlT9eNoGmjXEp6tuVPH6wH4HtT
BtCuReOg3980phcHj1bzybo9LN1SEbSMG8f04gcv1d+3d5zdl8feX1dv+i8uG8xn6/dy+1l9OefX
HwR9Z0KujJHOv0jXz6Hd87n7nH5cNbI7//fZJm4e2wsR4YHzB0YcVuDZW0ax/0hVhCXaICRTr08f
x+It++pNm/K7j+o+v3T7GL7cdSDKIn0LveH0IvYfqeLr43sz+4sdrN4Z+Tfga0f61j8WA/C3m0bx
+vIdzHxjNQDd2jbjvkn9uWBIF7rmN+Pe807ikbfW1FvGm3f57oYeu3Y4rZo2YsX2Ss44qYBJv/Wl
f1DX1izftj/pR/fH9u3A9AnF3HB6UYPzPv7V4UGB+ScXD2JEz7ZMftSXpq+c2oPGeTncP7k/B45W
84f3SsMu57lpoznzl+8nle5YZHRwj1byi8ebd52ogii2Sqzd2zXn1jP78OTHG+ia34zhPdqSlyP1
HsmOpl+nVnz73NhLJ6d0i/yIeWiJFRp+qrV54zwOHqu2gpvvJOjerjm3nXniojApSt/7wP7SzRvn
cvh4DcacCPCTw1RXQPgxSL4epr9+00a5fGdSf255ZmG971pZJ1Hfji0pLTsYlKdwvnJqD75y6olu
kk0b5dSVOJs2yqWqprruu0htD/7nBUIbDkWkruE2cH+GlpL9PW5yRBjdJ3pf81Q9Y9C9XTO27D1S
r1phYGFrBha2rjct0NDu+WHbe/wldP8i/fsJ4NyTO7N65wEa5Qavr2mjHIyBY9W1nH9KYV1w79G+
Obee0YdPSnfz0drd5OYI3zzzxJ3W7Wf15ZG31tRbXv/OvrT6j7GxVm+Xts0bse9wFTeP7cXdzy8N
+4wHhD9fwsnJkaASfTSTBgUf718NKb3749E3zujDsq0VQcE9sL2iqEMLrh7ZnVnzt9g60mdGB/d4
/PzSwRS1b8HVf/6Mv9xwKgePVXPHrMVRe0p0aNmEe887qa7Odfb0cXy0trzBdf34wpPr6m/j9esr
h1AY5mGYH11wMl3aNGPOql1cNLSQI1U1jO0bvnuX30u3n867q8vIS0FXq5dvH8P7a8rjrhf96SWD
OLkwesPwzMsG85dPWtKhZRN6dWjBok37uHBIIVv2HubiYV3ZUXmUjXsO1c3/wPkDadEkl6qa8Bfa
708dwPh+BRS1b87d5/Tj0uFdeWnxNiYP7sy/Fm7l1oBqnEA/mDqQDi2bMCnBAbIevvwU/vbpprpH
2UM9etVQ2rdowuqd++uV+P903Yiojfih3rhzHHPX7eHs/h2Z/cWOqC9reejiQQzp5tsHL90+hhXb
K6Mu+283jeLVZdvpGKbH2K1n9OZYVU1dQ6+I8L0pAzjzpAIM8OGXvvPjzbvG8Unpnrrf/eqKIfz9
s00M71H/vPj+1AGMK/Ztj4cvOyVilRzAv249nTmrdnHBKYWs2XmQW8/wFRpCt9+MyQPYd6iKkwtb
B3Uzfvyrw1NWKPQLl+bBXdswfUIxFw8t5PkFW7h6VA8Gd8tnrXWnNGPyAFo3a8TUU8IXkFJBQm9Z
U7JQkUnAo0Au8KQxZma0+UtKSszChfVLb0oppSITkUXGmLA9NFLeoCoiucAfgcnAQOBqEXH3mJtK
KeUxdvSWGQmUGmPWG2OOA88BF9mwHqWUUhHYEdy7AlsCPm+1pgURkWkislBEFpaXN1yPrZRSKnaO
9XM3xjxhjCkxxpQUFKTmHZNKKaV87Aju24DAp1e6WdOUUkqliR3BfQFQLCK9RKQxcBXwig3rUUop
FUHK+7kbY6pF5FvAW/i6Qj5tjFmR6vUopZSKzJaHmIwxrwOv27FspZRSDbPlIaa4EyFSDmxK8Ocd
gN0pTE4m0DxnB81zdkgmzz2NMWF7pLgiuCdDRBZGekLLqzTP2UHznB3synNGD/mrlFIqPA3uSinl
QV4I7k84nQAHaJ6zg+Y5O9iS54yvc1dKKVWfF0ruSimlQmhwV0opD8ro4C4ik0RkjYiUisgMp9OT
DBF5WkTKRGR5wLR2IvKOiKy1/m9rTRcR+Z2V72UiMjzgN9db868VkeudyEssRKS7iLwnIitFZIWI
3GlN93Kem4rIfBFZauX5x9b0XiIyz8rb89awHYhIE+tzqfV9UcCy7remrxGR85zJUexEJFdEFovI
a9ZnT+dZRDaKyBciskREFlrT0ntsG2My8h++oQ3WAb2BxsBSYKDT6UoiP+OB4cDygGkPAzOsv2cA
v7D+ngK8ge/FqKOBedb0dsB66/+21t9tnc5bhPx2AYZbf7cCvsT3chcv51mAltbfjYB5Vl7+CVxl
TX8c+Kb1923A49bfVwHPW38PtI73JkAv6zzIdTp/DeT9HuAfwGvWZ0/nGdgIdAiZltZj2/GNkMTG
Ow14K+Dz/cD9TqcryTwVhQT3NUAX6+8uwBrr7z8BV4fOB1wN/ClgetB8bv4HvAxMzJY8A82Bz4FR
+J5OzLOm1x3X+MZnOs36O8+aT0KP9cD53PgP38iwc4CzgdesPHg9z+GCe1qP7UyulonppSAZrpMx
Zof1906gk/V3pLxn5Daxbr2H4SvJejrPVvXEEqAMeAdfCbTCGFNtzRKY/rq8Wd9XAu3JsDwDvwW+
A9Ran9vj/Twb4G0RWSQi06xpaT22bRk4TKWeMcaIiOf6rYpIS+DfwF3GmP0S8AZ7L+bZGFMDDBWR
fOA/QH+Hk2QrETkfKDPGLBKRM51OTxqNNcZsE5GOwDsisjrwy3Qc25lccs+Gl4LsEpEuANb/Zdb0
SHnPqG0iIo3wBfZnjTEvWpM9nWc/Y0wF8B6+Kol8EfEXtALTX5c36/s2wB4yK89jgAtFZCO+9ymf
DTyKt/OMMWab9X8Zvov4SNJ8bGdycM+Gl4K8AvhbyK/HVy/tn/41q5V9NFBp3e69BZwrIm2tlvhz
rWmuI74i+lPAKmPMrwO+8nKeC6wSOyLSDF8bwyp8Qf5ya7bQPPu3xeXAu8ZX+foKcJXVs6QXUAzM
T08u4mOMud8Y080YU4TvHH3XGHMtHs6ziLQQkVb+v/Edk8tJ97HtdMNDko0WU/D1slgHfM/p9CSZ
l1nADqAKX93azfjqGucAa4H/Au2seQX4o5XvL4CSgOXcBJRa/250Ol9R8jsWX73kMmCJ9W+Kx/N8
CrDYyvNy4AFrem98gaoU+BfQxJre1Ppcan3fO2BZ37O2xRpgstN5izH/Z3Kit4xn82zlban1b4U/
NqX72NbhB5RSyoMyuVpGKaVUBBrclVLKgzS4K6WUB2lwV0opD9LgrpRSHqTBXSmlPEiDu1JKedD/
A4CXaMifAAi1AAAAAElFTkSuQmCC
">

```
plt.plot(t.running_success)
```




    [<matplotlib.lines.Line2D at 0x7f1a9ccc6940>]




<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXAAAAD4CAYAAAD1jb0+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0
dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3deXjb13Xg/e8FSADcwA3cKYnaFy9a
vcSOHVuOE8fZHCvNOqknTetJxn3rJs0kTdun27ydpPN2srhNM3GaNHknS7PYjt0sjR1JjpfYsiVZ
sq3NpEjJ4gqCO0ACIIA7fwA/CCSxkgBIgOfzPHpEgiBwIQEHB/eee67SWiOEEKLwmJZ7AEIIIRZH
ArgQQhQoCeBCCFGgJIALIUSBkgAuhBAFqiSfd+ZwOHRHR0c+71IIIQresWPHXFrrhvmX5zWAd3R0
cPTo0XzepRBCFDyl1MV4l8sUihBCFCgJ4EIIUaAkgAshRIGSAC6EEAVKArgQQhQoCeBCCFGgJIAL
IUSBkgAuhFixfv7yAMNTvuUexoolAVwIsSK53D7u+/5xvvt83D0sAgngQogVqnPIDcDro9PLPJKV
SwK4EGJF6nJOAXBxxLPMI1m5JIALIVakTqeRgc8s80hWLgngQogVyZhCcbl9TPsDyzyalUkCuBBi
Rep0urHbwg1TZR48PgngQogVZ8zjx+X2ccvWRgBeH5EAHo8EcLHiaK0JhfRyD0MsI2P++7btkQAu
GXhcEsDFivPjY71c9/mDzAZDyz0UsUw6IxUoe9fVUmUrkQCegARwseIcvTDK8JQPp+zAW7U6h9xU
WMy01ZSxrr5cAngCEsDFinPBFX6xDk54l3kkYrl0Od1saqxEKcXaunKZA09AArhYcbpd4Y0bQ5MS
wFerTucUmxqrAFhTV07v2AxBWRdZQAK4WFEmvbO43OGpk+XIwJ/pdPH+rz+HP1Cc8++Pnxrk97/z
4opeJJ6YmWVo0sfmpkoA1tVV4A+GCvYN/YnTQ9zw+YOcH3Zn/bYlgIsV5YLr8rbpfL9gQyHN//vz
0xzpGaV/vDh3/z1+eohfn3Hyav/Ecg8loa5IBcrmxnAAX1tXDsDFAp1GGZr00j/hpdJakvXblgAu
VpSeSAA3mxSDeQ7gj58e4uxguPph2F2cC6jGv+/BM85lHkliRg+UzZEplHX14QB+qUAXMkfcfgBq
yy1Zv20J4GJF6R72oBRc2WrP6xSK1poHDnZSYTEDFG0PaiOAHz63cgN455AbW6mJttoyAFqqbZhN
ioujhdnUasTjo7qsFEtJ9sOtBHCxolwY8dBWU8aauvK8TqEcPOPk9MAk/89tmwGi8/DFZGJ6llGP
n4YqKy/3TuBcoXPKnU43GxsqMZsUACVmE201ZQXb1GrE7ae+MvvZN0gAFytMj8vDekcFzXYbg5Ne
tM79YpvWmgcOdbK2rpyP3tiBSRVnBt4Tacv6u9evA1ZuFt7ldEfnvw2FXAs+7PbhqLTm5LYlgIsV
Q2tNz7CHDY4KmqtteGdDTM7kvgvdk68N83LvBPfduhFriZm6CmtxBnBXeHHwjiubaa22cejsygvg
bl+AvvEZNjdVzbl8TV05rxdoX/ARtw+HZOCi2LncfqZ8AdY7Kmiy2wByvpCpteYrv+6kraaM9+xu
B6ChylqUUyg9wx5MCtbWl7N/eyNPd7rwBYLLPaw5zkcqUDbNz8DryhmbnmXSO7scw1qSEY+f+grJ
wEWRMxbYOiIZOOQ+gD/T5eLEpXH+660bo4tMDVXFmYF3uzy015ZjLTGzf1sj0/4gR7pHl3tYc3TO
KyE0GKWEhVaJMhsMMT49K3PgovgZNeAbHJU0RzLwoRxWohjZd0u1jffubY9e7qi04IqUfhWTCyPh
9QWAGzY6sJWaVtw0SqdzCovZFA3YhjWR7wttS/2oJ/w8kjlwUfS6XR5KzYq22jIa7eEnfC4z8Oe6
Rzh6cYxP3BKe+zYYGXg+FlDzxVhfMAK4rdTMDRsdHDrrXFGPs2vIzYaGCkrMc0PT2kgteKEtZBpT
cTIHLopej8vNuvoKzCYVWUy05DSAP3Cwk8YqK+/bt2bO5Q2VVvzB/Cyg5svwlA+PPxgN4AD7tzXy
+ug054dXzuJgZ6SJ1Xx2Wym15aVcLLAAbmziqZcMXOTTtD+Q99PAe1weOuovB5gmuy1nUyhHukd4
vnuUj79pI7ZS85yfNVSFX2zFtBvTaBA2P4ADHDo7tCxjmm/GH+TS2HR0B+Z8a+vKV8Qc+PCUj5E0
nxsjnvD16iskAxd59A+/eo3bv/gUr/blp2dGKKS5MDLNhobLAabZbs1ZBv6Np3twVFr44LVrF/ys
IZItFdNCZk+cAN5aU8a25qoVs63+/LAbrYk2sZpvbX3Fsk+hjLh9vOMfn+bTPz6Z1vVdU5E58CrJ
wEWeaK15/PQg/mCIP/z+cdy+3E8l9E/M4A+E5gSY5mpbTnZjjrh9PHnOyYE97ZRZzAt+7ijCDPyC
y4OlxERrTdmcy2/b3sjRi2NMzCx/ed78Jlbzra0ro29shsAyndQUCmk+/eOTDE360m6s5fL4sJhN
VOWgkRVkEMCVUmal1EtKqZ9Fvl+vlDqilOpSSv1QKZWbzwgi7zqdbnrHZrh7Txuvj07zF4+8kvOF
rngZYpPdhsvtz3pr18dO9hMIae7e0x7350YG7iqiDLzb5aGjvjy6Pd2wf1sjwZDmqdeGl2lkl3U6
pygxKdbFTKPFWldXQSCkGVimgz6++UwPh88N01Kd/i5hYxu9UirldRcjkwz8fuBMzPd/D3xJa70J
GAM+ls2BieVjlJb9t7du5Y/fvIWfnujnx8d6U/6edzbIxPRs3D+pnuw90RLC2CmUcCmhcyr1CzaT
/tYPHe/lyjY7W5vjz7VWl5VSalZFlYHPX18w7FpTS2156YooJ+wcctPhqEjY9ClaSrgM0ygnL43z
9/9xlrfsaOKjN3Yw7Q8ylcYn0xG3L2c14ABp5fVKqXbg7cDfAZ9S4beT/cCHIlf5DvDXwNdyMEaR
Z4fOONnRYqeluoz7bt3E890j/NWjp9iztiZ6SkqsYEjzL09388UnXsOXIFv+xC0b+ewd2xLeZ/ew
hwqLObqACNAU2cwzNOmlvbY80a8y4w9y0/88xKdu38qHrls4px3r3OAUr/ZN8pfv2JHwOiaTor7C
mlYGHgppbvmHJ7nv1o28/5rk971cgiHNxRFP9IT3WGaT4tatjRw+5yQY0gsy9HzqcroTvqnC5VLC
iyPT3LgpX6MKHzLyhz84TpPdxv/33p08+Vr4zW5owovdVpr0d0c8/pzVgEP6GfiXgc8AxquzHhjX
WhtvQb1AW7xfVErdq5Q6qpQ6Ojy8/B/TRHLj036OXhyNvtjNJsWX37+LcouZ+773Et7ZuVuve1we
3vf15/j8L89y85YG/vIdOxb8ubq9mn8/2Z80C+9xeehwVMz5qGlk4IMTyQNpl9ONy+3nS79+bcH4
5nv4eC8lJsW7d7UmvV5DlTWtDNw55eP10WmOXxxPed3l0jc2w2xQz/l0E+vWbY2MTc9y4tJYnkd2
mS8Q5MKIJ+H8N4SfDxazKa8ZuNaazz38Cv3jXh744C6qy0svPy/TWJ9xTflyto0e0sjAlVLvAJxa
62NKqVsyvQOt9YPAgwD79u1bOTsGRFy/eW2YkA6/qA2Ndhv/6307+c//+iJ/+7PT/I/3XEUopPk/
z1/k8788g8Vs4svv38W7d7XGneuzlZr5s0deodPpZktT/AzrwoiHq9qq51yW7gulM3IAwPCUjx++
eIl7buiIe71gSPPIS33csrUxZV1uQ5U1rQXU3rFwMOkdX/7ytkS6I02s1jviB8ebtzRgNikOnnGy
d11dPocW1ePyENKwKcHzA8LJRHttGa/nsS/4D164xM9fHuAzd2yN/ttE2zykmIvXWuPy+HO2iQfS
y8BvBN6llLoA/BvhqZOvADVKKeMNoB3oy8kIRV4dOuukvsLCzvaaOZffsrWR//KmDXz/yOt865ke
PvwvR/irx05x3fp6Hv/km7hrd1vChRqj3jhRuZo/EOLS6PSCDLGmPNwEP1Ug7XS6KTUr9q6r5WtP
nk/YoOmZLhfOKR8H9sT9sDhHeDt96gy8L3L0Wt/Yyu1VfSHOAnGs6rJS9q2rXdZ58M6h5BUohjV1
+Wsre3Zwkr/591PctNnBx2/eGL3caLSW6nnp9gXwB0I5nQNPGcC11p/TWrdrrTuADwCHtNYfBg4D
741c7R7g0ZyNUuRFIBjiN68Nc8vWxrhzoZ9+y1Z2r63hb392mpd7x/nC3Vfx7Y9eE81IEmmutrGj
xc7hBAHi9dFpQhrWN8wNMEqpcF/wFJlO55Cb9Y4KPnX7FgYnvfz4aPwF14eP91JdVsr+OHPB84U7
EvpTLo72RgJ3/7h3yQcFnxuc4gMPPsfTndmdauxxeai0liTNBG/b3sjZwanoG1K+dTrdmFTiNxnD
uvryvPRD0Vpz/w9OYC8r5Yvv24Up5vVgKzVTU16a8pOhsQtzJcyBx/NZwguaXYTnxL+ZnSGJ5fLS
pXHGp2ejGfN8pWYTX/3QHu69eQP/8cc384Fr16ZdHhWuNx5lfHphk6hoF8I4VRLGwQ7JdDmn2NxY
xQ0b66NZ+PzSwynvLL86Ncg7d7bM6XuSSEOllWBIMxZnvLGMAO4PhpZctfLzl/t5vnuUj3zzBf7i
p6/gyVL9fXfkkIxk/1c3bW4A4LnzI1m5z0x1OadYV1+xYFfsfGvrypn0BuI+j7LpVP8k54am+PRb
tsxZWDeEE4vk/9/GJ7hcbaOHDAO41vpJrfU7Il93a62v1Vpv0lr/jta6eGquVqmDZ5yUmBQ3bXEk
vE5rTRl/duf2aElXuvZvaySkw3Ps8/VE52gXBvCmFJt5vLNBXh+dZlNjJUop/ui2zfSNz/DQ8blZ
+C9fGcQ7G+JAgtrv+YzNPKm6Ehpz4PO/XowTvRNsbqzkD25az/eOvM7bvvI0R7qXHlCNU46S2dpU
RU15aVbubzE6h+L3QJlvbZ5KCQ+ddaIU3La9Ke7Pm+ypN5kZz51cbaMH2YkpYhw+6+SajrqUpVGL
sbO9hvoKS9x51h6Xh7oKCzVxTu1utlsZnEi8aaJ7OLz4ZWy/vnmzg51ravjq4S5mY3bs/eR4Lxsc
FexaUxP3duZLdzt93/hMdN62dwnz4FprTl4aZ19HLX/+9h388N43APCBbzzPf//Z6ZTVNYl4Z4P0
jc+kDOAmk+KajjqO9OS/P/hsMESPK3kFiiFfXQkPnXWys70m4fRHOp8MjT4ouZxCyc3+TlFwLo1O
c25oir94+/ac3L7JpLhlayO/PjNEIBia0y40WYbYZLfhC4SYmJmNG+CNChSjAZJSivtv28Tvffso
j7zUx/v2reHS6DQv9Izy3966Ne0pn8sZeOIArrWmb2yG9+5tj+5eXayLI9NMzMxGF4+vXV/HL++/
iS/88mxkB6CT737sugVb4VO5NDqN1szpMZPIdevreOL0EAMTM7RUZ3Y/6fjNa8M8cryX4Lz34hl/
kEBIJ+yBEmtN7eVa8FwZnvJxsnecT755S8LrNFXbcLl9zAZDlJrj58HGHHidZOAi14wDbhPNf2fD
/m2NTMzM8tKluTXTyQJ4qpN5upxuzCZFh+PylM6tWxu5ss3OVw93EQiGePh4H0rBXbtTV58Yoh0J
k2TgLrcfXyDElqYq6iosS1oAPNkb/jfZGfMJocJawn+/60q++7HruDQ6zb8+25Px7XYnWV+Y7/oN
9QBZP6XH7QvwuYdf5p5vvcAzXS5O9U3M+dM97OaKVnv0/pOpsJbgqLTmtCvhk+ecaJ38tdBst6F1
8ufHiNtHdVlpwp2l2SAZuADCHxk76svZ0JA6C1qsm7Y4KInUG1/TEa6p9fgCDE36Egdw++Wa223N
9gU/7xxys66+fM7CpFKKP9q/mXv/zzEePdHPwy/18oYN9bRlkL1WWUuwlpiSLkwac95tNWW01ZQt
KQM/cWmcslJz3GmEN252cOvWRh55qZ/P3rFtwWEHycQeU5fK9hY7VbYSjvSMZPRml8xvz7v4zE9e
pm98hv9y8wY+efuWlAuVqaytK8tpBn74nJMmu5UrWhc+3wzN1ZcPHEn0qcgV6YOSS5KBC6b9AX57
foT92+Iv2GSL3VbKtevr5pQTxmtiFStVzW2ncypu0Lt9RxPbW+z8zb+f4uLIdMLGVYkopXBUJt9O
b2Tc7XVltNeW0beERcyTl8a5qq06YXA+sLcdl9vH012ujG63Z9iDo9JCdVnqdQ2zMQ+ehQx8xh/k
rx87xYe+cYQSk+InH38Dn7tz+5KDN4QXMnM1B+4PhHjqNRf7tzUmnW5rSuPIP5fbhyOHuzBBArgA
fts1gj8QitsrI9v2b2vk3NBU9CNwugE8XsmWPxDiwkj8AwDCWfgmJr0BykrNvO3K5ozHmmo7vZFx
Gxl43/jMoro2zgZDvNo/yc411Qmvc+vWRmrLS3kojaZisdKpQIl13fo6ul0enEto43vs4ih3PvA0
3/7tBf7zDR388v6bs7rDc219BQOR9sPznRuc4tjFxbcEOHphFLcvkDKZSWeX8IjHj6NKMnCRYwfP
Oqm0lkSnNXLJmFc05twvpJijtZSYqE9wtNqFEQ/BJItfb72imV1ravidfe1ULKIfc6rT6fvGZqgu
K6XKVkp7bRne2RAjnszrk88NTuEPhObMf89nKTHxrp2tPH56KKPe3T0jGQZwYx58EdUo3tkgn//F
Gd77v5/DHwjx/T+4jr9+1xVxe64vxdq6ckKaOWsOs8EQX3riNe584Gk++q8vLHpT1cGzTiwlJm7c
lHw+vq7CgsVsSh7A3bntgwISwFc9rTWHzzq5abMjp4sthg0Nlax3VETLCXtcHlqrbUlf5Ilqbo3t
14nqh00mxSP/9Qb+9t1XLmqsjkpr0iqU3rHp6Lx6W6Q6YjHz4Ccii7rz2xfMd/eedvyBEL94ZSCt
253yzjI85UvYAyWeK1vtVFjMHOnJrB78ld4J3vmPz/D1p7r5wDVr+dUnb+aGjYn3EyzFunmlhGcH
J7nrq8/ylYOdbGyoYNIbiC7eZurQWSdv2FBPuSX5G75Sika7NeEUymwwxNj0rMyBi9w6PTDJ4KQ3
p9Un8926tZHfnh9h2h9+oaVaYGuujr+dvtM5hVKwMcnC61Ia6TdUWRnx+BOeANM3PkN7bTiAG38v
pifKyUvj1FdYoreRyNXt1WxqrOTh4+lNo1xwhQPcekf6m65KzCb2ZjAP7g+E+OITr3HXPz/LlDfA
tz96DZ+/+yoqc3QCDVzezNMz7Oafn+ziXf/4LIMTXv73f9rLP31oDxD+N81U97CbHlf8trvxJKsF
H/Pk9jBjgwTwVe5QpMHULVvzF8Bv296IPxDi2a4RuofdKT/iJ8zAnW7W1pVnZWEsnoYqK1rDaJxt
21presdmaKs1MvDw34vZjXmyd5yda2pSvtkopbh7TxsvXhhL68DpVF0IE7lufR2dTnfKg3vPDU5x
11ef5YGDnbx7Vyu/+uTNeXkeNVRasZaY+B+/PMv//I9z3La9kcc/eTN3XNnMxoZKKizmaFlmJoxP
hbem+RjCu4Tj/xsZuzAbJAMXuaC15kcvXuLrT3Wzd11t3H4PuXJNRx2V1hIePt7LpDeQMoA3222M
ePwLugx2DbnT2r23WMaLL948+Pj0LNP+YPSgCbutFLutJONacLcvQKfTnXL6xPCe3W0oBQ8fT938
s8flQanLUw7pun5DeC3khSTz4B5fgA9943mcUz4e/Mhevvi+XWlVumSDyaTY1mKnrNTMVz6wi3/+
8J5opms2Ka5qr15UBn7orJMtTZVpt4kwGq3FW7iOnkYvGbjItqFJL7/37Rf5zEMvc0WrnS+/f1de
799SYuKmzQ5+dWoQSL1L0Ki5dcZkO4FgiG6XO+4JQdmSbDNPbAWKoa22POM58Fd6J9CapBUosVqq
y7hxo4OHX+pNuVB3weWhtbos408oV7XVYCs1JV3I/O7zFxnx+Hnwd/fylisyr/BZqm/ds4+nPnMr
7961sI3xzjU1nB6YTNhWOJ4p7ywv9IzO6YOfSrPdxsxskEnvwqZjI3nogwISwFcVrTWPnujjLV96
iue6R/ird+7gB39wfcaNqbLBaG4FqT/ix6sFvzAyzWxQ5zQDN3pYxGto1Rc5wCF23jpcC55ZAI/u
wEwzAwc4sLeNS6MzHE1RLtfj8qS1hX4+S4mJvetqeT5BY6sZf5AHn+rmps0O9qytzfj2s6G+0pow
49/VXsNsUHNmYCrt23u600UgpLktg70QsUf+zZePToQgAXzVcLl9fOK7x7n/306woaGCX/zRTXz0
xvVz+hznkzFXapyykky87fRdRg+UNPpnLJYjSUMrI9OOHXt4N+Z0RrXgJy+Ns66+nNoMMrW3XtFM
hcWctCZcax05iT7zAA5w3fp6zg1NxW3b+r0j4ez7/ts2L+q2c80ox8xkGuXgGSfVZaXsWZv+G2ns
LuH5XG4/FrMJuy23m90lgK8CWmve9/XnOHTWyZ++bRs/+fgNOd0yn46GKis719Swrq48YTMgQ7wX
ilFCmKwCZakqrCVUWMwJA3iltWROFtheW4bHH8yoTvvEpfGMsm+AcksJb7uqhZ+/MpCwS+GIx89U
GusLiVy3vg6tF86De2eDfP2pbm7YWM++POwbWIyWahsNVda0A3gopHnynJM3bWnIqE1Bss08xmn0
S6mCSocE8FVg1OOne9jDp9+6hY+/aeOynjwe6x/eezVfTGP+vbqsFOu8o9U6nW7aasoWtUEnE46q
+LXgvWMztNWUzXmBtkcrUdKbRhma9DIw4U26gSeRu/e04fYFousI80V3uC5iCgXCWaylZOE8+L+9
8DrDUz7+aIVm3xCu1tnZXsOJNCtRTvaOM+LxZ7wTudEe/oQWrxZ8xJP7PiggAXxVMF7Mm5McGLsc
NjdVpdWfWykVrgWPWcTsdLpzOn1iaKiMvxsztgbc0J7hZh4jQ9yV5gJmrOvXh5tzJapG6RkO/58n
Ook+FVupmd1rauZs6PHOBvnab85z7fq6tDoHLqdda6rpHvak9Wno0FknJgVv2tKQ0X3YSs3UJjha
zZWHXZggAXxVMHalrV/kfOhK0GS3RTOdYEhzfji3JYSGhoQZ+HS09ttgVKSkWwt+snccs0lxRWvm
AdxkUrxndxtPdw7HXUTrGfFQalYZdWCc77oN9Zzun2TSGw6CPz7Wy9Ckb8XOfccyPtW80juR8rqH
zjrZu642br/5VBLtURjJQydCkHayq0KPy0NJGouFK1mz3Rbdcn5pdBp/IBS3iVW2OSqtPDevGmNi
ZpYpb2DBv2dNeSkVFnPateAnL02wrblq0RuR7t7Txj8d7uI9X312wVTS4KSXNXXlGc3pznf9+joe
0OEGT2/c1MDXDnexd10tN2xc2dk3wNVtkYXM3nHeuDnxlv6BiRlO9U/y2Tu2Lep+wp8M5wZwrTUu
ty96qlMuSQBfBXqGPaytX9qLebk1V9sYPBXeNNHpjPRAyccUSpWV8elZ/IFQtFdMX7QGfG75pVKK
ttr0+oKHQpqTveO8c2frose2oaGST92+hbODkwt+trmpMqOSuHh2r62l1Kw40j3K0KSP/gkvnz9w
dc4X5rKhuryUDY6K6Jt+Io+d6AfgrVcs7t+q2W7j1b65//4efxBfICQZuMiOCyOeRc+FrhRNdhv+
QIjx6dnoMWrpHIK7VMZmnhGPL3rMWLQPeJxPNO215WnVgveMeJjyBtiVYQXKfLlcTCyzmNnZXsMz
XS5+/soAO9fUcHOSbHal2bkmPHatddw3Ha01Dx3vZffamkVXZTXZbYx45h6tZvSQlzlwsWShkM64
J/RKFFuy1TXkptluy8nhy/PFqwWPnsQTJ4AbteCpGAuYi6lAyafrNtRxqn+S3rEZ7r9tU0Fk34ad
7dUMT/kSNpw61T/Ja0NuDmR42Ees5urw0WrOmOfH5W30UoUilmhg0osvEErrSK2VLPYIq3xVoED8
7fR9YzPYSk1xt0m315Yx6Q1EF/4SOXlpnHKLOS+fIpbiuvXh+e4r2+xpN3laKVJt6HnoeC8Ws4l3
XN2y6PuIt0fB2Lmby9PoDRLAi5xRTlboGXhTzAuly+nOW+BzRLKo2EqUeDXghrY028qe6J3gqrbq
FVOTn8g1HXW8YUM9f37njoLKvgF2tNopNStOXFpYiTIbDPHYiX7evKNxUdUnhnhtHkYkgIts6Ym0
FN2QYUvRlaaxKvxCOXZxjJnZYF4qUCD+FEq4Bjx+/xjj8mQB3BcIcqZ/Mq0a+OVWZjHzg3uv5w0F
UHkyn7XEzI4We9wM/Dfnhhnx+Ll79+KnTyCmzcNEbAAPP1fqctzICiSAF71ul4eyUjNN9vy1i80F
S4kJR6WFZzrDh/rmawrFVmrGbiuZ09AqXg24IZ1a8LMDU/iDyY9QE9mxc00Nr/RNEJzXufGh473U
V1h409bMNu/MV1teimXeLmGX24fdVpKXE64kgBe5C5EFzEL7+BtPU8wJKJvy2MvFEXM2pscXYGx6
NmFNvaPSgrXElLQWPNqBUAJ4zu1sr8HtC9A97I5eNj7t5+AZJ+/a1ZqyD08qSima7NY5C6Uujx9H
nvrrSwAvcsVQgWIwFowcldaMuvctVex2eiMwJ9rhmE4t+FOvuWi222iNfPwWuWO8ScbWg//7ywP4
g6ElVZ/EMg52MIy4fTjyUEIIEsCLmj8Q4tLYTNEEcKP/cj620MeK3U7fF20jm7iHenttecIMfMTt
48lz4eyvGD4VrXQbHBVUWUvmHLH28PFetjZVcUWrPSv3MX87fb620YME8KJ2aWyaYEgXTQA3MvB8
zX8bHDEZuDG3vSZJW4L2JBn4Yyf7CYR01rI/kZzJpLh6TTUnI5Uo54fdvPT6OAf2LjzJZ7GMw42N
PvCuSCvZfJAAXsSiJYSLbCm60kQD+DJk4FO+AN7ZIL3jM1jMpqQlYm01ZYx6/Ez7Fx619fDxPq5o
tbO1eWV1hixmO9trODMwiXc2yCPH+zApuGtXW9Zuv7nahnc2xORMgEAwxNj0bF5KCEECeFG7MLK0
lqIrjfFGdEVb5t37liJ2M7QPruMAABcmSURBVI9xEn2yk4zaE9SCvzY0xSt9E5J959nONTUEQppT
/RM88lIfN21uoNGevfWHpphdwqORE4xyfZSaQQJ4Eet2eagpL13SRoWVZN+6Wh7/5M15P4fR6Co3
7PbRF9nEk0z0YId58+APHe+lxKR4167FN7ASmTPq7b/+m276xme4e0/2sm+Ye+RfdBNPnhbZUwZw
pZRNKfWCUuqkUuqUUupvIpevV0odUUp1KaV+qJQqjihRRHqGi6cCBcIVHluW4VAKIwN3RTLwVG15
4x3sEAxpfvpSH7dsbcjbx2sR1mS30Wy38fjpIaqsJbz1iuas3r4xtTc0cTmAr6QM3Afs11rvBHYB
dyilrgf+HviS1noTMAZ8LHfDFItRTCWEy8kI4JfGZnC5fSkz8IZKKxazac4UyrNdLoYmfdwt0yfL
Ymfk1KM7r2pZdP/1RIyj1QYnvdFqJcdKWcTUYUYVfGnkjwb2Az+JXP4d4K6cjFAsyrQ/wOCkt2jm
v5eTsSX65UgpWntd8gBuMilaa2xzdmM+fLwXu60k43MXRXYY9eAH9mb/DdRaYqauwjIngOcrA0+r
H7hSygwcAzYBXwXOA+Naa2OZvReIO7GklLoXuBdg7dq1Sx2vSNMFVzh4rC/wHigrQanZRF2FJboZ
ZP5BDvG01ZZFa8GnvLP8x6lBDuxpx1qS3exPpOfD166jpdrGNR25WT8xjvyrLiul1Kyw2/Jz1EJa
i5ha66DWehfQDlwLpH3+kNb6Qa31Pq31voaGpfUdEOmLnkouGXhWOCotXBwJvymmczRde015dA78
l68O4p0N5ST7E+mpLi/lPbvbc7Z5qjmynX4kcphxvjZpZVSForUeBw4DbwBqlFLG20w7EP94bLEs
jC6EHY7U2aJIzZgHLzGpaNlYMm21ZQxP+fDOBnnoWC/rHRXslt4nRau5Orwb0+X246jKXz1HOlUo
DUqpmsjXZcDtwBnCgfy9kavdAzyaq0GKzHW7PDTbbZRb5NS8bDBKCVtqbGn18Day9Bd6RjnSM8qB
Pdnb+SdWnia7DZfbz8CENy9HqRnSycBbgMNKqZeBF4EntNY/Az4LfEop1QXUA9/M3TBFpqQCJbuM
0r9UFSgG43r/dKgLgLt2Z7f2WKwsRilhl3Mqb9voIY1FTK31y8DuOJd3E54PFytQj8vDnVct/qgo
MZcxhZKsiVWs9rrw9V64MMobNtSn/XuiMBmN1maDOq91/rITswiNefyMT89KCWEWGQE83Qy8qcoa
nWrJ9s4/sfI0x6yL5KsGHCSAF6WeSA+UjnoJ4NliZFXpVKAAlJhNtFTbKCs18zb5JFT0YgN4PufA
ZYWrCBVbF8KV4IpWOzvX1ERPaU/H269uwVpiptIqL7NiVxM5Ws0fCK2sOXBReHpcHswmxRqZd82a
+korj953Y0a/87m3bc/RaMRKo5Si2W7j9dFpmQMXS9Pj8rCmtiwvh6oKIcJij/zLF3mFFyEpIRQi
/4xKlLo8ntcqAbzIaK3pcXnokAAuRF5d2WpnQ0NFXj/5yhx4kRma9DEzG5QSQiHy7N6bN/D7N23I
631KAC8y3ZEeKNKFUIj8UkphznO3BJlCKTLRLoRSQihE0ZMAXmR6hj1YS0y0ZPHQViHEyiQBvMhc
GPHQUV+R9NR0IURxkABeZLqlhFCIVUMCeBEJBEO8PjIt899CrBJShVKAnjs/wu99+0Vmg6E5l2sg
GNKSgQuxSkgAL0DHXx9jZjbIJ27ZyPypbmuJmTuubF6egQkh8koCeAEamJihpryUz96R9tnSQogi
JHPgBWhg3EtLdXp9qYUQxUsCeAHqn/DSWi113kKsdhLAC9DAxAwtNRLAhVjtJIAXmBl/kPHpWZlC
EUJIAC80/RMzALRKBi7EqicBPI9+dPQSl0anl3QbA+NeAMnAhRASwPNleMrHZ37yMj86emlJtxPN
wCWAC7HqSQDPk9MDk0A4kC+FkYE3Vefv3D0hxMokATxPzmQrgE/M4Ki0Yi0xZ2NYQogCJgE8T6IB
3L20AN4/4ZUFTCEEIAE8b4wA7lpiBt4/PiPz30IIQAJ4Xnhng5wf9mA2KYbdPrTWi7odrTUD47KJ
RwgRJgE8DzqH3ARDmt1rapgNaiZmZhd1O5PeAB5/UDJwIQQgATwvjOmTmzY3AOBa5Dz4QKSEUDJw
IQRIAM+L0wOTlFvM7OuoBcC5yHlw2cQjhIglATwPTg9MsrW5iiZ7uHZ7saWEso1eCBErZQBXSq1R
Sh1WSp1WSp1SSt0fubxOKfWEUqoz8ndt7odbeLTWnBmYZHuLnYbKcOB1uf2Luq2BcS9mk6KxSgK4
ECK9DDwA/InWegdwPXCfUmoH8KfAQa31ZuBg5HsxT9/4DFPeANtb7NjLSrCYTUvKwJuqrJjnn6Mm
hFiVUgZwrfWA1vp45Osp4AzQBrwb+E7kat8B7srVIAvZmYEpAHa02FFK4ai0LDqAD4x7aamR+W8h
RFhGc+BKqQ5gN3AEaNJaD0R+NAg0Jfide5VSR5VSR4eHh5cw1MJ0un8SpWBbcxUADVXWJVWhtMhJ
PEKIiLQDuFKqEngI+GOt9WTsz3R4Z0rc3Sla6we11vu01vsaGhqWNNhCdGZgknV15VRYw+dHOyqt
i8rAtdYMTHhplQxcCBGRVgBXSpUSDt7f01o/HLl4SCnVEvl5C+DMzRAL25nB8AKmYbEZ+KjHjy8Q
kgxcCBGVThWKAr4JnNFafzHmR48B90S+vgd4NPvDK2xuX4CLI9PsiAngjkorIx4/wVBm2+kHJsI1
4JKBCyEM6WTgNwIfAfYrpU5E/twJfAG4XSnVCbw58r2IcTayA3N+Bh4MacamMysl7BuXgxyEEHOV
pLqC1voZIFHd2m3ZHU5xMbbQb2+dG8AhvJ3eUZn+oQwD47KNXggxl+zEzKHTA1PYbSW0xsxbG0E7
04XMgQkvlhIT9RWWrI5RCFG4JIAv0uCEl+8+fzFpa9gzA5PsaA3XfxuMDDzTAN4/4aWl2jbntoQQ
q5sE8EV64FAnf/HTV3nyXPza9mBIc3ZeBQrMnULJxMC41IALIeaSAL4I3tkgPzvZD8BXDnbGzcIv
jHjwzoYWBPAKixlbaebb6QcmvLKAKYSYQwL4Ivz6zBCT3gBvv7qFE5fGebrTteA6xgLmjnkBXClF
Q1Vmm3mCIc3gpFcWMIUQc0gAX4SHj/fRUm3jf/3OTlqrbXGz8NP9k5SYFJubKhf8vqPSmlFHwuEp
H8GQlj7gQog5JIBnaHjKx29eG+au3W3YSs184paNHLs4xnPnR+Zc78zAJBsbKrGWmBfcRkOG2+ml
D7gQIh4J4Bl69EQfwZDmwJ42AH5n3xqa7Fa+crBzzvXODEyxvaUq7m00VFkZzmARU07iEULEIwE8
Qw8d72NnezWbGsPB2VZq5uNv2siRnlGe7w5n4aMeP4OTXna02uPehqPSyti0n9lgKK37NM7ClEVM
IUQsCeAZON0/yZmBSQ7sbZ9z+QevXYuj0so/Hgpn4WfibKGP1VBlRetwoE9H/7iXCosZe1nKjbNC
iFVEAngGHj7eS6lZ8c6rW+dcHs7CN/Bs1wjHLo6mFcAh/c08/eMztNSUySYeIcQcEsDTFAiG+OmJ
fvZva6Q2znb2D123lvoKCw8c7OL0wCQNVdaEvU6i2+nTnAeXgxyEEPFIAE/T050uXG4fd+9pj/vz
cksJf3DzBn7z2jAHzzgX1H/Hasw0A5dNPEKIOCSAp+mh473Ulpdy69bGhNf5yPXrqC0vZWJmNuH0
CVzOwNPZTu8PhHC5fbKJRwixgATwNEzMzPL46SHetbMVS0nif7IKawm/f9MGgIQlhABlFjOV1pK0
MvChSS9aSwWKEGIhKWtIwy9eGcAfCC2oPonnozd2AHD7jrhnPEelu52+X/qACyESkACehoeO9bKp
sZKr2qpTXrfcUsJ9t25KeT1HpSWtKRTjKDXZxCOEmE+mUFK4OOLh6MUxDuxpz2oZX9oZuGyjF0Ik
IAE8hYeP96EUvGd3W1ZvN91+KAPjXqrLSim3yIclIcRcEsBTePHCKFe3VdOc5TpsR6WVSW8AXyCY
9HpSAy6ESEQCeAqdTjdbmhJXlCzW5ZN5km+n7x/30lYj899CiIUkgCcxPu1neMoXt6f3UqV7uPHA
xIxUoAgh4pIAnkSX0w3A5sYcZuBJAviMP8jY9KxUoAgh4pIAnkRnJIBvasx+Bh5taJWklFAqUIQQ
yUgAT6JzyE1ZqTknc9D1leGGWMkycDnIQQiRjATwJDqdU2xqrMRkyn4bV2uJmeqy0vQycAngQog4
JIAn0eV0szkH0yeGVJt5jAy8qTp+W1ohxOomATyBKe8sAxNeNuWgAsWQajv9+WE3TXZr3IORhRBC
AngCuaxAMTRU2RJm4MGQ5qnOYW7c5MjZ/QshCpsE8AQ6owE8txl4ogD+0utjjE/Pctu25F0NhRCr
lwTwBLqcbiwlJtbUlefsPhqqrHj8Qab9gQU/O3jWSYlJcdMWycCFEPFJAE+gc2iKjQ2VmHNQgWJo
ME7mmVq4nf7QGSfXdNRht5Xm7P6FEIVNAngCnTmuQAFwRDfzeOdc3js2zbmhKW7bnvj4NiGESBnA
lVLfUko5lVKvxlxWp5R6QinVGfm7NrfDzK9pf4DesZmcB/CGaD+UuRn44bNOAG7dJgFcCJFYOhn4
t4E75l32p8BBrfVm4GDk+6Jx3ukByEkTq1iNCbbTHzrrpKO+nA2OipzevxCisKUM4Frrp4DReRe/
G/hO5OvvAHdleVzLqtM5BcCmHJYQAtRVWFBqbkfCaX+AZ8+PsH9bU1ZPABJCFJ/FzoE3aa0HIl8P
Aglr3ZRS9yqljiqljg4PDy/y7vKr0+mm1KxYV5+7ChSAErOJuvK5m3l+2zWCPxBiv0yfCCFSWPIi
ptZaAzrJzx/UWu/TWu9raGhY6t3lReeQm/WOCkrNuV/jdcw7Wu3QOScVFjPXrq/L+X0LIQrbYiPU
kFKqBSDytzN7Q1p+Xc6pnO7AjNVQZY1m4FprDp91cvOWBiwlUiAkhEhusVHiMeCeyNf3AI9mZzjL
zzsb5PXR6Zz0AI8ntqHVmYEpBia8Un0ihEhLOmWEPwCeA7YqpXqVUh8DvgDcrpTqBN4c+b4odA97
COncV6AYjO30WmsOnR0C4NatEsCFEKmVpLqC1vqDCX50W5bHsiIYFSj5nELxBUK4fQEOnnWyc01N
9LQeIYRIRiZa5+lyujGbFB2O3FagGIxgfW5wihOXxtkv2bcQIk0SwOfpHHKzrr48bz24jdPpf3y0
F62R7fNCiLRJAJ+n0zmV8y30sYwM/Gcv99NYZeWKVnve7lsIUdgkgMfwB0JcGJnO2/w3XM7APf4g
+7c1yu5LIUTaJIDHuDDiIRjSeatAAagtt0Rb1sruSyFEJiSAx+gcCp/Ck68acACzSVFfYcFiNsnx
aUKIjKQsI1xNOp1TKAUbG/IXwAE6HBXYbaVUWOW/QwiRPokYMTqdbtbWlWMrze8p8N/4yD7MZpn7
FkJkRgJ4jK6h3J/CE091uRybJoTInMyBRwSCIbpd7pz3ABdCiGyRAB5xcXSa2aBelgxcCCEWQwJ4
hFGBks8SQiGEWAoJ4BFdkSZW+a5AEUKIxZIAHtHpdNNWUyalfEKIgiEBPOLc4JRMnwghCooEcKDH
5eHs4BTXra9f7qEIIUTaJIADjxzvxaTgPbvblnsoQgiRtlUfwEMhzUPH+7hxk4PmattyD0cIIdK2
6gP4CxdG6Ruf4cCe9uUeihBCZGTVB/CHjvVSYTHz1iual3soQgiRkVUdwGf8QX7xygB3XtVCmSW/
DayEEGKpVnUA/9WpQTz+IAf2yvSJEKLwrOoA/tDxXtpqyri2o265hyKEEBlbtQF8cMLLs10uDuxp
w2SSXtxCiMKzagP4T0/0EdLwHqk+EUIUqFUZwLXWPHSsl73ralnvqFju4QghxKKsygD+at8knU43
d++RnZdCiMK1KgP4Q8d7sZSYeMdVrcs9FCGEWLRVF8D9gRCPnezn9u1NchalEKKgrboA/uQ5J6Me
Pwf2yvSJEKKwFe3pBf5AiGBIL7j8J8d6cVRauGlzwzKMSgghsqfoArh3NsgXn3iNbz7TEzeAA3zs
jespNa+6Dx9CiCJTVAH85KVx/uTHJ+lyujmwpz3uCTslJsXdUvsthCgCSwrgSqk7gK8AZuBftNZf
yMqoMuQPhHjgYCdf+815Gqus/P+/dy03b5EpEiFEcVt0AFdKmYGvArcDvcCLSqnHtNanszW4dJzu
n+RPfnySMwOTHNjTzl++cwfVZVJdIoQofkvJwK8FurTW3QBKqX8D3g1kPYD/+SOv8ELPaNyfXRjx
UF1m4Ru/u4/bdzRl+66FEGLFWkoAbwMuxXzfC1w3/0pKqXuBewHWrl27qDtqrSlLeGL8jZsc3H/b
ZmorLIu6bSGEKFQ5X8TUWj8IPAiwb9+++GUhKdx366asjkkIIYrBUmrp+oA1Md+3Ry4TQgiRB0sJ
4C8Cm5VS65VSFuADwGPZGZYQQohUFj2ForUOKKX+EPgV4TLCb2mtT2VtZEIIIZJa0hy41voXwC+y
NBYhhBAZkP3kQghRoCSACyFEgZIALoQQBUoCuBBCFCil9aL21izuzpQaBi4u8tcdgCuLwykU8rhX
l9X6uGH1PvZ0Hvc6rfWCDn15DeBLoZQ6qrXet9zjyDd53KvLan3csHof+1Iet0yhCCFEgZIALoQQ
BaqQAviDyz2AZSKPe3VZrY8bVu9jX/TjLpg5cCGEEHMVUgYuhBAihgRwIYQoUAURwJVSdyilziml
upRSf7rc48kVpdS3lFJOpdSrMZfVKaWeUEp1Rv6uXc4x5oJSao1S6rBS6rRS6pRS6v7I5UX92JVS
NqXUC0qpk5HH/TeRy9crpY5Enu8/jLRrLjpKKbNS6iWl1M8i3xf941ZKXVBKvaKUOqGUOhq5bNHP
8xUfwGMOT34bsAP4oFJqx/KOKme+Ddwx77I/BQ5qrTcDByPfF5sA8Cda6x3A9cB9kf/jYn/sPmC/
1nonsAu4Qyl1PfD3wJe01puAMeBjyzjGXLofOBPz/Wp53LdqrXfF1H4v+nm+4gM4MYcna639gHF4
ctHRWj8FzD+9+d3AdyJffwe4K6+DygOt9YDW+njk6ynCL+o2ivyx6zB35NvSyB8N7Ad+Erm86B43
gFKqHXg78C+R7xWr4HEnsOjneSEE8HiHJ7ct01iWQ5PWeiDy9SDQtJyDyTWlVAewGzjCKnjskWmE
E4ATeAI4D4xrrQORqxTr8/3LwGeAUOT7elbH49bA40qpY5ED32EJz/OcH2osskdrrZVSRVv3qZSq
BB4C/lhrPRlOysKK9bFrrYPALqVUDfAIsG2Zh5RzSql3AE6t9TGl1C3LPZ48e6PWuk8p1Qg8oZQ6
G/vDTJ/nhZCBr/bDk4eUUi0Akb+dyzyenFBKlRIO3t/TWj8cuXhVPHYArfU4cBh4A1CjlDKSq2J8
vt8IvEspdYHwlOh+4CsU/+NGa90X+dtJ+A37WpbwPC+EAL7aD09+DLgn8vU9wKPLOJaciMx/fhM4
o7X+YsyPivqxK6UaIpk3Sqky4HbC8/+HgfdGrlZ0j1tr/TmtdbvWuoPw6/mQ1vrDFPnjVkpVKKWq
jK+BtwCvsoTneUHsxFRK3Ul4zsw4PPnvlnlIOaGU+gFwC+H2kkPAXwE/BX4ErCXcivd9Wuv5C50F
TSn1RuBp4BUuz4n+GeF58KJ97EqpqwkvWpkJJ1M/0lr/rVJqA+HMtA54CfhPWmvf8o00dyJTKJ/W
Wr+j2B935PE9Evm2BPi+1vrvlFL1LPJ5XhABXAghxEKFMIUihBAiDgngQghRoCSACyFEgZIALoQQ
BUoCuBBCFCgJ4EIIUaAkgAshRIH6v3QOWipDHfaBAAAAAElFTkSuQmCC
">

```

```
