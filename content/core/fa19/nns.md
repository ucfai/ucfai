---
title: "Getting Started with Neural Networks"
linktitle: "Getting Started with Neural Networks"

date: "2019-10-02T17:30:00"
lastmod: "2019-10-02T17:30:00"

draft: false
toc: true
type: docs

weight: 4

menu:
  core_fa19:
    parent: Fall 2019
    weight: 4

authors: ["jarviseq", ]

urls:
  youtube: ""
  slides:  ""
  github:  "https://github.com/ucfai/core/blob/master/fa19/2019-10-02-nns/2019-10-02-nns.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-fa19-nns"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/fa19/2019-10-02-nns/2019-10-02-nns.ipynb"

location: "MSB 359"
cover: "https://cdn-images-1.medium.com/max/1200/1*4V4OU2GEzmOWHgCJ8varUQ.jpeg"

categories: ["fa19"]
tags: ["neural-nets", ]
abstract: >-
  You've heard about them: Beating humans at all types of games, driving cars, and recommending your next Netflix series to watch, but what ARE neural networks? In this lecture, you'll actually learn step by step how neural networks function and learn. Then, you'll deploy one yourself!
---
```python
# This is a bit of code to make things work on Kaggle
import os
from pathlib import Path

if os.path.exists("/kaggle/input"):
    DATA_DIR = Path("/kaggle/input")
else:
    raise ValueError("We don't know this machine.")
```

## Before we get started

We need to import some non-sense

Pytorch is imported as just torch, otherwise we've seen everything else before.

```python
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import time
```

## Tensors

Tensors live at the heart of Pytorch.

You can think of tensors as an Nth-dimensional data container similar to the containers that exist in numpy. 

Below we have some *magical* tensor stuff 
going on to show you how to make some tensors using the built-in tensor generating functions. 

```python
# create a tensor
new_tensor = torch.Tensor([[1, 2], [3, 4]])

# create a 2 x 3 tensor with random values
empty_tensor = torch.Tensor(2, 3)

# create a 2 x 3 tensor with random values between -1and 1
uniform_tensor = torch.Tensor(2, 3).uniform_(-1, 1)

# create a 2 x 3 tensor with random values from a uniform distribution on the interval [0, 1)
rand_tensor = torch.rand(2, 3)

# create a 2 x 3 tensor of zeros
zero_tensor = torch.zeros(2, 3)
```

To see what's inside of the tensor, put the name of the tensor into a code block and run it. 

These notebook environments are meant to be easy for you to debug your code, 
so this will not work if you are writing a python script and running it in a command line.

```python
new_tensor
```

You can replace elements in tensors with indexing. 
It works a lot like arrays you will see in many programming languages. 


```python
new_tensor[0][0] = 5
new_tensor
```

How the tensor is put together is going to be important, so there are some 
built-in commands in torch that allow you to find out some information about the tensor you are working with.

```python
# type of a tensor
print(new_tensor.type())  

# shape of a tensor
print(new_tensor.shape)    
print(new_tensor.size())   

# dimension of a tensor
print(new_tensor.dim())
```

## Coming from Numpy

Much of your data manipulation will be done in either pandas or numpy. 
To feed that manipulated data into a tensor for use in torch, you will have to use the `.from_numpy` command.

```python
np_ndarray = np.random.randn(2,2)
np_ndarray
```

```python
# NumPy ndarray to PyTorch tensor
to_tensor = torch.from_numpy(np_ndarray)

to_tensor
```

## Checking for CUDA

CUDA will speed up the training of your Neural Network greatly.

Your notebook should already have CUDA enabled, but the following command can be used to check for it.

TL:DR: CUDA rock for NNs

```python
torch.cuda.is_available()
```

## Defining Networks

In the example below, we are going to make a simple example to show how you will go about 
building a Neural Network using a randomly generated dataset. This will be a simple network with one hidden layer.

First, we need to set some placeholder variables to define how we want the network to be set up.

```python
n_in, n_h, n_out, batch_size = 10, 5, 1, 10
```

Next, we are going to generate our lovely randomised dataset.

We are not expecting any insights to come from this network as the data is generated randomly. 

```python
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
```

Next, we are going to define what our model looks like. The `Linear()` part applies a linear transformation to the 
incoming data, with `Sigmoid()` being the activation function that we use for that layer. 

So, for this network, we have two fully connected layers with a sigmoid as the activation function. 
This looks a lot like the network we saw in the slide deck with one input layer, one hidden layer, and one output layer. 

```python
model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.Sigmoid(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
```

Next, let's define what the loss fucntion will be.

For this example, we are going to use Mean Squared Error, but there are a ton of different loss functions we can use.

```python
criterion = nn.MSELoss()
```

Optimizer is how the network will be training. 

We are going to be using a standard gradient descent method in this example.

We will have a learning rate of 0.01, which is pretty standard too.
 You are going to want to keep this learning rate pretty low, as high learning rates cause problems in training.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Now, let's train!

To train, we combine all the different parts that we defined into one for loop.

```python
for epoch in range(50):
    # Forward Propagation
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()
```

In this example, we printed out the loss each time we completed an epoch.

This ran very quickly, but with more complex models, those outputs are going to be very 
important for checking on how your network is doing during the training process which could take hours if not days!

More likely than not, you're going to see that this network is not converging, which is to be expected with random data. 

In our next example, we're going to be building a proper model with an awesome dataset. 

## Diabetes dataset

What causes someone to have diabetes? 

With this dataset, we are going to see if some basic medical data about a person can help us 
predict if someone is diabetic or not using *magical* neural networks. 

First though, let's get that dataset and see what's inside.

```python
dataset = pd.read_csv(DATA_DIR / "train.csv", header=None).values

pd.DataFrame(dataset).head()
```

## What are we looking at?

This is a fairly small dataset that includes some basic information about an individual's health. 

Using this information, we should be able to make a model that will allow us to determine if a person has diabetes or not. 

The last column, `Outcome`, is a single digit that tells us if an individual has diabetes. 

We need to clean up the data a bit, so let's get rid of the first row with the labels on them.

```python
dataset = np.delete(dataset, 0, 0)

pd.DataFrame(dataset).head()
```

Alright, now let's break up our data into test and train set.

Once we have those sets, we'll need to set them to be tensors. 

This bit of code below does just that!

```python
# split into x and y sets

X = dataset[:,:-1].astype(np.float32)

Y = dataset[:,-1].astype(np.float32)

# Needed to make PyTorch happy
Y = np.expand_dims(Y, axis = 1)

# Test-Train split
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1)

# Here we're defining what component we'll use to train this model
# We want to use the GPU if available, if not we use the CPU
# If your device is not cuda, check the GPU option in the Kaggle Kernel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
```

### PyTorch Dataset
Our next step is to create PyTorch **Datasets** for our training and validation sets. 
**torch.utils.data.Dataset** is an abstract class that represents a dataset and 
has several handy attributes we'll utilize from here on out. 

To create one, we simply need to create a class which inherits from PyTorch's Dataset class and 
override the constructor, as well as the __len__() and __getitem__() methods.

```python
class PyTorch_Dataset(Dataset):
  
  def __init__(self, data, outputs):
        self.data = data
        self.outputs = outputs

  def __len__(self):
        'Returns the total number of samples in this dataset'
        return len(self.data)

  def __getitem__(self, index):
        'Returns a row of data and its output'
      
        x = self.data[index]
        y = self.outputs[index]

        return x, y
```

With the class written, we can now create our training and validation 
datasets by passing the corresponding data to our class

```python
train_dataset = PyTorch_Dataset(xTrain, yTrain)
val_dataset = PyTorch_Dataset(xTest, yTest)

datasets = {'Train': train_dataset, 'Validation': val_dataset}
```

### PyTorch Dataloaders

It's quite inefficient to load an entire dataset onto your RAM at once, so PyTorch uses **DataLoaders** to 
load up batches of data on the fly. We pass a batch size of 16,
 so in each iteration the loaders will load 16 rows of data and return them to us.

For the most part, Neural Networks are trained on **batches** of data so these DataLoaders greatly simplify 
the process of loading and feeding data to our network. The rank 2 tensor returned by the dataloader is of size (16, 8).

```python
dataloaders = {x: DataLoader(datasets[x], batch_size=16, shuffle=True, num_workers = 4)
              for x in ['Train', 'Validation']}
```

### PyTorch Model

We need to define how we want the neural network to be structured, 
so let's set those hyper-parameters and create our model.

```python
inputSize =  8         # how many classes of input
hiddenSize = 15        # Number of units in the middle
numClasses = 1         # Only has two classes
numEpochs = 20         # How many training cycles
learningRate = .01     # Learning rate

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
```

### PyTorch Training

Now we create an instance of this NeuralNet() class and define the loss function and optimizer 
we'll use to train our model. 
In our case we'll use Binary Cross Entropy Loss, a commonly used loss function binary classification problems.

For the optimizer we'll use Adam, an easy to apply but powerful optimizer which is an extension of the popular 
Stochastic Gradient Descent method. We need to pass it all of the parameters it'll train, 
which PyTorch makes easy with model.parameters(), and also the learning rate we'll use.

```python
# Creating our model
model = NeuralNet(inputSize, hiddenSize, numClasses)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = learningRate)
model.to(device)
print(model)
```

At this point we're finally ready to train our model! In PyTorch we have to write our own training loops 
before getting to actually train the model. This can seem daunting at first, so let's break up each stage of the training process. 

The bulk of the function is handled by a nested for loop, the outer looping through each epoch and the 
inner looping through all of the batches of images in our dataset. 
Each epoch has a training and validation phase, where batches are served from their respective loaders. 
Both phases begin by feeding a batch of inputs into the model, which implicity calls the forward() function on the input. 
Then we calculate the loss of the outputs against the true labels of the batch. 

If we're in training mode, here is where we perform back-propagation and adjust our weights. To do this, 
we first zero the gradients, then perform backpropagation by calling .backward() on the loss variable. 
Finally, we call optimizer.step() to adjust the weights of the model in accordance with the calculated gradients.

The remaining portion of one epoch is the same for both training and validation phases, 
and simply involves calculating and tracking the accuracy achieved in both phases. 
A nifty addition to this training loop is that it tracks the highest validation accuracy 
and only saves weights which beat that accuracy, ensuring that the best performing weights are returned from the function.

```python
def run_epoch(model, dataloaders, device, phase):
  
  running_loss = 0.0
  running_corrects = 0
    
  if phase == 'Train':
    model.train()
  else:
    model.eval()
  
  # Looping through batches
  for i, (inputs, labels) in enumerate(dataloaders[phase]):
    
    # ensures we're doing this calculation on our GPU if possible
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Zero parameter gradients
    optimizer.zero_grad()
    
    # Calculate gradients only if we're in the training phase
    with torch.set_grad_enabled(phase == 'Train'):
      
      # This calls the forward() function on a batch of inputs
      outputs = model(inputs)

      # Calculate the loss of the batch
      loss = criterion(outputs, labels)

      # Adjust weights through backpropagation if we're in training phase
      if phase == 'Train':
        loss.backward()
        optimizer.step()
      
    # Get binary predictions
    preds = torch.round(outputs)

    # Document statistics for the batch
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels)
    
  # Calculate epoch statistics
  epoch_loss = running_loss / datasets[phase].__len__()
  epoch_acc = running_corrects.double() / datasets[phase].__len__()
  
  return epoch_loss, epoch_acc
```

```python
def train(model, criterion, optimizer, num_epochs, dataloaders, device):
    start = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t|')
    print('-' * 73)
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        
        # Training phase
        train_loss, train_acc = run_epoch(model, dataloaders, device, 'Train')
        
        # Validation phase
        val_loss, val_acc = run_epoch(model, dataloaders, device, 'Validation')
           
        # Print statistics after the validation phase
        print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t|".format(epoch + 1, train_loss, train_acc, val_loss, val_acc))

        # Copy and save the model's weights if it has the best accuracy thus far
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    total_time = time.time() - start
    
    print('-' * 74)
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))

    # load best model weights and return them
    model.load_state_dict(best_model_wts)
    return model
```

Now, let's train the model!

```python
model = train(model, criterion, optimizer, numEpochs, dataloaders, device)
```

```python
# Function which generates predictions, given a set of inputs
def test(model, inputs, device):
  model.eval()
  inputs = torch.tensor(inputs).to(device)
  
  outputs = model(inputs).cpu().detach().numpy()
  
  preds = np.where(outputs > 0.5, 1, 0)
  
  return preds
```

```python
preds = test(model, xTest, device)
```

Now that our model has made some predictions, let's find the mathew's 

```python
# import functions for matthews and confusion matrix
from sklearn.metrics import confusion_matrix, matthews_corrcoef

matthews_corrcoef(preds, yTest)
```

Let's check the confusion matrix

```python
confusion_matrix(preds, yTest)
```

Ehhhhhh, that's not bad...

There's probably a bunch of things we could do to improve accuracy.

Why don't we have you give it a shot!

## Make this model better!

There is no right or wrong way to optimise this model.

Use your understanding of Neural Networks as a launching point.

You can use the previously 

There are many aspects to this model that can be changed to increase accuracy, like:
* Make the NN deeper (increase number of layers)
* Change the learning rate
* Add more hidden units
* train for more epochs
* and a whole bunch more!

[Just Do It](https://youtu.be/ZXsQAXx_ao0?t=2)

```python
#TODO, make a better model!

### BEGIN SOLUTION

inputSize =  8         # how many classes of input
hiddenSize = 15        # Number of units in the middle
numClasses = 1         # Only has two classes
numEpochs = 69         # How many training cycles
learningRate = .01     # Learning rate

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

model = train(model, criterion, optimizer, numEpochs, dataloaders, device)

predictions = test(model, xTest, device)

### END SOLUTION
```

```python
# Run this to generate the submission file for the competition!
### Make sure to name your model variable "model" ###

# load in test data:
test_data = pd.read_csv(DATA_DIR / "test.csv", header=None).values
# remove row with column labels:
test_data = np.delete(test_data, 0, 0)

# convert to float32 values
X = test_data.astype(np.float32)
# get indicies for each entry in test data
indicies = [i for i in range(len(X))]

# generate predictions
preds = test(model, X, device)

# create our pandas dataframe for our submission file. Squeeze removes dimensions of 1 in a numpy matrix Ex: (161, 1) -> (161,)
preds = pd.DataFrame({'Id': indicies, 'Class': np.squeeze(preds)})

# save submission csv
preds.to_csv('submission.csv', header=['Id', 'Class'], index=False)
```
