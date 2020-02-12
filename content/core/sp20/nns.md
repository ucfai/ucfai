---
title: "Introduction to Neural Networks"
linktitle: "Introduction to Neural Networks"

date: "2020-02-05T17:30:00"
lastmod: "2020-02-05T17:30:00"

draft: false
toc: true
type: docs

weight: 3

menu:
  core_sp20:
    parent: Spring 2020
    weight: 3

authors: ["jarviseq", "dillonnotdylan", ]

urls:
  youtube: "https://youtu.be/-3-nasQMyEk"
  slides:  "https://docs.google.com/presentation/d/17Kw1gwzo5YmXbHdHac-j8TnpIt4Re76u7uzjhfEEmQA"
  github:  "https://github.com/ucfai/core/blob/master/sp20/02-05-nns/02-05-nns.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-sp20-nns"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/sp20/02-05-nns/02-05-nns.ipynb"

location: "HPA1 112"
cover: "https://cdn-images-1.medium.com/max/1200/1*4V4OU2GEzmOWHgCJ8varUQ.jpeg"

categories: ["sp20"]
tags: ["neural-networks", "gradient-decent", "back-propagation", "nns", ]
abstract: >-
  You've heard about them: Beating humans at all types of games, driving cars, and recommending your next Netflix series to watch, but what ARE neural networks? In this lecture, you'll actually learn step by step how neural networks function and learn. Then, you'll deploy one yourself!
---
```python
# change this if running locally
DATA_DIR = "/kaggle/input/ucfai-core-sp20-nns"
# DATA_DIR = "."
```

## Before we get started

We need to import our packages will be using. Here we are going to see something new, Pytorch. Pytorch is a deep learning library used videly, developed by Facebook. We will be using Pytorch for the rest of the semester for our deep learning models. It has a variety of useful tools such as built in dataloaders, easily create dataset classes to handle our data, and more.

We will also be using numpy and pandas to handle our data loading.

```python
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import time

!pip install torchsummary
from torchsummary import summary
```

## Tensors

Tensors live at the heart of Pytorch.

You can think of tensors as an Nth-dimensional data container similar to the containers that exist in numpy. They have no notion of deep learning, gradients, or anything like that, they just N dimensional data to be used in computation.

The special thing about Pytorch tensors compared to numpy arrays is that they can run on a GPU as well as a CPU. So that means our data operations can be sped up by a huge margin by doing calculations on a GPU (or multiple). Lets check for a GPU now.

## Checking for CUDA

CUDA will speed up the training of your Neural Network greatly. It does so by parallelizing computations across all "cuda cores" on your GPU. Cuda cores are parallel processes, like on your CPU, but instead of 4 or 8 theres thousands of them.

Your notebook should already have CUDA enabled, but the following command can be used to check for it.

```python
torch.cuda.is_available()
```

Here we're defining the device we want to use to put our tensors and models on in this notebook. We can set the device to "cuda:0", which is our GPU, or "cpu" if cuda is not available.

If cuda is not available, check the GPU option in the Kaggle Kernel to enable a GPU for your notebook.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
```

Now that we have our device and know what tensors are, lets take a look at some basic tensors.

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

You can also put the tensor onto your device using the `.to(device)`. You can also use `.cuda()` which will try to put the tensor onto your GPU, but if you don't have one this will throw an error, thats why we define the device dynamically above.

Remember, the `.to(device)` method returns a **NEW** tensor that is on the gpu, it does not update in place.

```python
new_tensor = new_tensor.to(device)
new_tensor
```

How the tensor is shaped will be important, as we need to keep them in mind when building our models, so there are some 
built-in commands in torch that allow you to find out some information about the tensor you are working with.

```python
# data type of a tensor, notice that running on a GPU will give a type of cuda.<datatype>Tensor, 
# in this case a torch.cuda.FloatTensor
print(new_tensor.type())  

# shape of a tensor, both give the same thing
print(new_tensor.shape)    
print(new_tensor.size())   

# dimension of a tensor, how many dimensions is has (2D, 3D, etc.)
print(new_tensor.dim())
```

## Coming from Numpy

Much of your data manipulation will be done in either pandas or numpy. 
To feed that manipulated data into a tensor for use in torch, you will have to use the `.from_numpy` command. [Doc link](https://pytorch.org/docs/stable/torch.html#torch.from_numpy) 

**This tensor will share the same memory as the numpy array it came from, so edits to one will change the other.**

```python
np_ndarray = np.random.randn(2,2)
np_ndarray
```

```python
# NumPy ndarray to PyTorch tensor
to_tensor = torch.from_numpy(np_ndarray)

to_tensor
```

## Defining Networks

In the example below, we are going to make a model to show how you will go about 
building a Neural Network using a randomly generated dataset. This will be a simple network with one hidden layer.

First, we need to set some placeholder variables to define how we want the network to be set up.

```python
n_in, n_h, n_out, batch_size = 10, 5, 1, 10
```

### n_in? batch_size? What are these?
Lets deconstruct what these mean. `n` stands for nodes, so those variables define how many nodes we have on each of our layers, the input, hidden, and output, like we saw in the slides.

#### Batch size
This is something extremely important when going to train a model. Batch size defines how much data we feed into the network before backpropagating and updating the weights of our model. In the slides, we showed backpropagation after a single piece of data was feed into the network, while in reality our models will get fed *batches* of data, such as 32, 64, 128, etc. before we update our weights.

While each piece of data is fed into the network, our loss is calculated. The loss is accumlated (summed) for all pieces of data in the batch, then that total sum of the losses is used to find the gradients and update our weights.

**Now, why is this important?**
Think of this scenerio. Imagine you are lost in a forest, and you have a compass to guide you. You know you need to go North to leave the forest. You can't look at the compass and walk at the same time, so you have two options. You can take a step, then look at the compass, adjust your course, then take another step. Or you look at the compass, and walk for a few minutes, then check again. Now, imagine your path through the forest. In the first scenerio your path is very jagged and it takes you much longer to get out. The second scenerio your path through the forest is much more a smooth curve, and you leave the forest quickly since you spend less time checking the compass and readjusting.

This is analogous to backpropagating on every piece of data versus a batch of data. Your model's loss over time is much smoother and it normally will train faster. 

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
# a linear function is defined as nn.Linear(num_input_nodes, num_output_nodes)
model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.Sigmoid(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
```

Next, let's define what the loss function will be.

For this example, we are going to use Mean Squared Error, but there are a ton of different loss functions we can use.

MSE is defined as: ![](https://cdn-media-1.freecodecamp.org/images/hmZydSW9YegiMVPWq2JBpOpai3CejzQpGkNG)

Where `y` is our target (or called a label) and `y` with a `~` on top is our predicted value.

```python
criterion = nn.MSELoss()
```

Now we need an optimizer to update our weights of our model based on the loss. The loss function will calculate the gradients for each weight, and the optimizer will then update each of those weights based on it's gradient. How it updates the weights is based on the optimizer used.

In this example, we are going to be using a standard gradient descent method, called **Stochastic Gradient Descent.**
SGD will update our weights much like we showed in the slides, but instead of calculating the gradient for the entire data set it does it for a *randomly* selected subset of the data. Learn more [here.](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

We will have a learning rate of 0.01, which is a standard starting point.
You are going to want to keep this learning rate pretty low, as high learning rates cause problems in training, where the steps are too large such that the model can't converge to a minimum loss.

```python
# pass the parameters of our model to our optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

How does Pytorch calculate derivatives and update our weights? How does it know how our model is connected? These are great questions, although we can't cover them here for time sakes. Read up how Pytorch uses what it calls `autograd` and computational graphs [here.](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)

Now, let's train for 50 **epochs** (pronounced either *epic* or *e-pok*)!

To train, we combine all the different parts that we defined into one for loop.

(An epoch is a full pass through *all* of our training data. Normally, we do many epochs to train our model.)

```python
# put our model and data onto our device for increase speed
model = model.to(device)
x, y = x.to(device), y.to(device)
for epoch in range(50):
    # Forward Pass
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    
    # Zero the gradients, this is needed so we don't keep gradients from the last interation
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()
```

In this example, we printed out the loss each time we completed an epoch.

This ran very quickly, but with more complex models, those outputs are going to be very 
important for checking on how your network is doing during the training process in which it could take hours, if not days!

More likely than not, you're going to see that this network is not converging, which is to be expected with random data. 

In our next example, we're going to be building a proper model with an awesome dataset. 

## Diabetes dataset

What causes someone to have diabetes? 

With this dataset, we are going to see if some basic medical data about a person can help us 
predict if someone is diabetic or not using *magical* neural networks. 

First though, let's get that dataset and see what's inside.

```python
dataset = pd.read_csv(f"{DATA_DIR}/train.csv", header=None)

dataset.head()
```

## What are we looking at?

This is a fairly small dataset that includes some basic information about an individual's health. 

Using this information, we should be able to make a model that will allow us to determine if a person has diabetes or not. 

The last column, `Outcome`, is a single digit that tells us if an individual has diabetes. 

We may need to clean up the data a bit, so lets take a look at it.

```python
dataset.info()
```

It seems each column is an object instead of a integer/float. This is because our column labels is actually the first row entry in our dataset.

Let's now get a numpy array of our data using `.values`, get rid of the first row with the labels on them, and convert it to floats.

```python
dataset = dataset.values
dataset = np.delete(dataset, 0, 0).astype(np.float32)

dataset = pd.DataFrame(dataset) # convert back to a dataframe
dataset.head()
```

```python
dataset.info()
```

Finally, lets make sure there is no NaN values in our dataset.

```python
dataset.isna().sum()
```

Alright, no nulls! We are good to go. Let's break up our data into test and train set.

Once we have those sets, we'll need to convert them to tensors. 

This bit of code below does just that!

```python
# get numpy array from our dataframe
dataset = dataset.values

# split into x and y sets

X = dataset[:,:-1].astype(np.float32)

Y = dataset[:,-1].astype(np.float32)

# Our Y shape is missing the second axis, so add that now since pytorch won't accept the data otherwise
Y = np.expand_dims(Y, axis = 1)

# Test-Train split
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1)
```

### PyTorch Dataset
Our next step is to create PyTorch **Datasets** for our training and validation sets. 
**torch.utils.data.Dataset** is an abstract class that represents a dataset and 
has several handy attributes we'll utilize from here on out. Check out the docs [here.](https://pytorch.org/docs/stable/data.html)

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
        # for more advanced dataset, more preprocessing would go here
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

Pytorch dataloaders are especially efficient since they load in data while the model is training in advance, so a GPU can be dedicated to training while the CPU handles all the data preprocessing and loading.

```python
dataloaders = {x: DataLoader(datasets[x], batch_size=16, shuffle=True, num_workers = 4)
              for x in ['Train', 'Validation']}
```

### PyTorch Models

Much like how we can define a custom dataset, we can create a class to define a custom model. Models inherit from the `torch.nn.Module` class. Only two functions must be overwritten, the `__init__` constructor and the `forward` method, which defines the forward pass through each of the layers of the network. Models contain our layer types, and activation functions, plus any other needed operations (like concatenation or matrix multiply). Read up on nn.Module [here.](https://pytorch.org/docs/stable/nn.html)

There are many ways to create models, you can define layers statically and call them one by one, you can build layers using loops based on some input parameters (like number of layers), or you can load from a config file. Some are more extendable then others, but we will keep it basic and define each step so you can see the process of creating a model.

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # call the constructor of the super class
        super(NeuralNet, self).__init__()
        # define our input->hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # define our hidden->output layer
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        # pass through our first linear function
        x = self.fc1(x)
        # apply relu activation function
        x = F.relu(x)
        # pass through second layer, which outputs our final raw value
        x = self.fc2(x)
        # apply sigmoid activation to get a prediction value between 0 and 1
        x = torch.sigmoid(x)
        
        return x
```

### PyTorch Training

Now we create an instance of this NeuralNet() class and define the loss function and optimizer 
we'll use to train our model. 
In our case we'll use Binary Cross Entropy Loss, a commonly used loss function binary classification problems. It calculates the **entropy** between a target value and predicted value, the lower the entropy the closer the two values are. Read up more [here!](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy)

For the optimizer we'll use Adam, a powerful optimizer which is an extension of the popular 
Stochastic Gradient Descent method. When in doubt, this one of the best optimizers to first try. It does cool things like *dynamically reduce learning rate* and generate *momentum* to get out of local minima. Check it out [here!](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)

We need to pass it all of the parameters it'll train, 
which PyTorch makes easy with model.parameters(), and also the learning rate we'll use.

There is also a helpful package to print summaries of torch models, called `torchsummary.`

```python
inputSize =  8         # how many pieces of data for input
hiddenSize = 15        # Number of units in the middle hidden layer
numClasses = 1         # Only has two classes, so 1 output node
numEpochs = 20         # How many training cycles
learningRate = .01     # Learning rate

# Creating our model
model = NeuralNet(inputSize, hiddenSize, numClasses)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = learningRate)
model.to(device) # remember to put your model onto the device!

summary(model, X.shape)
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
  
  # holds values for our total loss and accuracy
  running_loss = 0.0
  running_corrects = 0
  
  # put model into the proper mode.
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
    running_loss += loss.item() * inputs.size(0) # .item() gets the value of the loss as a raw value
    running_corrects += torch.sum(preds == labels) # sums all the times where the prediction equals our label (correct)
    
  # Calculate epoch statistics
  epoch_loss = running_loss / len(datasets[phase])
  epoch_acc = running_corrects.double() / len(datasets[phase])
  
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
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt

matthews_corrcoef(preds, yTest)
```

Let's check the confusion matrix

```python
confusion = confusion_matrix(preds, yTest)

def plot_confusion_matrix(confusion):
  categories = ["Not Diabetic", "Diabetic"]
  fig, ax = plt.subplots()
  im = ax.imshow(confusion)
  ax.set_yticks(np.arange(len(categories)))
  ax.set_yticklabels(categories)

  for i in range(len(categories)):
    for j in range(len(confusion)):
      ax.text(i, j, confusion[i, j], ha="center", va="center", color="white")

plot_confusion_matrix(confusion)
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
test_data = pd.read_csv(f"{DATA_DIR}/test.csv", header=None).values
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
