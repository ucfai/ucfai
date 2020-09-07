---
title: How We Give Our Computers Eyes and Eyes
linktitle: How We Give Our Computers Eyes and Eyes

date: '2019-10-16T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 0

menu:
  core_fa19:
    parent: Fall 2019

authors: [danielzgsilva, brandons209]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: https://kaggle.com/ucfaibot/core-fa19-cnns
  colab: ''

papers: {}

location: MSB 359
cover: ''

categories: [fa19]
tags: [computer vision, CNNs, convolutional networks, deep learning, neural networks]
abstract: >-
  Ever wonder how Facebook tells you which friends to tag in your photos,
  or how Siri can even understand your request? In this meeting we'll dive
  into convolutional neural networks and give you all the tools to build
  smart systems such as these. Join us in learning how we can grant our  computers
  the gifts of hearing and sight!

---

```python
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-cnns").exists():
    DATA_DIR /= "ucfai-core-fa19-cnns"
elif DATA_DIR.exists():
    # no-op to keep the proper data path for Kaggle
    pass
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-cnns/data
    DATA_DIR = Path("data")
```

## Convolutional Neural Networks and Transfer Learning Workshop
There is a notebook in the github repo for this workshop that has much of the content from the slides in there for your convenience. 

## Set up 

Importing some of the libraries we'll be using, as well as PyTorch:


```python
# standard imports (Numpy, Pandas, Matplotlib)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# PyTorch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms
from torchsummary import summary

# Extras
import time
import os
import glob
```


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
%pylab inline
random.seed(42)
```

## Building a Convolutional Neural Network with PyTorch

Now that we understand the details behind CNNs, let's take a look at how we can build one of these networks using the **[PyTorch](https://pytorch.org/docs/stable/index.html)** framework. As I mentioned earlier, CNNs can be used to understand all sorts of data, but for this meeting we'll build a network to classify images. This is called **Computer Vision**.

Before we can begin building our model, we need to set up our dataset in such a way that allows PyTorch to properly load each image.

#### Introduction to the dataset

The dataset which we'll be working with is the popular dog breeds dataset, which contains a few thousand pictures of 133 different breeds of dogs. Naturally, our goal will be to create a model which can predict the breed of dog of any given image.

Example of a Bulldog <br>
<img src = "https://drive.google.com/uc?id=12BamjMMri9N3nkiGvS186US7FXKrUnLH">
 <br>Here's a German Shepard<br>
<img src= "https://drive.google.com/uc?id=1KuIfY2niIJ-7e-B5gNzz1joCAbkqcpXe">

#### PyTorch data transformations

The first step in doing so is to define the transformations that will be applied to our data. These are simply the preprocessing steps that are applied to each image before being fed into our model.

As you can see above, the pictures are all different dimensions, while most CNNs expect each input to be a consistent size... So we define a fixed size for every image as well as a few other constants which I'll explain in a bit.


```python
input_size = (224,224)
batch_size = 32
num_workers = 4
```

This code defines the transformations for each of our datasets (Training, Validation, and Test sets). **Compose()** simply chains together PyTorch transformations. 

The first transformation we apply is the resizing step we discussed above. The next step, `ToTensor()`, transforms the pixel array into a PyTorch `Tensor` and rescales each pixel value to be between $[0, 1]$. This is required for an input to be consumed by PyTorch. Finally, we normalize each `Tensor` to have a mean $\mu = 0$ and variance $\sigma^2 = 1$. Research supports that Neural Networks tend to perform much better on normalized data... 



```python
data_transforms = {
    'Train': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ]),
    'Validation': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ]),
    'Test': transforms.Compose([transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ])
}
```

#### PyTorch datasets and dataloaders

Our next step is to create a PyTorch `Dataset` for each of our training, validation, and test sets. `torch.utils.data.Dataset` is an abstract class that represents a dataset and has several handy attributes we'll utilize from here on out.

If you look at the folder of images we downloaded earlier you'll see it's structured something like this:
```
/imageFolder/Train/Breed1/image_1.jpg
/imageFolder/Train/Breed1/image_2.jpg
.
.
/imageFolder/Train/Breed_133/image_3.jpg
/imageFolder/Train/Breed_133/image_4.jpg

/imageFolder/Validation/Breed1/image_5.jpg
/imageFolder/Validation/Breed1/image_6.jpg
.
.
/imageFolder/Validation/Breed_133/image_7.jpg
/imageFolder/Validation/Breed_133/image_8.jpg

/imageFolder/Test/Breed1/image_9.jpg
/imageFolder/Test/Breed1/image_10.jpg
.
.
/imageFolder/Test/Breed_133/image_11.jpg
/imageFolder/Test/Breed_133/image_12.jpg
```
This structure with subfolders for each class of image is so popular that PyTorch created this function, ImageFolder, which takes a folder and returns a Dataset class for us. The label for each image is automatically interpretted from the name of the folder it sits in. In the line of code below we use this function to create a dictionary of PyTorch Datasets (Train, Validation, Test), passing in the dictionary of transformations we defined above.


```python
image_datasets = {
    x: ImageFolder(os.path.join(DATA_DIR, x),data_transforms[x])
    for x in ['Train', 'Validation']
}

# dataset class to load images with no labels, for our testing set to submit to
#   the competition
class ImageLoader(Dataset):
    def __init__(self, root, transform=None):
        # get image file paths
        self.images = sorted(
            glob.glob(os.path.join(root, "*")),
            key=self.glob_format
        )
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
            return img
        else:
            return transforms.ToTensor(img)
        
    @staticmethod
    def glob_format(key):     
        key = key.split("/")[-1].split(".")[0]     
        return "{:04d}".format(int(key))
    
image_datasets['Test'] = ImageLoader(
    str(DATA_DIR / "Test"),
    transform=data_transforms["Test"]
)
```

The pixel array of each image is actually quite large, so it'd be inefficient to load the entire dataset onto your RAM at once. Instead, we use PyTorch DataLoaders to load up batches of images on the fly. Earlier we defined a batch size of 32, so in each iteration the loaders will load 32 images and apply our transformations, before returning them to us.

For the most part, Neural Networks are trained on **batches** of data so these `DataLoader`s greatly simplify the process of loading and feeding data to our network. The rank 4 tensor returned by the `DataLoader` is of size (32, 224, 224, 3).


```python
dataloaders = {
    x: DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    for x in ['Train', 'Validation']
}

test_loader = DataLoader(
    dataset=image_datasets['Test'],
    batch_size=1,
    shuffle=False
)
```

Every PyTorch `Dataset` has an attribute, `.classes`, which is an array containing all of the image classes in the dataset. In our case, breeds of dog in the dataset. 


```python
dog_breeds = image_datasets['Train'].classes
print(dog_breeds)
```


```python
# Just printing the number of images in each dataset we created

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}

print('Train Length: {} | Valid Length: {} | Test Length: {}'.format(
    dataset_sizes['Train'], 
    dataset_sizes['Validation'],
    dataset_sizes['Test']
))
```


```python
# Here we're defining what component we'll use to train this model
# We want to use the GPU if available, if not we use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
```

#### Visualizing the dataset

Once we've set up our PyTorch datasets and dataloaders, grabbing individual images or batches of images is super simple. Below I've defined 2 functions we can use to take a look at the dogs in our dataset.

The first one here indexes into our training set, grabs a given number of random images, and plots them. A PyTorch `Dataset` is *sort of* a 2D array, where the first dimension represents the images themselves, and the second dimension contains the pixel array and the label of the image.

The second function allows us to plot a batch of images served up by our PyTorch dataloader.


```python
# Plots a given number of images from a PyTorch Data
def show_random_imgs(num_imgs):
    for i in range(num_imgs):
        # We're plotting images from the training set
        train_dataset = image_datasets['Train']
        
        # Choose a random image
        rand = np.random.randint(0, len(train_dataset) + 1)
        
        # Read in the image
        ex = img.imread(train_dataset.imgs[rand][0])
        
        # Get the image's label
        breed = dog_breeds[train_dataset.imgs[rand][1]]
        
        # Show the image and print out the image's size
        #   (really the shape of it's array of pixels)
        plt.imshow(ex)
        print('Image Shape: ' + str(ex.shape))
        plt.axis('off')
        plt.title(breed)
        plt.show()
        

# Plots a batch of images served up by PyTorch    
def show_batch(batch):
    # Undo the transformations applied to the images when loading a batch
    batch = batch.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    batch = std * batch + mean
    batch = np.clip(batch, 0, 1)
    
    # Plot the batch
    plt.axis('off')
    plt.imshow(batch)
    
    # pause a bit so that plots are updated
    plt.pause(0.001)
```


```python
show_random_imgs(3)
```


```python
# Get a batch of training data (32 random images)
imgs, classes = next(iter(dataloaders['Train']))

# This PyTorch function makes a grid of images from a batch for us
batch = torchvision.utils.make_grid(imgs)

show_batch(batch)
```

#### Defining a network in PyTorch

Now its time to finally build our CNN.  In PyTorch, a model is represented by a normal Python class that inherits from the master nn.Module class. Inheriting from this master class grants your model all the methods and attributes needed to train and work with your model. There are, however, 2 things you need to write yourself:
 - `__init__(self)`: Here is where you define the layers and overall architecture of your model
 - `forward(self, x)`: This method takes an input, x, computes a forward pass through the network and outputs predictions. Writing it essentially involves connecting your layers and setting up the flow of the input through your layers.
 
 

Below are the signatures of the PyTorch functions that create each of the layers we discussed. Try to use them to build your first CNN! I provided some comments that hopefully guide you in terms of what should happen at each step.

- `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`
- `nn.ReLU(x)`
- `nn.MaxPool2d(kernel_size, stride, padding)`
- `nn.BatchNorm2d(num_features)` - num_features is the number of channels it receives
- `nn.Dropout(p)` - p is probability of an element to be zeroed
- `nn.Linear(in_features, out_features)` – fully connected layer (matrix multiplications used in the classification portion of a network)


```python
# It is good practice to maintain input dimensions as the image is passed
#   through convolution layers
# With a default stride of 1, and no padding, a convolution will reduce image
#   dimenions to:
#     out = in - m + 1, where _m_ is the size of the kernel and _in_ is a
#        dimension of the input

# Use this function to calculate the padding size neccessary to create an output
#   of desired dimensions

def get_padding(input_dim, output_dim, kernel_size, stride):
    # Calculates padding necessary to create a certain output size,
    # given a input size, kernel size and stride
    padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2
  
    if padding < 0:
        return 0
    else:
        return padding
```


```python
# Make sure you calculate the padding amount needed to maintain the spatial
#   size of the input after each Conv layer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # nn.Sequential() is simply a container that groups layers into one object
        # Pass layers into it separated by commas
        self.block1 = nn.Sequential(
            
            # The first convolutional layer. Think about how many channels the
            #   input starts off with
            # Let's have this first layer extract 32 features
            ### BEGIN SOLUTION
            nn.Conv2d(3, 32, 3, 1, 1),
            ### END SOLUTION
            
            # Don't forget to apply a non-linearity
            ### BEGIN SOLUTION
            nn.ReLU())
            ### END SOLUTION
        
        self.block2 =  nn.Sequential(
            
            # The second convolutional layer. How many channels does it receive,
            #   given the number of features extracted by the first layer?
            # Have this layer extract 64 features
            ### BEGIN SOLUTION
            nn.Conv2d(32, 64, 3, 1, 1),
            ### END SOLUTION
            
            # Non linearity
            ### BEGIN SOLUTION
            nn.ReLU(),
            ### END SOLUTION
            
            # Lets introduce a Batch Normalization layer
            ### BEGIN SOLUTION
            nn.BatchNorm2d(64),
            ### END SOLUTION
            
            # Downsample the input with Max Pooling
            ### BEGIN SOLUTION
            nn.MaxPool2d(2, 2, 0)
            ### END SOLUTION
        )
        
        # Mimic the second block here, except have this block extract 128
        #   features
        self.block3 =  nn.Sequential(
            ### BEGIN SOLUTION
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, 0)
            ### END SOLUTION
        )
        
        # Applying a global pooling layer
        # Turns the 128 channel rank 4 tensor into a rank 2 tensor of size
        #   32 x 128 (32 128-length arrays, one for each of the inputs in a
        #   batch)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128, 512)
        
        # Introduce dropout to reduce overfitting
        self.drop_out = nn.Dropout(0.5)
        
        # Final fully connected layer creates the prediction array
        self.fc2 = nn.Linear(512, len(dog_breeds))
    
    # Feed the input through each of the layers we defined 
    def forward(self, x):
        
        # Input size changes from (32 x 3 x 224 x 224) to (32 x 32 x 224 x 224)
        x = self.block1(x)
        
        # Size changes from (32 x 32 x 224 x 224) to (32 x 64 x 112 x 112)
        #   after max pooling
        x = self.block2(x)
        
        # Size changes from (32 x 64 x 112 x 112) to (32 x 128 x 56 x 56)
        #   after max pooling
        x = self.block3(x)
        
        # Reshapes the input from (32 x 128 x 56 x 56) to (32 x 128)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer, size changes from (32 x 128) to (32 x 512)
        x = self.fc1(x)
        x = self.drop_out(x)
        
        # Size change from (32 x 512) to (32 x 133) to create prediction arrays
        #   for each of the images in the batch
        x = self.fc2(x)
        
        return x
```

Now we create an instance of this `CNN()` class and define the loss function and optimizer we'll use to train our model. In our case we'll use `CrossEntropyLoss`. You'll notice we never added a `Softmax` activation after our last layer. That's because PyTorch's `CrossEntropyLoss` applies a softmax before calculating log loss, a commonly used loss function for single label classification problems.

For the optimizer we'll use `Adam`, an easy to apply but powerful optimizer which is an extension of the popular Stochastic Gradient Descent method. We need to pass it all of the parameters it'll train, which PyTorch makes easy with `model.parameters()`, and also the learning rate we'll use.


```python
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 5

model.to(device)
summary(model, (3, 224, 224))
```

## Training a model in PyTorch 

At this point we're finally ready to train our model! In PyTorch we have to write our own training loops before getting to actually train the model. This can seem daunting at first, so let's break up each stage of the training process. 

The bulk of the function is handled by a nested for loop, the outer looping through each epoch and the inner looping through all of the batches of images in our dataset. Each epoch has a training and validation phase, where batches are served from their respective loaders. Both phases begin by feeding a batch of inputs into the model, which implicity calls the `forward()` function on the input. Then we calculate the loss of the outputs against the true labels of the batch. 

If we're in training mode, here is where we perform back-propagation and adjust our weights. To do this, we first zero the gradients, then perform backpropagation by calling `.backward()` on the loss variable. Finally, we call `optimizer.step()` to adjust the weights of the model in accordance with the calculated gradients.

The remaining portion of one epoch is the same for both training and validation, and simply involves calculating and tracking the accuracy achieved in both phases. A nifty addition to this training loop is that it tracks the highest validation accuracy and only saves weights which beat that accuracy, ensuring that the best performing weights are returned from the function.


```python
def run_epoch(epoch, model, optimizer, dataloaders, device, phase):
  
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

            # Gets the predictions of the inputs (highest value in the array)
            _, preds = torch.max(outputs, 1)

            # Adjust weights through backpropagation if we're in training phase
            if phase == 'Train':
                loss.backward()
                optimizer.step()

        # Document statistics for the batch
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    # Calculate epoch statistics
    epoch_loss = running_loss / image_datasets[phase].__len__()
    epoch_acc = running_corrects.double() / image_datasets[phase].__len__()

    return epoch_loss, epoch_acc

```


```python
def train(model, criterion, optimizer, num_epochs, dataloaders, device):
    start = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t| Epoch Time |')
    print('-' * 86)
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        
        epoch_start = time.time()
       
        # Training phase
        train_loss, train_acc = run_epoch(epoch, model, optimizer, dataloaders, device, 'Train')
        
        # Validation phase
        val_loss, val_acc = run_epoch(epoch, model, optimizer, dataloaders, device, 'Validation')
        
        epoch_time = time.time() - epoch_start
           
        # Print statistics after the validation phase
        print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.0f}m {:.0f}s     |"
                      .format(epoch + 1, train_loss, train_acc, val_loss, val_acc, epoch_time // 60, epoch_time % 60))

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

#### Testing a model

Creating a function that generates and prints predictions on a given number of images from our test set:


```python
def test_model(model, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(num_images, (10,10))

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloaders['Validation']):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Actual: {} \n Prediction: {}'.format(dog_breeds[labels[j]], dog_breeds[preds[j]]))
                
                image = images.cpu().data[j].numpy().transpose((1, 2, 0))
                
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                
                plt.imshow(image)
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```

After defining these functions, training and testing our model is straightforward from here on out. Simply call the train() function with the required parameters and let your GPU go to work!


```python
# Make sure to comment this out when you go to "Commit" the kaggle notebook!
# otherwise, it'll run this model along with your other models down below.
model = train(model, criterion, optimizer, epochs, dataloaders, device)
```

Ouch! Our model doesn't seem to be performing very well at all. After 20 epochs of training we're barely able to achieve a 10% accuracy on our validation set... Hang in there, in a bit I'll go into some methods we can use to achieve a much better accuracy.

In the meantime, let's quickly take a look at how we can save our PyTorch models. Then we'll test and visualize our model. 

## Saving a model in PyTorch 

There are many ways to save a PyTorch model, however the most robust method is described below. This allows you to load up a model for both testing and further training.

The most important part to understand from the code below is what the `model` and `optimizer` `.state_dict()`s are. The model state_dict is essentially a dictionary which contains all of the learned weights and biases in the model, while the optimizer contains information about the optimizer’s state hyperparameters used.

Other than the state_dicts, we also save the class used to build the model architecture, as well as the optimizer and loss function. Putting all of this together allows us to save, move around, and later restore our model to it's exact state after training.. A `.pt` file extension is commonly used to bundle all of this together.


```python
torch.save({
    'model' : CNN(),
    'epoch' : epochs,
    'model_state_dict': model.state_dict(),
    'optimizer' : optimizer,
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion' : criterion,
    'device' : device
}, 'base_model.pt')
```

Creating a function which unpacks the `.pt` file we saved earlier and loads up the model's saved weights and optimizer state:


```python
def load_checkpoint(filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    epoch = checkpoint['epoch']
    model.to(device)

    return model, optimizer, criterion, epoch
```

Loading our model up...


```python
model, optimizer, criterion, epoch = load_checkpoint('base_model.pt')
```

Let's test our model on a couple of dogs!


```python
test_model(model, 6)
```

As expected, our model is predicting the wrong breed for the majority of test images. Why is this?

In short, building and training a CNN from scratch is possible, however most problems require significantly more complex models, trained on huge amounts of data. Of course, the computational power and amount of data needed to train these networks accurately are not always available. This is why the idea of **Transfer Learning** has become so popular. It allows everyday people, like me and you, to build accurate and powerful models with limited resources.

## Transfer Learning  

In transfer learning, we take the architecture and weights of a pre-trained model (one that has been trained on millions of images belonging to 1000’s of classes, on several high power GPU’s for several days) and use the pre-learned features to solve our own novel problem.

PyTorch actually comes with a number of models which have already been trained on the Imagenet dataset we discussed earlier, making it quite simple for us to apply this method of transfer learning. We'll be using a powerful but lighweight model called `ResNet18`, which we import like so:
```python
from torchvision.models import resnet18
```

The next block of code might look a bit foreign. What we're doing is actually looping through all of the model's pretrained weights and **freezing** them. This means that during training, these weights will not be updating at all. We then take the entire ResNet model and put it into one block of our model, named feature_extraction. It's important to understand that when you load a pretrained model you are only receiving the feature extraction block, or the convolutional layers. It's up to us to define a classification block which can take all of the features the ResNet model extracted and use them to actually classify an image.


```python
class PreTrained_Resnet(nn.Module):
    def __init__(self):
        super(PreTrained_Resnet, self).__init__()
        
        # Loading up a pretrained ResNet18 model
        resnet = resnet18(pretrained = True)
        
        # Freeze the entire pretrained network
        for layer in resnet.parameters():
            layer.requires_grad = False
            
        self.feature_extraction = resnet
        
        # Write the classifier block for this network      
            # Tip: ResNet18's feature extraction portion ends up with 1000
            #   feature maps, and then implements a Global Average Pooling layer
            # So what would the size and dimension of the output tensor be?
            # Think about how can we take that output tensor and transform it
            #   into an array of dog breed predictions...
        self.classifier = nn.Sequential(
            ### BEGIN SOLUTION
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(dog_breeds))
            ### END SOLUTION
        )
    
    # Write the forward method for this network (it's quite simple since we've
    #   defined the network in blocks already)
    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.feature_extraction(x)
        x = self.classifier(x)
        return x
        ### END SOLUTION
```


```python
# Instantiate a pretrained network using the class we've just defined (call it
#  'pretrained')

### BEGIN SOLUTION
pretrained = PreTrained_Resnet()
### END SOLUTION

# Then define the loss function and optimizer to use for training (let's use
#   Adam again, with the same parameters as before)
### BEGIN SOLUTION
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(pretrained.classifier.parameters(), lr = 0.0001)
### END SOLUTION

# Define your number of epochs to train and map your model to the gpu
# Keep epochs to 5 for time purposes during the workshop
### BEGIN SOLUTION
epochs2 = 5
pretrained.to(device)
### END SOLUTION

summary(pretrained, (3,224,224))
```


```python
pretrained = train(
    pretrained,
    criterion2,
    optimizer2,
    epochs2,
    dataloaders,
    device
)
```

This quick example shows the power of transfer learning. With relatively few lines of code we're able to achieve over an 80% accuracy on this dog breeds dataset! And there are still a number of things we could have done, or do from here, to achieve even better performance. This includes things such as:
- Unfreezing the last few layers of the ResNet base and training some more on our specific dataset (more on this in a bit)
- Optimizing the hyperparameters of our model (learning rate, etc.)
- Utilizing an even more powerful pretrained architecture (`ResNet34`, `ResNet50`, etc.)
- Creating a custom learning rate schedule

We'll save the model, then load it back up using the function we defined earlier.


```python
torch.save({
    'model' : PreTrained_Resnet(),
    'epoch' : epochs2,
    'model_state_dict': pretrained.state_dict(),
    'optimizer' : optimizer2,
    'optimizer_state_dict': optimizer2.state_dict(),
    'criterion' : criterion2,
    'device' : device
}, 'pretrained.pt')
```


```python
pretrained, optimizer2, criterion2, epoch2 = load_checkpoint('pretrained.pt')
```

Finally we can test our new pretrained ResNet model! As you can see, with transfer learning we can create quite accurate models relatively easily.


```python
test_model(pretrained, 6)
```

#### More on Transfer Learning

In this example, we simply took a pretrained model and added our classification (fully connected layers) block right on top. We froze the entire pretrained network and only updated the weights of our fully connected layers. This means we didn't change the pretrained weights at all, and only used what it had 'learned' from the dataset which it was trained on. 

However, I mentioned earlier that we could achieve even better performance if we unfroze the last few layers of the pretrained model and trained them some on our specific dataset. But why?

<img src = "https://drive.google.com/uc?id=10ce5aTD47lIsO1eYfZmbs_sbDDUfaZiT"> <img src = "https://drive.google.com/uc?id=1BfHJXrWwl4oVyPZ2_p602nD9HkF4RoSR">

Going back to the layer visualizations we saw earlier, we know the earlier layers of the pretrained network learn to recognize simple lines, patterns, objects, etc. However, as we progress in the network, the layers learn to recognize things more specific to the dataset which it was trained on. In this case, ImageNet, which we described a bit earlier.

If you remember, ImageNet contains images that are *somewhat* similar to our dog breeds dataset, so much of what the model 'learned' also applied to our dataset. Hence why we were able to achieve a pretty good accuracy without adjusting the pretrained model whatsoever. 

Of course, much of what the deeper layers learned from ImageNet did **not** apply to dog images. This is why training the last few layers would be beneficial. It would allow the model to adjust and recognize rich features specific to **only dogs**. Things such as types of dog ears, tails, fur, noses, etc. etc.


```python
# Run this to generate the submission file for the competition!
### Make sure to name your model variable "pretrained" ###

# generate predictions
preds = []
pretrained = pretrained.to(device)
pretrained.eval()
for img in test_loader:
    outputs = pretrained(img.to(device))
    _, outputs = torch.max(outputs, 1)
    preds += [outputs.item()]

# create our pandas dataframe for our submission file. Squeeze removes
#   dimensions of 1 in a numpy matrix Ex: (161, 1) -> (161,)
indicies = ["{}.jpg".format(x) for x in range(len(image_datasets['Test']))]
preds = pd.DataFrame({'Id': indicies, 'Class': np.squeeze(preds)})

# save submission csv
preds.to_csv('submission.csv', header=['Id', 'Class'], index=False)
print("Submission generated!")
```

## Thank you for coming out tonight! 

## Don't forget to sign in at <a href="ucfai.org/signin">ucfai.org/signin</a> if you didn't get the chance to swipe in, and see you next week!
