---
title: "Machine Learning Applications"
linktitle: "Machine Learning Applications"

date: "2020-02-26T17:30:00"
lastmod: "2020-03-18T23:54:14.006180763"

draft: false
toc: true
type: docs

weight: 6

menu:
  core_sp20:
    parent: Spring 2020
    weight: 6

authors: ["brandons209", "nspeer12", ]

urls:
  youtube: ""
  slides:  ""
  github:  ""
  kaggle:  "https://kaggle.com/ucfaibot/core-sp20-ml-apps"
  colab:   ""

location: "HPA1 112"
cover: ""

categories: ["sp20"]
tags: ["Applications", "Pokemon", "Pokedex", "Exoplanets", "Machine Learning", ]
abstract: >-
  It's time to put what you have learned into action. Here, we have prepared some datasets for you to build a a model to solve. This is different from past meetings, as it will be a full workshop. We provide the data sets and a notebook that gets you started, but it is up to you to build a model to solve the problem. So, what will you be doing? We have two datasets, one is using planetary data to predict if a planet is an exoplanet or not, so your model can help us find more Earth-like planets that could contain life! The second dataset will be used to build a model that mimics a pokedex! Well, not fully, but the goal is to predict the name of a pokemon and also predict its type (such as electric, fire, etc.) This will be extremely fun and give you a chance to apply what you have learned, with us here to help!
---

```python
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-ml-apps").exists():
    DATA_DIR /= "ucfai-core-sp20-ml-apps"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-ml-apps/data
    DATA_DIR = Path("data")
```


```python
from pathlib import Path
import os

if os.path.exists("/kaggle/input/ucfai-core-sp20-ml-apps"):
    DATA_DIR = Path("/kaggle/input/ucfai-core-sp20-ml-apps")
else:
    DATA_DIR = Path("data/")
```


```python
# general stuff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd

# sklearn models and metrics
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# pytorch imports
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

!pip install torchsummary
from torchsummary import summary
```

# Kepler Exoplanet Search
NASA's Kepler Mission is a quest to search for planets that may be suitable for life. In order to do this, NASA surveyed many solar systems in the Milky Way and identified planets with unique qualities similar to Earth. These qualities inclue:
- Determine the percentage of terrestrial and large planets that are in or near the habitable zoen from a variety of stars
- Determine the distribution of sizes and shapes of orbits
- Determine the planet relectivities, size, masses and densities of short-period giant planets
- Identify additional planets in each solar system
- Determine the properties of stars that have planetary systems

## Transit Method of Detecting Extrasolar Planets
A "transit" is the event where a planet passes in front of a star, as viewed from Earth. Occasionally we can observe Venus or Mercury transit the Sun. Kepler finds planets by looking for tiny dips in the brightness of a star when a planet crosses in front of it. Once detected, the planet's orbital size can be calculated from the period of the orbit, and the mass of the sun can be calculated using Kepler's Third Law. The size of the planet is found by analyzing the dip in the amount of light we perceive here on earth. When we have the planet's orbital size and temperature of the sun, we can assume some characteristics of the planet, such as temperature, and from that we can assume whether or not a planet is habitable.

source: https://www.nasa.gov/mission_pages/kepler/overview/index.html    
NASA dataset: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi


```python
df = pd.read_csv(DATA_DIR / 'cumulative.csv', delimiter=',')
df.head()
```

## About the Dataset
`KOI`: Kepler Object of Interest    
`kepoi_name` the name of the target   
`koi_disposition` the disposition in the literature towards being an expolanet canidate   
`koi_pdisposition` the disposition from data analysis towards being an exoplanet canidate   
`koi_score` A value between 0 and 1 that indicates the confidence of the koi disposition   
`koi_period` the period of the orbit    
`koi_impact` the impact of the transit. The dip of observed light.    
`koi_slogg` The base-10 logarithm of the acceleration due to gravity at the surface of the star.    
`koi_srad` The radius of the sun    

Locational data relative to Earth    
`ra` Right ascension. https://en.wikipedia.org/wiki/Right_ascension    
`dec` Declination. https://en.wikipedia.org/wiki/Declination    

more explanations: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html

Our goal is to predict whether a planet is a CANDIDATE for being an exoplanet or a FALSE POSITIVE, which just means it isn't a candidate.

![](http://)We also want to drop some of these columns, since we don't need the names, flags, or the koi_score (since that can skew our model since it practically gives the answer.)


```python
# make a new column that is a binary classificaiton of whether or not a planet is a canidate
disposition = [0] * len(df['koi_pdisposition'])
for i in range(len(df['koi_pdisposition'])):
    disposition[i] = 1 if (df['koi_pdisposition'][i] == 'CANDIDATE') else 0 

df.insert(1, "disposition", disposition)

columns = ["disposition", "koi_period", "koi_impact", "koi_srad", "koi_slogg", "ra", "dec"]
df = df[columns].dropna()
df.head()
```

## Visualize the data
Next up, lets take a look at the how the data looks when the disposition is 0 (false positive) or 1 (candidate). Below we setup some scatter plots where green dots represent candidates and red dots represent false positives. This can help us see what model we might need. For example, if the data is grouped nicely K-nearest neighbors may help us!


```python
# visualize data

# custom color map for our dataset
color = np.where(df['disposition'] == 1, 'green', 'red')
fig, axs = plt.subplots(3, figsize=(15,10))

# make sure to play around with these to better understand the dataset

axs[0].scatter(df['koi_slogg'], df['koi_impact'], c=color)
axs[0].set_xlabel("impact")
axs[0].set_ylabel("slogg")

axs[1].scatter(df['koi_srad'], df['koi_impact'], c=color)
axs[1].set_xlabel("impact")
axs[1].set_ylabel("srad")

axs[2].scatter(df['ra'], df['dec'], c=color)
axs[2].set_xlabel("ra")
axs[2].set_ylabel("dec")
```


```python
#X_train, X_test, Y_train, Y_test = train_test_split(df)
X = pd.DataFrame(columns=['koi_period', 'koi_slogg', 'koi_srad', 'koi_impact', 'ra', 'dec'], data=df).values
Y = pd.DataFrame(columns=['disposition'], data=df).values.ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make a model
### BEGIN SOLUTION
# having a bit of fun looking for optimal neighbors
n_best = 1
best_score = 0
for i in range(1, 100):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    corr = matthews_corrcoef(Y_test, predictions)
    print(i, ":", corr)
    if corr > best_score:
        best_score = corr
        n_best = i

print(n_best, 'is the best n!')

model = RandomForestClassifier(150)
model.fit(X_train, Y_train)

### END SOLUTION

# make predictions and test accuracy
### BEGIN SOLUTION
predictions = model.predict(X_test)

# test accuracy
print(matthews_corrcoef(Y_test, predictions))
### END SOLUTION
```

# Next up: Pokedex
So now that we found some alien life, lets try to work with some image and csv data to build a pokedex model!

### Dataset
The dataset contains 150 folders, one for each generation one pokemon and each folder contains 60 images for every pokemon. In all there are over 10,000+ images for all the pokemon. Some of the images have some noise, like other characters from the show and what not, so this dataset can benefit from some data cleaning by manually going through and removing images that have a lot of noise.    
Side note: I am not too familiar with pokemon, but this is missing Nidoran, which looks like Nidorino? So I am not sure about that.

There is also a csv file that contains each pokemons name and it's type. Each pokemon has one primary type, but some have an optional secondary type. In this case, some pokemon's secondary type could be NULL, so we have to deal with that before training.

### The goal
The goal of this is to build a model which can predict a pokemon's name and also it's type. There a few ways to go about doing this, one way is to have a model with two output layers, one for name and one for type. You can also have two seperate models to predict type and name, although this is a bit more resource intensive.

Remember, type could be multi-label, so from our CNN workshop we know that we want to use sigmoid activation for that layer so get multiple labels.

A good first try would be to just predict the primary type, and worry about the secondary type later. In this case, we can use softmax for each output layer.

### Data loading
As used in our CNN workshop, PyTorch has a very nice `ImageFolder` dataset that will load in images and their class based on the folder structure of the data. This structure should be `root/class1/images`, `root/class2/images`, etc. It will also apply PyTorch transforms as well to the images. For the types CSV, we load it using pandas.

Since our image data isn't split up into training and testing sets, we can do that using PyTorch's `random_split` function, which will split datasets into different lengths. The classes returned for this aren't ImageFolders, then are called Subsets, a subset of a dataset.


```python
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms

from torch.utils.data import random_split

# define transform for image data:
input_size = (224, 224)

# this will resize the image, convert it to a tensor, convert pixel values to be in range of [0, 1]
# and normalize the data
data_transform = transforms.Compose([transforms.Resize(input_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
                                    ])

# load in data
dataset = ImageFolder(DATA_DIR / "dataset", transform=data_transform)
types = pd.read_csv(DATA_DIR / "pokemon_fixed.csv")

# split data
test_split = 0.2

# get number of samples that should be in training set
train_size = int(len(dataset) * (1 - test_split))

# split the dataset into training and testing Subsets
train, test = random_split(dataset, (train_size, len(dataset) - train_size))

print(f"Number of training samples: {len(train)}, testing: {len(test)}")
print("\n".join(dataset.classes))
```

Now, lets take a look at some images from the dataset, and the csv file.


```python
def show_random_imgs(num_imgs):
    for i in range(num_imgs):
        # Choose a random image
        rand = np.random.randint(0, len(dataset) + 1)
        
        # Read in the image
        ex = img.imread(dataset.imgs[rand][0])
        
        # Get the image's label
        pokemon = dataset.classes[dataset.imgs[rand][1]]
        
        # Show the image and print out the image's size (really the shape of it's array of pixels)
        plt.imshow(ex)
        print('Image Shape: ' + str(ex.shape))
        plt.axis('off')
        plt.title(pokemon)
        plt.show()

show_random_imgs(3)
```


```python
types.head()
```

Now, lets get our dataloader and our types ready for training.

Here, the secondary type is dropped to just train on the primary type. Once you train on the primary type try predicting the secondary type as well! This is a great challenge for you to do after the workshop.


```python
batch_size = 16

# Define train and test dataloaders
# Name them train_dataloader and test_dataloader
### BEGIN SOLUTION
train_dataloader = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=2)
test_dataloader = DataLoader(test, shuffle=True, batch_size=batch_size, num_workers=2)
### END SOLUTION

types = types.drop("Type2", axis=1)

# need to convert the pokemon name to their class number
classes = {name.lower(): i for i, name in enumerate(dataset.classes)}
types = types.replace(to_replace=classes)

# now we need to turn the types to class indicies as well
unique_types = sorted(types.Type1.unique())
int_to_type = {i: t for i, t in enumerate(unique_types)}
type_to_int = {t: i for i, t in enumerate(unique_types)}

types = types.replace(to_replace=type_to_int)

# turn dataframe into a dictionary
# keys are the class number of the pokemon and it gives the type for the pokemon
types = {t[0]: t[1] for t in types.values}

# finally, lets make a function to get a tensor of target types given input pokemon names
# input should be a torch tensor
def get_types(pokemon_classes):
    return torch.tensor([types[c.item()] for c in pokemon_classes])

print(get_types([torch.tensor(5)]))
```

## Build the model
Now, lets build that model! Please refer to our CNN workshop if you need help. The general structure is below, I would suggest to use a pretrain model and freeze the first few layers of the model, and leave the rest to train. You can also "fine tune" (not freeze any layers) and train. This is because our dataset is different from the original dataset the pretrain models were changed on. Or you can build your own model, up to you!

There is also an example model that returns 2 outputs for your reference. You will also need to calculate loss **seperately** for each output. An example is also shown below.

If you don't want to try to do both at once first, just try predicting the type or just the name and see the results you get!


```python
# example two output model
class TwoOutputModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(500, 250)
        self.out1 = nn.Linear(250, 10)
        self.out2 = nn.Linear(250, 1)
    
    def forward(x):
        x = self.fc1(x)
        
        out1 = self.out1(x)
        out2 = self.out2(x)
        
        return out1, out2

# two output loss example:
# c = nn.CrossEntropyLoss()
# target = 1
# output_1, output_2, = model(input)
# loss_1 = c(output_1, target)
# loss_2 = c(output_2, target)
# loss = loss_1 + loss_2
# loss.backward()
# optimizer.step()
```

Here is the padding function right from our CNNs workshop for your conveinence.


```python
# It is good practice to maintain input dimensions as the image is passed through convolution layers
# With a default stride of 1, and no padding, a convolution will reduce image dimenions to:
            # out = in - m + 1, where m is the size of the kernel and in is a dimension of the input

# Use this function to calculate the padding size neccessary to create an output of desired dimensions

def get_padding(input_dim, output_dim, kernel_size, stride):
  # Calculates padding necessary to create a certain output size,
  # given a input size, kernel size and stride
  
  padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2
  
  if padding < 0:
    return 0
  else:
    return padding

get_padding(224, 224, 3, 1)
```

Here is the structure for the CNN model here. The two output layers are included to help guide you. Everything else you'll need to build!


```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # put your model layers here
        ### BEGIN SOLUTION
        # Loading up a pretrained ResNet18 model
        resnet = resnet18(pretrained = True)

        self.feature_extraction = resnet
        
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        ### END SOLUTION
        
        # define our two output layers
        self.name = nn.Linear(512, len(dataset.classes))
        self.type_ = nn.Linear(512, len(unique_types))
    
    # Write the forward method for this network (it's quite simple since we've defined the network in blocks already)
    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.feature_extraction(x)
        x = self.classifier(x)
        ### END SOLUTION
        
        # x should be the output from your last layer before the output layers
        name = self.name(x)
        type_ = self.type_(x)
        return name, type_
```

## Train and Test
Now lets see what your model can do. Define the criterion (cross entropy loss), optimizer, number of epochs, and the model. Then use the training functions below to train your model.

The training and testing code is also different then what you have seen before, it is a bit simpler. This is so that you see different ways of creating training and testing loops for PyTorch so that you become more familiar when creating your own loops and looking at someone else's.


```python
model = CNN()

# define the criterion and optimizer below, with those names
### BEGIN SOLUTION
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
### END SOLUTION

# Note: may take a bit to train on kaggle!
epochs = 10
checkpoint_path = "best.model.pt"

model.to(device)
summary(model, (3, *input_size))
```


```python
# Defines a test run through our testing data
def test(epoch):
    model.eval()
    test_loss = 0
    correct_names = 0
    correct_types = 0
    total = 0
    for i, data in enumerate(test_dataloader):
        with torch.no_grad(): # doesn't calculate gradients since we are testing
            inputs, targets = data
            
            type_targets = get_types(targets).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # get model outputs, run criterion on each output, then sum losses
            # write out the first 2 parts here, name them loss_name and loss_type for each loss
            ### BEGIN SOLUTION
            name_out, type_out = model(inputs)
            loss_name = criterion(name_out, targets)
            loss_type = criterion(type_out, type_targets)
            ### END SOLUTION
            
            loss = loss_name + loss_type

        test_loss += loss.item()
        _, predicted_names = name_out.max(dim=1)
        _, predicted_type = type_out.max(dim=1)
        
        total += targets.size(0)
        correct_names += predicted_names.eq(targets).sum().item()
        correct_types += predicted_type.eq(type_targets).sum().item()
    
    # defines loss, name accuracy, type accuracy
    results = (test_loss/len(test_dataloader), (correct_names / total) * 100.0, (correct_types / total) * 100.0)
    
    # epoch less than 0 means we are just testing outside training
    if epoch < 0: 
        print("Test Results: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(
            results[0], results[1], results[2]))
    else:
        print("Epoch [{}] Test: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(
            epoch + 1, results[0], results[1], results[2]))

        return results
```


```python
# Training phase
print_step = len(train_dataloader) // 50
best_loss = 0

for e in range(epochs):
    model.train()
    
    # define inital metric values
    train_loss = 0
    correct_names = 0
    correct_types = 0
    total = 0
    
    for i, data in enumerate(train_dataloader):
        inputs, targets = data
        
        # get our type targets using our helper function
        type_targets = get_types(targets).to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # zero out previous gradients
        optimizer.zero_grad()

        # forward
        name_out, type_out = model(inputs)
        
        # backward
        # get model outputs, run criterion on each output, then sum losses
        # write out the first 2 parts here, name them loss_name and loss_type for each loss
        ### BEGIN SOLUTION
        name_out, type_out = model(inputs)
        loss_name = criterion(name_out, targets)
        loss_type = criterion(type_out, type_targets)
        ### END SOLUTION
        
        # sum the losses for backprop
        loss = loss_name + loss_type
        
        # calculate gradients and update weights
        loss.backward()
        optimizer.step()
        
        # calculate our accuracy metrics and loss
        train_loss += loss.item() # .item() extracts the raw loss value from the tensor object
        _, predicted_names = name_out.max(dim=1)
        _, predicted_type = type_out.max(dim=1)
        
        total += targets.size(0)
        correct_names += predicted_names.eq(targets).sum().item()
        correct_types += predicted_type.eq(type_targets).sum().item()

        if i % print_step == 0:
            print("Epoch [{} / {}], Batch [{} / {}]: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(
                e+1, epochs, i+1, len(train_dataloader), train_loss/(i+1), (correct_names / total) * 100.0, (correct_types / total) * 100.0))

    print("Epoch [{} / {}]: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(
        e+1, epochs, train_loss/(len(train_dataloader)), (correct_names / total) * 100.0, (correct_types / total) * 100.0))

    val_loss, val_name_acc, val_type_acc = test(e)
    
    if val_loss < best_loss or e == 0: # model improved
        print('---Loss improved! Saving Checkpoint---')
        state = {'net': model.state_dict(), 'loss': val_loss, 'epoch': e}
        torch.save(state, checkpoint_path)
        best_loss = val_loss

```


```python
best_cp = torch.load(checkpoint_path)
model.load_state_dict(best_cp["net"])

# Lets see the final results
test(-1)
```


```python
def display_results(num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(num_images, (15,20))

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_dataloader): 
            images = images.to(device)
            
            type_targets = get_types(targets)
            
            name_out, type_out = model(images)
            _, predicted_names = name_out.max(dim=1)
            _, predicted_type = type_out.max(dim=1)

            for j in range(images.size()[0]):
                # plot images for display
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                # title is the actual and predicted values
                ax.set_title('Actual Name: {}\nPrediction: {}\nActual Type: {}\n Prediction: {}'.format(
                    dataset.classes[targets[j]], dataset.classes[predicted_names[j]], int_to_type[type_targets[j].item()],
                    int_to_type[predicted_type[j].item()]))
                
                image = images.cpu().data[j].numpy().transpose((1, 2, 0))
                
                # undo our pytorch transform for display
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


```python
display_results(6)
```

# Thank you!
I hope this was an extremely fun and rewarding workshop for you. If you didn't finish, **don't worry**! I highly encourage you to finish it on your own, or meet up with other club members. All of the core coordinators are on discord too, and are happy to help you if you ping them.

Please leave us feedback by filling out this quick form: https://ucfai.org/feedback

Have a good night!
