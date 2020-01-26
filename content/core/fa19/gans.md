---
title: "A look behind DeepFake - GANs"
linktitle: "A look behind DeepFake - GANs"

date: "2019-10-30T17:30:00"
lastmod: "2019-10-30T17:30:00"

draft: false
toc: true
type: docs

weight: 8

menu:
  core_fa19:
    parent: Fall 2019
    weight: 8

authors: ["brandons209", ]

urls:
  youtube: ""
  slides:  ""
  github:  "https://github.com/ucfai/core/blob/master/fa19/2019-10-30-gans/2019-10-30-gans.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-fa19-gans"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/fa19/2019-10-30-gans/2019-10-30-gans.ipynb"

location: "MSB 359"
cover: "https://i1.wp.com/cyxu.tv/wp-content/uploads/2019/03/horse2zebra.jpg?resize=386%2C385"

categories: ["fa19"]
tags: ["GANs", "generative", "adversial", "cyclegan", "deepfake", "cGAN", ]
abstract: >-
  GANs are relativity new in the machine learning world, but they have proven to be a very powerful model. Recently, they made headlines in the DeepFake network, being able to mimic someone else in real time video and audio. There has also been cycleGAN, which takes one domain (horses) and makes it look like something similar (zebras). Come and learn the secret behind these type of networks, you will be suprised how intuitive it is! The lecture will cover the basics of GANs and different types, with the workshop covering how we can generate human faces, cats, dogs, and other cute creatures!
---
```python
# This is a bit of code to make things work on Kaggle
import os
from pathlib import Path

if os.path.exists("/kaggle/input/ucfai-core-fa19-gans"):
    DATA_DIR = Path("/kaggle/input/ucfai-core-fa19-gans")
else:
    DATA_DIR = Path("data/")

!pip install torchsummary
```

# Creating New Celebrities
In this notebook we will be generating unqiue human faces based off of celebrities. Maybe one of them will look like their kid? This dataset contains around 200,000 pictures of celebrities faces, all of them aligned to the center of the image. This is important so the GAN can learned the features of the face properly when generating.

Our network will be a DCGAN since we are working with image data, a popular domain for generating new data with GANs. 

As always, lets import all of our libraries needed, and our helper function from printing Epoch results nicely.

```python
# general imports
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.utils as vutils
from torch.utils.data import random_split

# uncomment to use specific seed for randomly generating weights and noise
# seed = 999
# torch.manual_seed(seed)

```

```python
try:
    import torchsummary
except:
    torchsummary = None

from tabulate import tabulate

BATCH_TEMPLATE = "Epoch [{} / {}], Batch [{} / {}]:"
EPOCH_TEMPLATE = "Epoch [{} / {}]:"
TEST_TEMPLATE = "Epoch [{}] Test:"

def print_iter(curr_epoch=None, epochs=None, batch_i=None, num_batches=None, writer=None, msg=False, **kwargs):
    """
    Formats an iteration. kwargs should be a variable amount of metrics=vals
    Optional Arguments:
        curr_epoch(int): current epoch number (should be in range [0, epochs - 1])
        epochs(int): total number of epochs
        batch_i(int): current batch iteration
        num_batches(int): total number of batches
        writer(SummaryWriter): tensorboardX summary writer object
        msg(bool): if true, doesn't print but returns the message string

    if curr_epoch and epochs is defined, will format end of epoch iteration
    if batch_i and num_batches is also defined, will define a batch iteration
    if curr_epoch is only defined, defines a validation (testing) iteration
    if none of these are defined, defines a single testing iteration
    if writer is not defined, metrics are not saved to tensorboard
    """
    if curr_epoch is not None:
        if batch_i is not None and num_batches is not None and epochs is not None:
            out = BATCH_TEMPLATE.format(curr_epoch + 1, epochs, batch_i, num_batches)
        elif epochs is not None:
            out = EPOCH_TEMPLATE.format(curr_epoch + 1, epochs)
        else:
            out = TEST_TEMPLATE.format(curr_epoch + 1)
    else:
        out = "Testing Results:"

    floatfmt = []
    for metric, val in kwargs.items():
        if "loss" in metric or "recall" in metric or "alarm" in metric or "prec" in metric:
            floatfmt.append(".4f")
        elif "accuracy" in metric or "acc" in metric:
            floatfmt.append(".2f")
        else:
            floatfmt.append(".6f")

        if writer and curr_epoch:
            writer.add_scalar(metric, val, curr_epoch)
        elif writer and batch_i:
            writer.add_scalar(metric, val, batch_i * (curr_epoch + 1))

    out += "\n" + tabulate(kwargs.items(), headers=["Metric", "Value"], tablefmt='github', floatfmt=floatfmt)

    if msg:
        return out
    print(out)

def summary(model, input_dim):
    if torchsummary is None:
        raise(ModuleNotFoundError, "TorchSummary was not found!")
    torchsummary.summary(model, input_dim)
```

### Dataloader
Let's define our dataset and dataloader using Pytorch's Imagefolder. This was used also in our CNN workshop. The built in Imagefolder dataset will load in images from all subfolders of the given path, and pass it through our transforms without the need for creating a custom dataset. For time purposes, we will set a variable called `thanos_level` that will cut our dataset in half, thirds, fourths etc so we can train on a subset of the 200,000 images. For 5 Epochs, the whole dataset will take about a half hour to train, half will be 15 minutes, and a fourth will be about 6 minutes. For best results, use the whole dataset! 

For transforms, we use a resize down to our image size (keeping it small for speed purposes), center crop the image so the face is centered in the image, convert it to a tensor and normalize it with a STD and mean of 0.5. When this is all done, our RGB scalar values will be betweenn -1 and 1, inclusive, the same as what our generator output will be.

Its important to visualize our data before building the model, so lets take a look and plot some images from the dataset.

```python
image_size = (64, 64)
batch_size = 128
num_workers = 4

# I'm sorry little one
thanos_level = 4

dataset = ImageFolder(str(DATA_DIR), transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# comment out if you want to use whole dataset
dataset, _ = random_split(dataset, [int(len(dataset) / thanos_level), len(dataset) - int((len(dataset) / thanos_level))])

# TODO:Create the dataloader from our dataset above
### BEGIN SOLUTION
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
### END SOLUTION

print("Length of dataset: {}, dataloader: {}".format(len(dataset), len(dataloader)))

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
```

### Generator
Here we will define our generator model. I've created a simple function that will give us a block of the network, which includes the Convolution Tranpose (which convolves and upsamples in one layer), a batch normalization, and our activation function ReLU. I've also included the the `get_padding` helper function we used before, which calculators the required padding needed. You can use the function when building the model or just run it below manually and hardcode the padding.

We start with our input size and want to upsample and reduce the number filters until the final layer has 3 channels for RGB, and 64x64, our output size.

It is important as you build the model to keep track of the size of the feature maps as the network gets deeper, as we need to make sure our output size matches the size we set above! If you want larger sizes we can add more layers to the generator. Try doing 128x128 images after the workshop!

```python
def get_padding(output_dim, input_dim, kernel_size, stride):
    """
    Calculates padding given in output and input dim, and parameters of the convolutional layer

    Arguments should all be integers. Use this function to calculate padding for 1 dimesion at a time.
    Output dimensions should be the same or bigger than input dimensions

    Returns 0 if invalid arguments were passed, otherwise returns an int or tuple that represents the padding.
    """

    padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2

    if padding < 0:
        return 0
    else:
        return padding

print(get_padding(32, 64, 4, 2))
```

```python
def gen_block(input_channels, output_channels, kernel_size, stride, padding):
    layers = [nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False)]
    layers += [nn.BatchNorm2d(output_channels)]
    layers += [nn.ReLU(inplace=True)]
    
    return layers
    
class Generator(nn.Module):
    def __init__(self, channels=3, input_size=100, output_dim=64):
        super(Generator, self).__init__()
        self.channels = channels
        self.input_size = input_size
        self.output_size = output_dim
        self.layers = self.build_layers()
        
    def forward(self, x):
        return self.layers(x).squeeze()
    
    def build_layers(self):
        layers = []
        in_c = self.input_size
        out_c = self.output_size * 8
        
        # dim: out_c x 4 x 4
        layers += gen_block(in_c, out_c, 4, 1, 0)
        in_c = out_c
        out_c = self.output_size * 4
        
        # TODO: Create the next two blocks the same way the above one is created
        # Use kernel size of 4 and a stride of 2. Whats the padding?
        ### BEGIN SOLUTION
        # dim: out_c x 8 x 8
        layers += gen_block(in_c, out_c, 4, 2, 1)
        in_c = out_c
        out_c = self.output_size * 2
        
        # dim: out_c x 16 x 16
        layers += gen_block(in_c, out_c, 4, 2, 1)
        in_c = out_c
        out_c = self.output_size
        ### END SOLUTION
        # dim: out_c x 32 x 32
        layers += gen_block(in_c, out_c, 4, 2, 1)
        in_c = out_c
        out_c = self.channels
        
        # dim: out_c x 64 x 64
        # don't use batch norm in the last layer since its the output.
        layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.Tanh()]
        
        return nn.Sequential(*layers)
```

### Discriminator
Now for the discriminator. This will be a simple CNN that we have seen before. The few differences is that we are going to use [LeakyReLU](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7) which is a adapatation to ReLU. LeakyReLU has a chance to "leak" negative values from the function output, instead of zeroing out **all** negative values. This has shown to give better results for the discriminator and help avoid the issues mentioned at the end of the slides.

Instead of max pooling, we use larger strides to halve our input size down until 1 node, which will be our output for the discriminator of either Real or Fake. Pooling in GANs is usually never used as it almost always creates models that don' train. Its better to have a larger stride to reduce size of the feature maps. Since we want the generator to produce images representing the input, it needs context of the whole image, so max pooling would not help here.

Another important note is to not use batch normalization in the first or last block of the discriminator, it can cause the model to not train.

```python
def discrim_block(input_channels, output_channels, kernel_size, stride, padding):
    layers = []
    layers += [nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False)]
    layers += [nn.BatchNorm2d(output_channels)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    
    return layers

class Discriminator(nn.Module):
    def __init__(self, channels=3, input_dim=64):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.input_dim = input_dim
        self.layers = self.build_layers()
        
    def forward(self, x):
        return self.layers(x).squeeze()
    
    def build_layers(self):
        layers = []
        in_c = self.channels
        out_c = self.input_dim
        
        # dim: out_c x 32 x 32
        layers += [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
        in_c = out_c
        out_c = self.input_dim * 2
        # TODO: Create the next 2 blocks for the discriminator. Kernel size of 4 and a stride of 2
        # this is quite similar to the generator...
        ### BEGIN SOLUTION
        # dim: out_c x 16 x 16
        layers += discrim_block(in_c, out_c, 4, 2, 1)
        in_c = out_c
        out_c = self.input_dim * 4
        
        # dim: out_c x 8 x 8
        layers += discrim_block(in_c, out_c, 4, 2, 1)
        in_c = out_c
        out_c = self.input_dim * 8
        ### END SOLUTION
        # dim: out_c x 4 x 4
        layers += discrim_block(in_c, out_c, 4, 2, 1)
        in_c = out_c
        out_c = 1
        
        # dim: 1
        layers += [nn.Conv2d(in_c, out_c, 4, 1, 0), nn.Sigmoid()]
        
        return nn.Sequential(*layers)
        
```

### Define function for initalizing weights
Lets define a function to initalize our weights a certain way, this is following the DCGAN paper and their parameters they used. Convolution weights are randomized from a normal distrubution with a mean of 0 and STD of 0.02, with batch normalization weights randomized from a nnormal distrubution with a mean of 1 and a STD of 0.02.

This is how the DCGAN paper had it, so it should *hopefully* help produce the best results.

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

### Create models, optimizers, and loss function
Now let's create our generator and discriminator models and apply our weight initalization function to them.

We also need to define the optimizer, device, and loss function. From the DCGAN paper, we will use the Adam optimizer with different betas parameters. These betas define how aggresive the optimizer is in reducing the learning rate on a plateau. The GAN suffers if the optimizer is too agreesive, so we reduce this behavior. **We need two optimizers, one for the generator and one for the discriminator.**

Our loss function will be Binary Cross Entropy since we have binary labels.

For purposes of visualizing our model lets define some fixed noise which we will generate examples on each batch iteration, so we can see how the model improves throughout training.

```python
gen_input = 100
gen_output = 64

gen = Generator(input_size=gen_input, output_dim=gen_output)
gen.apply(weights_init)
discrim = Discriminator(channels=3, input_dim=gen_output)
discrim.apply(weights_init)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: {}".format(device))
gen.to(device)
discrim.to(device)

# hyperparameters from DCGAN paper
learn_rate = 0.0002

optG = optim.Adam(gen.parameters(), lr=learn_rate, betas=(0.5, 0.999))
optD = optim.Adam(discrim.parameters(), lr=learn_rate, betas=(0.5, 0.999))

# TODO: Define our criterion (loss function)
### BEGIN SOLUTION
criterion = nn.BCELoss()
### END SOLUTION
fixed_noise = torch.randn(gen_output, gen_input, 1, 1, device=device)

real_label = 1
fake_label = 0

print("Generator:")
summary(gen, (gen_input, 1, 1))
print("\nDiscriminator:")
summary(discrim, (3, gen_output, gen_output))
```

### Train Model
It is time to train. Remember our training loop:
1. Train Discriminator
    1. Feed real images in, calculate loss, and backprop through the discriminator.
    2. Feed fake images in, calculate loss, and backprop through the discriminator.
    3. Sum the losses then update our weights based on both of these losses with our optimizer.
2. Train Generator
    1. Take fake images used to update the discriminator and feed them into the discriminator model again. However, the labels for this will be 1 instead of 0, since the generator's goal is to get the discriminator to predict it's generated images as real. Here the loss is calculated for the generator, *based* on the discriminator's output.
    2. Update weights for generator using the optimizer.
3. Loop **1 and 2** until done training.

```python
start_time = time.time()

epochs = 5
print_step = 50

gen_imgs = []

for e in range(epochs):
    g_train_loss = 0
    d_train_loss = 0
    e_time = time.time()
    
    for i, data in enumerate(dataloader):

        # Train Discriminator
        
        # only need images from data, don't care about class from ImageFolder
        images = data[0].to(device)
        b_size = images.size(0)
        labels = torch.full((b_size,), real_label, device=device)
        
        # train on real
        discrim.zero_grad()
        d_output = discrim(images).view(-1)
        loss_real = criterion(d_output, labels)
        loss_real.backward()
      
        # get fake data from generator
        noise = torch.randn(b_size, gen_input, 1, 1, device=device)
        fake_images = gen(noise)
        # this replaces all values in labels with fake_label, which is zero in this case
        labels.fill_(fake_label)
        
        # calculate loss and update gradients on fake
        # must detach the fake images from the computational graph of the generator, so that gradients arent updated for the generator
        d_output = discrim(fake_images.detach()).view(-1)
        loss_fake = criterion(d_output, labels)
        loss_fake.backward()
        
        # add up real and fake loss
        d_loss = loss_real + loss_fake
        
        # optimize weights after calculating real and fake loss then backprogating on each
        optD.step()
        
        d_train_loss += d_loss.item()
        
        # Train Generator
        gen.zero_grad()
        labels.fill_(real_label)
        # get new output from discriminator for fake images, which is now updated from our above step
        d_output = discrim(fake_images).view(-1)
        # calculate the Generator's loss based on this, use real_labels since fake images should be real for generator
        # i.e the generator wants the discriminator to output real for it's fake images, so thats the target for generator
        g_loss = criterion(d_output, labels)
        g_loss.backward()
        optG.step()
        
        g_train_loss += g_loss.item()
        
        if i % print_step == 0:
            print_iter(curr_epoch=e, epochs=epochs, batch_i=i, num_batches=len(dataloader), d_loss=d_train_loss/(i+1), g_loss=g_train_loss/(i+1))
            # save example images
            gen.eval()
            with torch.no_grad():
                fake_images = gen(fixed_noise).detach().cpu()
                gen.train()
                gen_imgs.append(vutils.make_grid(fake_images, padding=2, normalize=True))
                
    print_iter(curr_epoch=e, epochs=epochs, d_loss=d_train_loss/(i+1), g_loss=g_train_loss/(i+1))
    print("\nEpoch {} took {:.2f} minutes.\n".format(e+1, (time.time() - e_time) / 60))
    
print("Model took {:.2f} minutes to train.".format((time.time() - start_time) / 60))
```

### View Results
This segment of code will create a small animation that goes through the generator's output through training. Notice how the features become more clearer as time goes on. Its able to produce a human face in RGB, amazing!

```python
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in gen_imgs]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
```

### Final Results
This will show the last epoch's results, which hopefully will be our best.

```python
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(gen_imgs[-1],(1,2,0)))
plt.show()
```

# Closing Thoughts
Now that we've built a GAN, the possibilities are endless for what you can apply this too! Getting this model to train is another story though, it'll be lot of playing around and trial/error, but for a very amazing result. I suggest you find datasets of cats or some images that you can try to use this model to train on. You can also try your hand on implementing a cGAN or InfoGAN model, using this as a base. Take the time to explore what you can do and try it out!

For this dataset, try increasing the size of the model to generate larger image sizes, like 128, 128. You would need to add a layer to the generator and discriminator, and probably reduce your batch size and such. You can also try training on the whole dataset for a longer time and see what you get!
