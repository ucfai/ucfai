---
title: "How We Can Give Our Computers Eyes and Ears"
linktitle: "How We Can Give Our Computers Eyes and Ears"
date: "2019-10-16T00:00:00Z"
lastmod: "2019-10-16T00:00:00Z"
draft: false # Is this a draft? true/false
toc: true # Show table of contents? true/false
type: docs # Do not modify.

menu:
  core_fa19:
    parent: Fall 2019
    weight: 6   

weight: 6

authors: ["danielzgsilva", "brandons209"]

urls:
  youtube: "#"
  slides:  "#"
  github:  "#"
  kaggle:  "#"
  colab:   "#"

categories: ["fa19"]
tags: ["Computer Vision", "CNNs", "Convolutional Neural Networks"]
description: >-
  Ever wonder how Facebook tells you which friends to tag in your photos,
  or how Siri can even understand your request? In this meeting we'll dive
  into convolutional neural networks and give you all the tools to build
  smart systems such as these. Join us in learning how we can grant our 
  computers the gifts of hearing and sight!
---

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># This is a bit of code to make things work on Kaggle</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">from</span> <span class="nn">pathlib</span> <span class="kn">import</span> <span class="n">Path</span>

<span class="k">if</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">exists</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-cnns&quot;</span><span class="p">):</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-cnns&quot;</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s2">&quot;We don&#39;t know this machine.&quot;</span><span class="p">)</span>

<span class="c1"># install torch summary</span>
<span class="o">!</span>pip install torchsummary
</pre></div>



# Convolutional Neural Networks and Transfer Learning Workshop
There is a notebook in the github repo for this workshop that has much of the content from the slides in there for your convenience. 

## Set up 

Importing some of the libraries we'll be using, as well as PyTorch:



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># standard imports (Numpy, Pandas, Matplotlib)</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">matplotlib.image</span> <span class="k">as</span> <span class="nn">img</span>
<span class="kn">from</span> <span class="nn">PIL</span> <span class="kn">import</span> <span class="n">Image</span>
<span class="kn">from</span> <span class="nn">PIL</span> <span class="kn">import</span> <span class="n">ImageFile</span>
<span class="n">ImageFile</span><span class="o">.</span><span class="n">LOAD_TRUNCATED_IMAGES</span> <span class="o">=</span> <span class="kc">True</span>

<span class="c1"># PyTorch imports </span>
<span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">torch.nn</span> <span class="k">as</span> <span class="nn">nn</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="k">as</span> <span class="nn">F</span>
<span class="kn">import</span> <span class="nn">torchvision.utils</span>
<span class="kn">from</span> <span class="nn">torch.utils.data</span> <span class="kn">import</span> <span class="n">DataLoader</span><span class="p">,</span> <span class="n">Dataset</span>
<span class="kn">from</span> <span class="nn">torch</span> <span class="kn">import</span> <span class="n">optim</span>
<span class="kn">from</span> <span class="nn">torchvision.datasets</span> <span class="kn">import</span> <span class="n">ImageFolder</span>
<span class="kn">from</span> <span class="nn">torchvision.models</span> <span class="kn">import</span> <span class="n">resnet18</span>
<span class="kn">from</span> <span class="nn">torchvision</span> <span class="kn">import</span> <span class="n">transforms</span>
<span class="kn">from</span> <span class="nn">torchsummary</span> <span class="kn">import</span> <span class="n">summary</span>

<span class="c1"># Extras</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">import</span> <span class="nn">glob</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%</span><span class="k">reload_ext</span> autoreload
<span class="o">%</span><span class="k">autoreload</span> 2
<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="o">%</span><span class="k">pylab</span> inline
<span class="n">random</span><span class="o">.</span><span class="n">seed</span><span class="p">(</span><span class="mi">42</span><span class="p">)</span>
</pre></div>



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



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">input_size</span> <span class="o">=</span> <span class="p">(</span><span class="mi">224</span><span class="p">,</span><span class="mi">224</span><span class="p">)</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">32</span>
<span class="n">num_workers</span> <span class="o">=</span> <span class="mi">4</span>
</pre></div>



This code defines the transformations for each of our datasets (Training, Validation, and Test sets). **Compose()** simply chains together PyTorch transformations. 

The first transformation we apply is the resizing step we discussed above. The next step, **ToTensor()**, transforms the pixel array into a PyTorch **Tensor** and rescales each pixel value to be between 0 and 1. This is required for an input to be consumed by PyTorch. Finally, we normalize each Tensor to have a mean of 0 and variance of 1. Research supports that Neural Networks tend to perform much better on normalized data... 




<div class=" highlight hl-ipython3"><pre><span></span><span class="n">data_transforms</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;Train&#39;</span><span class="p">:</span> <span class="n">transforms</span><span class="o">.</span><span class="n">Compose</span><span class="p">([</span><span class="n">transforms</span><span class="o">.</span><span class="n">Resize</span><span class="p">(</span><span class="n">input_size</span><span class="p">),</span>
                          <span class="n">transforms</span><span class="o">.</span><span class="n">ToTensor</span><span class="p">(),</span>
                          <span class="n">transforms</span><span class="o">.</span><span class="n">Normalize</span><span class="p">(</span><span class="n">mean</span><span class="o">=</span><span class="p">[</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">],</span>
                                 <span class="n">std</span><span class="o">=</span><span class="p">[</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">])</span>
    <span class="p">]),</span>
    <span class="s1">&#39;Validation&#39;</span><span class="p">:</span> <span class="n">transforms</span><span class="o">.</span><span class="n">Compose</span><span class="p">([</span><span class="n">transforms</span><span class="o">.</span><span class="n">Resize</span><span class="p">(</span><span class="n">input_size</span><span class="p">),</span>
                          <span class="n">transforms</span><span class="o">.</span><span class="n">ToTensor</span><span class="p">(),</span>
                          <span class="n">transforms</span><span class="o">.</span><span class="n">Normalize</span><span class="p">(</span><span class="n">mean</span><span class="o">=</span><span class="p">[</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">],</span>
                                 <span class="n">std</span><span class="o">=</span><span class="p">[</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">])</span>
    <span class="p">]),</span>
    <span class="s1">&#39;Test&#39;</span><span class="p">:</span> <span class="n">transforms</span><span class="o">.</span><span class="n">Compose</span><span class="p">([</span><span class="n">transforms</span><span class="o">.</span><span class="n">Resize</span><span class="p">(</span><span class="n">input_size</span><span class="p">),</span>
                               <span class="n">transforms</span><span class="o">.</span><span class="n">ToTensor</span><span class="p">(),</span>
                               <span class="n">transforms</span><span class="o">.</span><span class="n">Normalize</span><span class="p">(</span><span class="n">mean</span><span class="o">=</span><span class="p">[</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">],</span>
                                 <span class="n">std</span><span class="o">=</span><span class="p">[</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">])</span>
    <span class="p">])</span>
<span class="p">}</span>
</pre></div>



#### PyTorch datasets and dataloaders

Our next step is to create PyTorch **Datasets** for each of our training, validation, and test sets. **torch.utils.data.Dataset** is an abstract class that represents a dataset and has several handy attributes we'll utilize from here on out.

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



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">image_datasets</span> <span class="o">=</span> <span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">ImageFolder</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">DATA_DIR</span><span class="p">,</span> <span class="n">x</span><span class="p">),</span><span class="n">data_transforms</span><span class="p">[</span><span class="n">x</span><span class="p">])</span>
                  <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">,</span> <span class="s1">&#39;Validation&#39;</span><span class="p">]}</span>

<span class="c1"># dataset class to load images with no labels, for our testing set to submit to the competition</span>
<span class="k">class</span> <span class="nc">ImageLoader</span><span class="p">(</span><span class="n">Dataset</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">root</span><span class="p">,</span> <span class="n">transform</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
        <span class="c1"># get image file paths</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">images</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">glob</span><span class="o">.</span><span class="n">glob</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">root</span><span class="p">,</span> <span class="s2">&quot;*&quot;</span><span class="p">)),</span> <span class="n">key</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">glob_format</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">transform</span> <span class="o">=</span> <span class="n">transform</span>
        
    <span class="k">def</span> <span class="fm">__len__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">images</span><span class="p">)</span>
    
    <span class="k">def</span> <span class="fm">__getitem__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">idx</span><span class="p">):</span>
        <span class="n">img</span> <span class="o">=</span> <span class="n">Image</span><span class="o">.</span><span class="n">open</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">images</span><span class="p">[</span><span class="n">idx</span><span class="p">])</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">transform</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="n">img</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">img</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">img</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="k">return</span> <span class="n">transforms</span><span class="o">.</span><span class="n">ToTensor</span><span class="p">(</span><span class="n">img</span><span class="p">)</span>
        
    <span class="nd">@staticmethod</span>
    <span class="k">def</span> <span class="nf">glob_format</span><span class="p">(</span><span class="n">key</span><span class="p">):</span>     
        <span class="n">key</span> <span class="o">=</span> <span class="n">key</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">&quot;/&quot;</span><span class="p">)[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">&quot;.&quot;</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>     
        <span class="k">return</span> <span class="s2">&quot;</span><span class="si">{:04d}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">int</span><span class="p">(</span><span class="n">key</span><span class="p">))</span>
    
<span class="n">image_datasets</span><span class="p">[</span><span class="s1">&#39;Test&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">ImageLoader</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">DATA_DIR</span> <span class="o">/</span> <span class="s2">&quot;Test&quot;</span><span class="p">),</span> <span class="n">transform</span><span class="o">=</span><span class="n">data_transforms</span><span class="p">[</span><span class="s2">&quot;Test&quot;</span><span class="p">])</span>
</pre></div>



The pixel array of each image is actually quite large, so it'd be inefficient to load the entire dataset onto your RAM at once. Instead, we use PyTorch DataLoaders to load up batches of images on the fly. Earlier we defined a batch size of 32, so in each iteration the loaders will load 32 images and apply our transformations, before returning them to us.

For the most part, Neural Networks are trained on **batches** of data so these DataLoaders greatly simplify the process of loading and feeding data to our network. The rank 4 tensor returned by the dataloader is of size (32, 224, 224, 3).



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataloaders</span> <span class="o">=</span> <span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">image_datasets</span><span class="p">[</span><span class="n">x</span><span class="p">],</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">num_workers</span> <span class="o">=</span> <span class="n">num_workers</span><span class="p">)</span>
              <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">,</span> <span class="s1">&#39;Validation&#39;</span><span class="p">]}</span>

<span class="n">test_loader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">dataset</span> <span class="o">=</span> <span class="n">image_datasets</span><span class="p">[</span><span class="s1">&#39;Test&#39;</span><span class="p">],</span> <span class="n">batch_size</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>



Every PyTorch dataset has an attribute,  **classes**, which is an array containing all of the image classes in the dataset. In our case, breeds of dog in the dataset. 



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dog_breeds</span> <span class="o">=</span> <span class="n">image_datasets</span><span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">classes</span>
<span class="nb">print</span><span class="p">(</span><span class="n">dog_breeds</span><span class="p">)</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Just printing the number of images in each dataset we created</span>

<span class="n">dataset_sizes</span> <span class="o">=</span> <span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="nb">len</span><span class="p">(</span><span class="n">image_datasets</span><span class="p">[</span><span class="n">x</span><span class="p">])</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">,</span> <span class="s1">&#39;Validation&#39;</span><span class="p">,</span> <span class="s1">&#39;Test&#39;</span><span class="p">]}</span>

<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Train Length: </span><span class="si">{}</span><span class="s1"> | Valid Length: </span><span class="si">{}</span><span class="s1"> | Test Length: </span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">dataset_sizes</span><span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">],</span> 
                                                                     <span class="n">dataset_sizes</span><span class="p">[</span><span class="s1">&#39;Validation&#39;</span><span class="p">],</span> <span class="n">dataset_sizes</span><span class="p">[</span><span class="s1">&#39;Test&#39;</span><span class="p">]))</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Here we&#39;re defining what component we&#39;ll use to train this model</span>
<span class="c1"># We want to use the GPU if available, if not we use the CPU</span>

<span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">device</span><span class="p">(</span><span class="s2">&quot;cuda:0&quot;</span> <span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span> <span class="k">else</span> <span class="s2">&quot;cpu&quot;</span><span class="p">)</span>
<span class="n">device</span>
</pre></div>



#### Visualizing the dataset

Once we've set up our PyTorch datasets and dataloaders, grabbing individual images or batches of images is super simple. Below I've defined 2 functions we can use to take a look at the dogs in our dataset.

The first one here indexes into our training set, grabs a given number of random images, and plots them. A PyTorch dataset is *sort of* a 2d array, where the first dimension represents the images themselves, and the second dimension contains the pixel array and the label of the image.

The second function allows us to plot a batch of images served up by our PyTorch dataloader.



<div class=" highlight hl-ipython3"><pre><span></span>  <span class="c1"># Plots a given number of images from a PyTorch Data</span>
<span class="k">def</span> <span class="nf">show_random_imgs</span><span class="p">(</span><span class="n">num_imgs</span><span class="p">):</span>
  
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">num_imgs</span><span class="p">):</span>
        <span class="c1"># We&#39;re plotting images from the training set</span>
        <span class="n">train_dataset</span> <span class="o">=</span> <span class="n">image_datasets</span><span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">]</span>
        
        <span class="c1"># Choose a random image</span>
        <span class="n">rand</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">train_dataset</span><span class="p">)</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
        
        <span class="c1"># Read in the image</span>
        <span class="n">ex</span> <span class="o">=</span> <span class="n">img</span><span class="o">.</span><span class="n">imread</span><span class="p">(</span><span class="n">train_dataset</span><span class="o">.</span><span class="n">imgs</span><span class="p">[</span><span class="n">rand</span><span class="p">][</span><span class="mi">0</span><span class="p">])</span>
        
        <span class="c1"># Get the image&#39;s label</span>
        <span class="n">breed</span> <span class="o">=</span> <span class="n">dog_breeds</span><span class="p">[</span><span class="n">train_dataset</span><span class="o">.</span><span class="n">imgs</span><span class="p">[</span><span class="n">rand</span><span class="p">][</span><span class="mi">1</span><span class="p">]]</span>
        
        <span class="c1"># Show the image and print out the image&#39;s size (really the shape of it&#39;s array of pixels)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">ex</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Image Shape: &#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">ex</span><span class="o">.</span><span class="n">shape</span><span class="p">))</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="n">breed</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
       
  <span class="c1"># Plots a batch of images served up by PyTorch    </span>
<span class="k">def</span> <span class="nf">show_batch</span><span class="p">(</span><span class="n">batch</span><span class="p">):</span>
  
    <span class="c1"># Undo the transformations applied to the images when loading a batch</span>
    <span class="n">batch</span> <span class="o">=</span> <span class="n">batch</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span><span class="o">.</span><span class="n">transpose</span><span class="p">((</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">))</span>
    <span class="n">mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">])</span>
    <span class="n">std</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">])</span>
    <span class="n">batch</span> <span class="o">=</span> <span class="n">std</span> <span class="o">*</span> <span class="n">batch</span> <span class="o">+</span> <span class="n">mean</span>
    <span class="n">batch</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">clip</span><span class="p">(</span><span class="n">batch</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
    
    <span class="c1"># Plot the batch</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">batch</span><span class="p">)</span>
    
    <span class="c1"># pause a bit so that plots are updated</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">pause</span><span class="p">(</span><span class="mf">0.001</span><span class="p">)</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="n">show_random_imgs</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Get a batch of training data (32 random images)</span>
<span class="n">imgs</span><span class="p">,</span> <span class="n">classes</span> <span class="o">=</span> <span class="nb">next</span><span class="p">(</span><span class="nb">iter</span><span class="p">(</span><span class="n">dataloaders</span><span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">]))</span>

<span class="c1"># This PyTorch function makes a grid of images from a batch for us</span>
<span class="n">batch</span> <span class="o">=</span> <span class="n">torchvision</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">make_grid</span><span class="p">(</span><span class="n">imgs</span><span class="p">)</span>

<span class="n">show_batch</span><span class="p">(</span><span class="n">batch</span><span class="p">)</span>
</pre></div>



#### Defining a network in PyTorch

Now its time to finally build our CNN.  In PyTorch, a model is represented by a normal Python class that inherits from the master nn.Module class. Inheriting from this master class grants your model all the methods and attributes needed to train and work with your model. There are, however, 2 things you need to write yourself:
 -  **__init__(self)**: Here is where you define the layers and overall architecture of your model
 -  **forward(self, x)**: This method takes an input, x, computes a forward pass through the network and outputs predictions. Writing it essentially involves connecting your layers and setting up the flow of the input through your layers.
 
 

Below are the signatures of the PyTorch functions that create each of the layers we discussed. Try to use them to build your first CNN! I provided some comments that hopefully guide you in terms of what should happen at each step.

-  nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
-  nn.ReLU(x)
-  nn.MaxPool2d(kernel_size, stride, padding)
-  nn.BatchNorm2d(num_features) - num_features is the number of channels it receives
-  nn.Dropout(p) - p is probability of an element to be zeroed
-  nn.Linear(in_features, out_features) – fully connected layer (matrix multiplications used in the classification portion of a network)



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># It is good practice to maintain input dimensions as the image is passed through convolution layers</span>
<span class="c1"># With a default stride of 1, and no padding, a convolution will reduce image dimenions to:</span>
            <span class="c1"># out = in - m + 1, where m is the size of the kernel and in is a dimension of the input</span>

<span class="c1"># Use this function to calculate the padding size neccessary to create an output of desired dimensions</span>

<span class="k">def</span> <span class="nf">get_padding</span><span class="p">(</span><span class="n">input_dim</span><span class="p">,</span> <span class="n">output_dim</span><span class="p">,</span> <span class="n">kernel_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">):</span>
  <span class="c1"># Calculates padding necessary to create a certain output size,</span>
  <span class="c1"># given a input size, kernel size and stride</span>
  
  <span class="n">padding</span> <span class="o">=</span> <span class="p">(((</span><span class="n">output_dim</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span> <span class="o">*</span> <span class="n">stride</span><span class="p">)</span> <span class="o">-</span> <span class="n">input_dim</span> <span class="o">+</span> <span class="n">kernel_size</span><span class="p">)</span> <span class="o">//</span> <span class="mi">2</span>
  
  <span class="k">if</span> <span class="n">padding</span> <span class="o">&lt;</span> <span class="mi">0</span><span class="p">:</span>
    <span class="k">return</span> <span class="mi">0</span>
  <span class="k">else</span><span class="p">:</span>
    <span class="k">return</span> <span class="n">padding</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Make sure you calculate the padding amount needed to maintain the spatial size of the input</span>
<span class="c1"># after each Conv layer</span>

<span class="k">class</span> <span class="nc">CNN</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
  
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">CNN</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        
        <span class="c1"># nn.Sequential() is simply a container that groups layers into one object</span>
        <span class="c1"># Pass layers into it separated by commas</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">block1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span>
            
            <span class="c1"># The first convolutional layer. Think about how many channels the input starts off with</span>
            <span class="c1"># Let&#39;s have this first layer extract 32 features</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span>
            <span class="c1">### END SOLUTION</span>
            
            <span class="c1"># Don&#39;t forget to apply a non-linearity</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">())</span>
            <span class="c1">### END SOLUTION</span>
        
        <span class="bp">self</span><span class="o">.</span><span class="n">block2</span> <span class="o">=</span>  <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span>
            
            <span class="c1"># The second convolutional layer. How many channels does it receive, given the number of features extracted by the first layer?</span>
            <span class="c1"># Have this layer extract 64 features</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="mi">32</span><span class="p">,</span> <span class="mi">64</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span>
            <span class="c1">### END SOLUTION</span>
            
            <span class="c1"># Non linearity</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">(),</span>
            <span class="c1">### END SOLUTION</span>
            
            <span class="c1"># Lets introduce a Batch Normalization layer</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">BatchNorm2d</span><span class="p">(</span><span class="mi">64</span><span class="p">),</span>
            <span class="c1">### END SOLUTION</span>
            
            <span class="c1"># Downsample the input with Max Pooling</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">MaxPool2d</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
            <span class="c1">### END SOLUTION</span>
        <span class="p">)</span>
        
        <span class="c1"># Mimic the second block here, except have this block extract 128 features</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">block3</span> <span class="o">=</span>  <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="mi">128</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">(),</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">BatchNorm2d</span><span class="p">(</span><span class="mi">128</span><span class="p">),</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">MaxPool2d</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
            <span class="c1">### END SOLUTION</span>
        <span class="p">)</span>
        
        <span class="c1"># Applying a global pooling layer</span>
        <span class="c1"># Turns the 128 channel rank 4 tensor into a rank 2 tensor of size 32 x 128 (32 128-length arrays, one for each of the inputs in a batch)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">global_pool</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">AdaptiveAvgPool2d</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
        
        <span class="c1"># Fully connected layer</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="mi">512</span><span class="p">)</span>
        
        <span class="c1"># Introduce dropout to reduce overfitting</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">drop_out</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Dropout</span><span class="p">(</span><span class="mf">0.5</span><span class="p">)</span>
        
        <span class="c1"># Final fully connected layer creates the prediction array</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">512</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">dog_breeds</span><span class="p">))</span>
    
    <span class="c1"># Feed the input through each of the layers we defined </span>
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        
        <span class="c1"># Input size changes from (32 x 3 x 224 x 224) to (32 x 32 x 224 x 224)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">block1</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        
        <span class="c1"># Size changes from (32 x 32 x 224 x 224) to (32 x 64 x 112 x 112) after max pooling</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">block2</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        
        <span class="c1"># Size changes from (32 x 64 x 112 x 112) to (32 x 128 x 56 x 56) after max pooling</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">block3</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        
        <span class="c1"># Reshapes the input from (32 x 128 x 56 x 56) to (32 x 128)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">global_pool</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">),</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
        
        <span class="c1"># Fully connected layer, size changes from (32 x 128) to (32 x 512)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">fc1</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">drop_out</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        
        <span class="c1"># Size change from (32 x 512) to (32 x 133) to create prediction arrays for each of the images in the batch</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">fc2</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        
        <span class="k">return</span> <span class="n">x</span>
</pre></div>



Now we create an instance of this CNN() class and define the loss function and optimizer we'll use to train our model. In our case we'll use CrossEntropyLoss. You'll notice we never added a Softmax activation after our last layer. That's because PyTorch's CrossEntropyLoss applies a softmax before calculating log loss, a commonly used loss function for single label classification problems.

For the optimizer we'll use Adam, an easy to apply but powerful optimizer which is an extension of the popular Stochastic Gradient Descent method. We need to pass it all of the parameters it'll train, which PyTorch makes easy with model.parameters(), and also the learning rate we'll use.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">CNN</span><span class="p">()</span>
<span class="n">criterion</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">CrossEntropyLoss</span><span class="p">()</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span> <span class="o">=</span> <span class="mf">0.0001</span><span class="p">)</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">model</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="n">summary</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="p">(</span><span class="mi">3</span><span class="p">,</span><span class="mi">224</span><span class="p">,</span><span class="mi">224</span><span class="p">))</span>
</pre></div>



## Training a model in PyTorch 

At this point we're finally ready to train our model! In PyTorch we have to write our own training loops before getting to actually train the model. This can seem daunting at first, so let's break up each stage of the training process. 

The bulk of the function is handled by a nested for loop, the outer looping through each epoch and the inner looping through all of the batches of images in our dataset. Each epoch has a training and validation phase, where batches are served from their respective loaders. Both phases begin by feeding a batch of inputs into the model, which implicity calls the forward() function on the input. Then we calculate the loss of the outputs against the true labels of the batch. 

If we're in training mode, here is where we perform back-propagation and adjust our weights. To do this, we first zero the gradients, then perform backpropagation by calling .backward() on the loss variable. Finally, we call optimizer.step() to adjust the weights of the model in accordance with the calculated gradients.

The remaining portion of one epoch is the same for both training and validation, and simply involves calculating and tracking the accuracy achieved in both phases. A nifty addition to this training loop is that it tracks the highest validation accuracy and only saves weights which beat that accuracy, ensuring that the best performing weights are returned from the function.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">run_epoch</span><span class="p">(</span><span class="n">epoch</span><span class="p">,</span> <span class="n">model</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">phase</span><span class="p">):</span>
  
    <span class="n">running_loss</span> <span class="o">=</span> <span class="mf">0.0</span>
    <span class="n">running_corrects</span> <span class="o">=</span> <span class="mi">0</span>

    <span class="k">if</span> <span class="n">phase</span> <span class="o">==</span> <span class="s1">&#39;Train&#39;</span><span class="p">:</span>
        <span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">model</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>

    <span class="c1"># Looping through batches</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">inputs</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">dataloaders</span><span class="p">[</span><span class="n">phase</span><span class="p">]):</span>
    
        <span class="c1"># ensures we&#39;re doing this calculation on our GPU if possible</span>
        <span class="n">inputs</span> <span class="o">=</span> <span class="n">inputs</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
        <span class="n">labels</span> <span class="o">=</span> <span class="n">labels</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

        <span class="c1"># Zero parameter gradients</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
    
        <span class="c1"># Calculate gradients only if we&#39;re in the training phase</span>
        <span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">set_grad_enabled</span><span class="p">(</span><span class="n">phase</span> <span class="o">==</span> <span class="s1">&#39;Train&#39;</span><span class="p">):</span>
      
            <span class="c1"># This calls the forward() function on a batch of inputs</span>
            <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">inputs</span><span class="p">)</span>

            <span class="c1"># Calculate the loss of the batch</span>
            <span class="n">loss</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span>

            <span class="c1"># Gets the predictions of the inputs (highest value in the array)</span>
            <span class="n">_</span><span class="p">,</span> <span class="n">preds</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>

            <span class="c1"># Adjust weights through backpropagation if we&#39;re in training phase</span>
            <span class="k">if</span> <span class="n">phase</span> <span class="o">==</span> <span class="s1">&#39;Train&#39;</span><span class="p">:</span>
                <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
                <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>

        <span class="c1"># Document statistics for the batch</span>
        <span class="n">running_loss</span> <span class="o">+=</span> <span class="n">loss</span><span class="o">.</span><span class="n">item</span><span class="p">()</span> <span class="o">*</span> <span class="n">inputs</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
        <span class="n">running_corrects</span> <span class="o">+=</span> <span class="n">torch</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">preds</span> <span class="o">==</span> <span class="n">labels</span><span class="o">.</span><span class="n">data</span><span class="p">)</span>
    
    <span class="c1"># Calculate epoch statistics</span>
    <span class="n">epoch_loss</span> <span class="o">=</span> <span class="n">running_loss</span> <span class="o">/</span> <span class="n">image_datasets</span><span class="p">[</span><span class="n">phase</span><span class="p">]</span><span class="o">.</span><span class="fm">__len__</span><span class="p">()</span>
    <span class="n">epoch_acc</span> <span class="o">=</span> <span class="n">running_corrects</span><span class="o">.</span><span class="n">double</span><span class="p">()</span> <span class="o">/</span> <span class="n">image_datasets</span><span class="p">[</span><span class="n">phase</span><span class="p">]</span><span class="o">.</span><span class="fm">__len__</span><span class="p">()</span>

    <span class="k">return</span> <span class="n">epoch_loss</span><span class="p">,</span> <span class="n">epoch_acc</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">train</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">criterion</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">num_epochs</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">):</span>
    <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>

    <span class="n">best_model_wts</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">state_dict</span><span class="p">()</span>
    <span class="n">best_acc</span> <span class="o">=</span> <span class="mf">0.0</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;| Epoch</span><span class="se">\t</span><span class="s1"> | Train Loss</span><span class="se">\t</span><span class="s1">| Train Acc</span><span class="se">\t</span><span class="s1">| Valid Loss</span><span class="se">\t</span><span class="s1">| Valid Acc</span><span class="se">\t</span><span class="s1">| Epoch Time |&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;-&#39;</span> <span class="o">*</span> <span class="mi">86</span><span class="p">)</span>
    
    <span class="c1"># Iterate through epochs</span>
    <span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">num_epochs</span><span class="p">):</span>
        
        <span class="n">epoch_start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>
       
        <span class="c1"># Training phase</span>
        <span class="n">train_loss</span><span class="p">,</span> <span class="n">train_acc</span> <span class="o">=</span> <span class="n">run_epoch</span><span class="p">(</span><span class="n">epoch</span><span class="p">,</span> <span class="n">model</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="s1">&#39;Train&#39;</span><span class="p">)</span>
        
        <span class="c1"># Validation phase</span>
        <span class="n">val_loss</span><span class="p">,</span> <span class="n">val_acc</span> <span class="o">=</span> <span class="n">run_epoch</span><span class="p">(</span><span class="n">epoch</span><span class="p">,</span> <span class="n">model</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="s1">&#39;Validation&#39;</span><span class="p">)</span>
        
        <span class="n">epoch_time</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">epoch_start</span>
           
        <span class="c1"># Print statistics after the validation phase</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;| </span><span class="si">{}</span><span class="se">\t</span><span class="s2"> | </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">| </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">| </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">| </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">| </span><span class="si">{:.0f}</span><span class="s2">m </span><span class="si">{:.0f}</span><span class="s2">s     |&quot;</span>
                      <span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">train_loss</span><span class="p">,</span> <span class="n">train_acc</span><span class="p">,</span> <span class="n">val_loss</span><span class="p">,</span> <span class="n">val_acc</span><span class="p">,</span> <span class="n">epoch_time</span> <span class="o">//</span> <span class="mi">60</span><span class="p">,</span> <span class="n">epoch_time</span> <span class="o">%</span> <span class="mi">60</span><span class="p">))</span>

        <span class="c1"># Copy and save the model&#39;s weights if it has the best accuracy thus far</span>
        <span class="k">if</span> <span class="n">val_acc</span> <span class="o">&gt;</span> <span class="n">best_acc</span><span class="p">:</span>
            <span class="n">best_acc</span> <span class="o">=</span> <span class="n">val_acc</span>
            <span class="n">best_model_wts</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">state_dict</span><span class="p">()</span>

    <span class="n">total_time</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">start</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;-&#39;</span> <span class="o">*</span> <span class="mi">74</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Training complete in </span><span class="si">{:.0f}</span><span class="s1">m </span><span class="si">{:.0f}</span><span class="s1">s&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">total_time</span> <span class="o">//</span> <span class="mi">60</span><span class="p">,</span> <span class="n">total_time</span> <span class="o">%</span> <span class="mi">60</span><span class="p">))</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Best validation accuracy: </span><span class="si">{:.4f}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">best_acc</span><span class="p">))</span>

    <span class="c1"># load best model weights and return them</span>
    <span class="n">model</span><span class="o">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">best_model_wts</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">model</span>
</pre></div>



#### Testing a model

Creating a function that generates and prints predictions on a given number of images from our test set:



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">test_model</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">num_images</span><span class="p">):</span>
    <span class="n">was_training</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">training</span>
    <span class="n">model</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>
    <span class="n">images_so_far</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">num_images</span><span class="p">,</span> <span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">10</span><span class="p">))</span>

    <span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
        <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">images</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">dataloaders</span><span class="p">[</span><span class="s1">&#39;Validation&#39;</span><span class="p">]):</span>
            <span class="n">images</span> <span class="o">=</span> <span class="n">images</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
            <span class="n">labels</span> <span class="o">=</span> <span class="n">labels</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

            <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">images</span><span class="p">)</span>
            <span class="n">_</span><span class="p">,</span> <span class="n">preds</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>

            <span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">images</span><span class="o">.</span><span class="n">size</span><span class="p">()[</span><span class="mi">0</span><span class="p">]):</span>
                <span class="n">images_so_far</span> <span class="o">+=</span> <span class="mi">1</span>
                <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplot</span><span class="p">(</span><span class="n">num_images</span><span class="o">//</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">images_so_far</span><span class="p">)</span>
                <span class="n">ax</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>
                <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Actual: </span><span class="si">{}</span><span class="s1"> </span><span class="se">\n</span><span class="s1"> Prediction: </span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">dog_breeds</span><span class="p">[</span><span class="n">labels</span><span class="p">[</span><span class="n">j</span><span class="p">]],</span> <span class="n">dog_breeds</span><span class="p">[</span><span class="n">preds</span><span class="p">[</span><span class="n">j</span><span class="p">]]))</span>
                
                <span class="n">image</span> <span class="o">=</span> <span class="n">images</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">data</span><span class="p">[</span><span class="n">j</span><span class="p">]</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span><span class="o">.</span><span class="n">transpose</span><span class="p">((</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">))</span>
                
                <span class="n">mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">])</span>
                <span class="n">std</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">])</span>
                <span class="n">image</span> <span class="o">=</span> <span class="n">std</span> <span class="o">*</span> <span class="n">image</span> <span class="o">+</span> <span class="n">mean</span>
                <span class="n">image</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">clip</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
                
                <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>
                <span class="k">if</span> <span class="n">images_so_far</span> <span class="o">==</span> <span class="n">num_images</span><span class="p">:</span>
                    <span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">(</span><span class="n">mode</span><span class="o">=</span><span class="n">was_training</span><span class="p">)</span>
                    <span class="k">return</span>
        <span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">(</span><span class="n">mode</span><span class="o">=</span><span class="n">was_training</span><span class="p">)</span>
</pre></div>



After defining these functions, training and testing our model is straightforward from here on out. Simply call the train() function with the required parameters and let your GPU go to work!



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Make sure to comment this out when you go to &quot;Commit&quot; the kaggle notebook!</span>
<span class="c1"># otherwise, it&#39;ll run this model along with your other models down below.</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">train</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">criterion</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">epochs</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>
</pre></div>



Ouch! Our model doesn't seem to be performing very well at all. After 20 epochs of training we're barely able to achieve a 10% accuracy on our validation set... Hang in there, in a bit I'll go into some methods we can use to achieve a much better accuracy.

In the meantime, let's quickly take a look at how we can save our PyTorch models. Then we'll test and visualize our model. 

## Saving a model in PyTorch 

There are many ways to save a PyTorch model, however the most robust method is described below. This allows you to load up a model for both testing and further training.

The most important part to understand from the code below is what the model and optimizer **state_dict's** are. The model state_dict is essentially a dictionary which contains all of the learned weights and biases in the model, while the optimizer contains information about the optimizer’s state hyperparameters used.

Other than the state_dicts, we also save the class used to build the model architecture, as well as the optimizer and loss function. Putting all of this together allows us to save, move around, and later restore our model to it's exact state after training.. A **.tar** file extension is commonly used to bundle all of this together.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">torch</span><span class="o">.</span><span class="n">save</span><span class="p">({</span>
            <span class="s1">&#39;model&#39;</span> <span class="p">:</span> <span class="n">CNN</span><span class="p">(),</span>
            <span class="s1">&#39;epoch&#39;</span> <span class="p">:</span> <span class="n">epochs</span><span class="p">,</span>
            <span class="s1">&#39;model_state_dict&#39;</span><span class="p">:</span> <span class="n">model</span><span class="o">.</span><span class="n">state_dict</span><span class="p">(),</span>
            <span class="s1">&#39;optimizer&#39;</span> <span class="p">:</span> <span class="n">optimizer</span><span class="p">,</span>
            <span class="s1">&#39;optimizer_state_dict&#39;</span><span class="p">:</span> <span class="n">optimizer</span><span class="o">.</span><span class="n">state_dict</span><span class="p">(),</span>
            <span class="s1">&#39;criterion&#39;</span> <span class="p">:</span> <span class="n">criterion</span><span class="p">,</span>
            <span class="s1">&#39;device&#39;</span> <span class="p">:</span> <span class="n">device</span>
            <span class="p">},</span> <span class="s1">&#39;base_model.tar&#39;</span><span class="p">)</span>
</pre></div>



Creating a function which unpacks the .tar file we saved earlier and loads up the model's saved weights and optimizer state:



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">load_checkpoint</span><span class="p">(</span><span class="n">filepath</span><span class="p">):</span>
    <span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">device</span><span class="p">(</span><span class="s2">&quot;cuda:0&quot;</span> <span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span> <span class="k">else</span> <span class="s2">&quot;cpu&quot;</span><span class="p">)</span>
    <span class="n">checkpoint</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">filepath</span><span class="p">,</span> <span class="n">map_location</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
    <span class="n">model</span> <span class="o">=</span> <span class="n">checkpoint</span><span class="p">[</span><span class="s1">&#39;model&#39;</span><span class="p">]</span>
    <span class="n">model</span><span class="o">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">checkpoint</span><span class="p">[</span><span class="s1">&#39;model_state_dict&#39;</span><span class="p">])</span>
    <span class="n">optimizer</span> <span class="o">=</span> <span class="n">checkpoint</span><span class="p">[</span><span class="s1">&#39;optimizer&#39;</span><span class="p">]</span>
    <span class="n">optimizer</span> <span class="o">=</span> <span class="n">optimizer</span><span class="o">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">checkpoint</span><span class="p">[</span><span class="s1">&#39;optimizer_state_dict&#39;</span><span class="p">])</span>
    <span class="n">criterion</span> <span class="o">=</span> <span class="n">checkpoint</span><span class="p">[</span><span class="s1">&#39;criterion&#39;</span><span class="p">]</span>
    <span class="n">epoch</span> <span class="o">=</span> <span class="n">checkpoint</span><span class="p">[</span><span class="s1">&#39;epoch&#39;</span><span class="p">]</span>
    <span class="n">model</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">model</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">criterion</span><span class="p">,</span> <span class="n">epoch</span>
</pre></div>



Loading our model up...



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">criterion</span><span class="p">,</span> <span class="n">epoch</span> <span class="o">=</span> <span class="n">load_checkpoint</span><span class="p">(</span><span class="s1">&#39;base_model.tar&#39;</span><span class="p">)</span>
</pre></div>



Let's test our model on a couple of dogs!



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">test_model</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="mi">6</span><span class="p">)</span>
</pre></div>



As expected, our model is predicting the wrong breed for the majority of test images. Why is this?

In short, building and training a CNN from scratch is possible, however most problems require significantly more complex models, trained on huge amounts of data. Of course, the computational power and amount of data needed to train these networks accurately are not always available. This is why the idea of **Transfer Learning** has become so popular. It allows everyday people, like me and you, to build accurate and powerful models with limited resources.

## Transfer Learning  

In transfer learning, we take the architecture and weights of a pre-trained model 
(one that has been trained on millions of images belonging to 1000’s of classes, on several high power GPU’s for several days) 
and use the pre-learned features to solve our own novel problem.

PyTorch actually comes with a number of models which have already been trained on the Imagenet dataset we discussed earlier, making it quite simple for us to apply this method of transfer learning. We'll be using a powerful but lighweight model called ResNet18, which we import like so:
-  from torchvision.models import resnet18

The next block of code might look a bit foreign. What we're doing is actually looping through all of the model's pretrained weights and **freezing** them. This means that during training, these weights will not be updating at all. We then take the entire ResNet model and put it into one block of our model, named feature_extraction. It's important to understand that when you load a pretrained model you are only receiving the feature extraction block, or the convolutional layers. It's up to us to define a classification block which can take all of the features the ResNet model extracted and use them to actually classify an image.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">class</span> <span class="nc">PreTrained_Resnet</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">PreTrained_Resnet</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        
        <span class="c1"># Loading up a pretrained ResNet18 model</span>
        <span class="n">resnet</span> <span class="o">=</span> <span class="n">resnet18</span><span class="p">(</span><span class="n">pretrained</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
        
        <span class="c1"># Freeze the entire pretrained network</span>
        <span class="k">for</span> <span class="n">layer</span> <span class="ow">in</span> <span class="n">resnet</span><span class="o">.</span><span class="n">parameters</span><span class="p">():</span>
            <span class="n">layer</span><span class="o">.</span><span class="n">requires_grad</span> <span class="o">=</span> <span class="kc">False</span>
            
        <span class="bp">self</span><span class="o">.</span><span class="n">feature_extraction</span> <span class="o">=</span> <span class="n">resnet</span>
        
        <span class="c1"># Write the classifier block for this network      </span>
            <span class="c1"># Tip: ResNet18&#39;s feature extraction portion ends up with 1000 feature maps, and then implements a Global Average Pooling layer</span>
            <span class="c1"># So what would the size and dimension of the output tensor be?</span>
            <span class="c1"># Think about how can we take that output tensor and transform it into an array of dog breed predictions...</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">classifier</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span>
            <span class="c1">### BEGIN SOLUTION</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">1000</span><span class="p">,</span> <span class="mi">512</span><span class="p">),</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">(),</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">Dropout</span><span class="p">(</span><span class="mf">0.5</span><span class="p">),</span>
            <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">512</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">dog_breeds</span><span class="p">))</span>
            <span class="c1">### END SOLUTION</span>
        <span class="p">)</span>
    
    <span class="c1"># Write the forward method for this network (it&#39;s quite simple since we&#39;ve defined the network in blocks already)</span>
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="c1">### BEGIN SOLUTION</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">feature_extraction</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">classifier</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">x</span>
        <span class="c1">### END SOLUTION</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Instantiate a pretrained network using the class we&#39;ve just defined (call it &#39;pretrained&#39;)</span>

<span class="c1">### BEGIN SOLUTION</span>
<span class="n">pretrained</span> <span class="o">=</span> <span class="n">PreTrained_Resnet</span><span class="p">()</span>
<span class="c1">### END SOLUTION</span>

<span class="c1"># Then define the loss function and optimizer to use for training (let&#39;s use Adam again, with the same parameters as before)</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">criterion2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">CrossEntropyLoss</span><span class="p">()</span>
<span class="n">optimizer2</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">pretrained</span><span class="o">.</span><span class="n">classifier</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span> <span class="o">=</span> <span class="mf">0.0001</span><span class="p">)</span>
<span class="c1">### END SOLUTION</span>

<span class="c1"># Define your number of epochs to train and map your model to the gpu</span>
<span class="c1"># Keep epochs to 5 for time purposes during the workshop</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">epochs2</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">pretrained</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="c1">### END SOLUTION</span>

<span class="n">summary</span><span class="p">(</span><span class="n">pretrained</span><span class="p">,</span> <span class="p">(</span><span class="mi">3</span><span class="p">,</span><span class="mi">224</span><span class="p">,</span><span class="mi">224</span><span class="p">))</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="n">pretrained</span> <span class="o">=</span> <span class="n">train</span><span class="p">(</span><span class="n">pretrained</span><span class="p">,</span> <span class="n">criterion2</span><span class="p">,</span> <span class="n">optimizer2</span><span class="p">,</span> <span class="n">epochs2</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>
</pre></div>



This quick example shows the power of transfer learning. With relatively few lines of code we're able to achieve over an 80% accuracy on this dog breeds dataset! And there are still a number of things we could have done, or do from here, to achieve even better performance. This includes things such as:
 -  Unfreezing the last few layers of the ResNet base and training some more on our specific dataset (more on this in a bit)
 -  Optimizing the hyperparameters of our model (learning rate, etc.)
 -  Utilizing an even more powerful pretrained architecture (Resnet34, 50, etc.)
 -  Creating a custom learning rate schedule

We'll save the model, then load it back up using the function we defined earlier.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">torch</span><span class="o">.</span><span class="n">save</span><span class="p">({</span>
            <span class="s1">&#39;model&#39;</span> <span class="p">:</span> <span class="n">PreTrained_Resnet</span><span class="p">(),</span>
            <span class="s1">&#39;epoch&#39;</span> <span class="p">:</span> <span class="n">epochs2</span><span class="p">,</span>
            <span class="s1">&#39;model_state_dict&#39;</span><span class="p">:</span> <span class="n">pretrained</span><span class="o">.</span><span class="n">state_dict</span><span class="p">(),</span>
            <span class="s1">&#39;optimizer&#39;</span> <span class="p">:</span> <span class="n">optimizer2</span><span class="p">,</span>
            <span class="s1">&#39;optimizer_state_dict&#39;</span><span class="p">:</span> <span class="n">optimizer2</span><span class="o">.</span><span class="n">state_dict</span><span class="p">(),</span>
            <span class="s1">&#39;criterion&#39;</span> <span class="p">:</span> <span class="n">criterion2</span><span class="p">,</span>
            <span class="s1">&#39;device&#39;</span> <span class="p">:</span> <span class="n">device</span>
            <span class="p">},</span> <span class="s1">&#39;pretrained.tar&#39;</span><span class="p">)</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="n">pretrained</span><span class="p">,</span> <span class="n">optimizer2</span><span class="p">,</span> <span class="n">criterion2</span><span class="p">,</span> <span class="n">epoch2</span> <span class="o">=</span> <span class="n">load_checkpoint</span><span class="p">(</span><span class="s1">&#39;pretrained.tar&#39;</span><span class="p">)</span>
</pre></div>



Finally we can test our new pretrained ResNet model! As you can see, with transfer learning we can create quite accurate models relatively easily.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">test_model</span><span class="p">(</span><span class="n">pretrained</span><span class="p">,</span> <span class="mi">6</span><span class="p">)</span>
</pre></div>



#### More on Transfer Learning

In this example, we simply took a pretrained model and added our classification (fully connected layers) block right on top. We froze the entire pretrained network and only updated the weights of our fully connected layers. This means we didn't change the pretrained weights at all, and only used what it had 'learned' from the dataset which it was trained on. 

However, I mentioned earlier that we could achieve even better performance if we unfroze the last few layers of the pretrained model and trained them some on our specific dataset. But why?

<img src = "https://drive.google.com/uc?id=10ce5aTD47lIsO1eYfZmbs_sbDDUfaZiT"> <img src = "https://drive.google.com/uc?id=1BfHJXrWwl4oVyPZ2_p602nD9HkF4RoSR">

Going back to the layer visualizations we saw earlier, we know the earlier layers of the pretrained network learn to recognize simple lines, patterns, objects, etc. However, as we progress in the network, the layers learn to recognize things more specific to the dataset which it was trained on. In this case, ImageNet, which we described a bit earlier.

If you remember, ImageNet contains images that are *somewhat* similar to our dog breeds dataset, so much of what the model 'learned' also applied to our dataset. Hence why we were able to achieve a pretty good accuracy without adjusting the pretrained model whatsoever. 

Of course, much of what the deeper layers learned from ImageNet did **not** apply to dog images. This is why training the last few layers would be beneficial. It would allow the model to adjust and recognize rich features specific to **only dogs**. Things such as types of dog ears, tails, fur, noses, etc. etc.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Run this to generate the submission file for the competition!</span>
<span class="c1">### Make sure to name your model variable &quot;pretrained&quot; ###</span>

<span class="c1"># generate predictions</span>
<span class="n">preds</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">pretrained</span> <span class="o">=</span> <span class="n">pretrained</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="n">pretrained</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>
<span class="k">for</span> <span class="n">img</span> <span class="ow">in</span> <span class="n">test_loader</span><span class="p">:</span>
    <span class="n">outputs</span> <span class="o">=</span> <span class="n">pretrained</span><span class="p">(</span><span class="n">img</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">))</span>
    <span class="n">_</span><span class="p">,</span> <span class="n">outputs</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
    <span class="n">preds</span> <span class="o">+=</span> <span class="p">[</span><span class="n">outputs</span><span class="o">.</span><span class="n">item</span><span class="p">()]</span>

<span class="c1"># create our pandas dataframe for our submission file. Squeeze removes dimensions of 1 in a numpy matrix Ex: (161, 1) -&gt; (161,)</span>
<span class="n">indicies</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;</span><span class="si">{}</span><span class="s2">.jpg&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">image_datasets</span><span class="p">[</span><span class="s1">&#39;Test&#39;</span><span class="p">]))]</span>
<span class="n">preds</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;Id&#39;</span><span class="p">:</span> <span class="n">indicies</span><span class="p">,</span> <span class="s1">&#39;Class&#39;</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">preds</span><span class="p">)})</span>

<span class="c1"># save submission csv</span>
<span class="n">preds</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;submission.csv&#39;</span><span class="p">,</span> <span class="n">header</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Id&#39;</span><span class="p">,</span> <span class="s1">&#39;Class&#39;</span><span class="p">],</span> <span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Submission generated!&quot;</span><span class="p">)</span>
</pre></div>



## Thank you for coming out tonight! 

## Don't forget to sign in at <a href="ucfai.org/signin">ucfai.org/signin</a> if you didn't get the chance to swipe in, and see you next week!
