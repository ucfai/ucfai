---
title: "A look behind DeepFake - GANs"
linktitle: "A look behind DeepFake - GANs"
date: "2019-10-30T00:00:00Z"
lastmod: "2019-10-30T00:00:00Z"
draft: false # Is this a draft? true/false
toc: true # Show table of contents? true/false
type: docs # Do not modify.

menu:
  core_fa19:
    parent: Fall 2019
    weight: 8   

weight: 8

authors: ["brandons209"]

urls:
  youtube: "#"
  slides:  "#"
  github:  "#"
  kaggle:  "#"
  colab:   "#"

categories: ["fa19"]
tags: ["GANs",  "generative", "adversial", "cyclegan", "deepfake", "cGAN"]
description: >-
  GANs are relativity new in the machine learning world, but they have proven to be a very
  powerful model. Recently, they made headlines in the DeepFake network, being able to mimic
  someone else in real time video and audio. There has also been cycleGAN, which takes one domain (horses)
  and makes it look like something similar (zebras). Come and learn the secret behind these type of networks,
  you will be suprised how intuitive it is! The lecture will cover the basics of GANs and different types,
  with the workshop covering how we can generate human faces, cats, dogs, and other cute creatures!
--- 

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># This is a bit of code to make things work on Kaggle</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">from</span> <span class="nn">pathlib</span> <span class="kn">import</span> <span class="n">Path</span>

<span class="k">if</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">exists</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-gans&quot;</span><span class="p">):</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-gans&quot;</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;data/&quot;</span><span class="p">)</span>

<span class="o">!</span>pip install torchsummary
</pre></div>



# Creating New Celebrities
In this notebook we will be generating unqiue human faces based off of celebrities. Maybe one of them will look like their kid? This dataset contains around 200,000 pictures of celebrities faces, all of them aligned to the center of the image. This is important so the GAN can learned the features of the face properly when generating.

Our network will be a DCGAN since we are working with image data, a popular domain for generating new data with GANs. 

As always, lets import all of our libraries needed, and our helper function from printing Epoch results nicely.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># general imports</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">import</span> <span class="nn">math</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">matplotlib.animation</span> <span class="k">as</span> <span class="nn">animation</span>
<span class="kn">from</span> <span class="nn">IPython.display</span> <span class="kn">import</span> <span class="n">HTML</span>

<span class="c1"># torch imports</span>
<span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">torch.nn</span> <span class="k">as</span> <span class="nn">nn</span>
<span class="kn">import</span> <span class="nn">torch.optim</span> <span class="k">as</span> <span class="nn">optim</span>
<span class="kn">import</span> <span class="nn">torch.backends.cudnn</span> <span class="k">as</span> <span class="nn">cudnn</span>
<span class="kn">import</span> <span class="nn">torchvision.transforms</span> <span class="k">as</span> <span class="nn">transforms</span>

<span class="kn">from</span> <span class="nn">torch.utils.data</span> <span class="kn">import</span> <span class="n">DataLoader</span>
<span class="kn">from</span> <span class="nn">torchvision.datasets</span> <span class="kn">import</span> <span class="n">ImageFolder</span>

<span class="kn">import</span> <span class="nn">torchvision.utils</span> <span class="k">as</span> <span class="nn">vutils</span>
<span class="kn">from</span> <span class="nn">torch.utils.data</span> <span class="kn">import</span> <span class="n">random_split</span>

<span class="c1"># uncomment to use specific seed for randomly generating weights and noise</span>
<span class="c1"># seed = 999</span>
<span class="c1"># torch.manual_seed(seed)</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="k">try</span><span class="p">:</span>
    <span class="kn">import</span> <span class="nn">torchsummary</span>
<span class="k">except</span><span class="p">:</span>
    <span class="n">torchsummary</span> <span class="o">=</span> <span class="kc">None</span>

<span class="kn">from</span> <span class="nn">tabulate</span> <span class="kn">import</span> <span class="n">tabulate</span>

<span class="n">BATCH_TEMPLATE</span> <span class="o">=</span> <span class="s2">&quot;Epoch [</span><span class="si">{}</span><span class="s2"> / </span><span class="si">{}</span><span class="s2">], Batch [</span><span class="si">{}</span><span class="s2"> / </span><span class="si">{}</span><span class="s2">]:&quot;</span>
<span class="n">EPOCH_TEMPLATE</span> <span class="o">=</span> <span class="s2">&quot;Epoch [</span><span class="si">{}</span><span class="s2"> / </span><span class="si">{}</span><span class="s2">]:&quot;</span>
<span class="n">TEST_TEMPLATE</span> <span class="o">=</span> <span class="s2">&quot;Epoch [</span><span class="si">{}</span><span class="s2">] Test:&quot;</span>

<span class="k">def</span> <span class="nf">print_iter</span><span class="p">(</span><span class="n">curr_epoch</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">batch_i</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">num_batches</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">writer</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">msg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Formats an iteration. kwargs should be a variable amount of metrics=vals</span>
<span class="sd">    Optional Arguments:</span>
<span class="sd">        curr_epoch(int): current epoch number (should be in range [0, epochs - 1])</span>
<span class="sd">        epochs(int): total number of epochs</span>
<span class="sd">        batch_i(int): current batch iteration</span>
<span class="sd">        num_batches(int): total number of batches</span>
<span class="sd">        writer(SummaryWriter): tensorboardX summary writer object</span>
<span class="sd">        msg(bool): if true, doesn&#39;t print but returns the message string</span>

<span class="sd">    if curr_epoch and epochs is defined, will format end of epoch iteration</span>
<span class="sd">    if batch_i and num_batches is also defined, will define a batch iteration</span>
<span class="sd">    if curr_epoch is only defined, defines a validation (testing) iteration</span>
<span class="sd">    if none of these are defined, defines a single testing iteration</span>
<span class="sd">    if writer is not defined, metrics are not saved to tensorboard</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="n">curr_epoch</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">batch_i</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span> <span class="ow">and</span> <span class="n">num_batches</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span> <span class="ow">and</span> <span class="n">epochs</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="n">out</span> <span class="o">=</span> <span class="n">BATCH_TEMPLATE</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">curr_epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">epochs</span><span class="p">,</span> <span class="n">batch_i</span><span class="p">,</span> <span class="n">num_batches</span><span class="p">)</span>
        <span class="k">elif</span> <span class="n">epochs</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="n">out</span> <span class="o">=</span> <span class="n">EPOCH_TEMPLATE</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">curr_epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">epochs</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">out</span> <span class="o">=</span> <span class="n">TEST_TEMPLATE</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">curr_epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">out</span> <span class="o">=</span> <span class="s2">&quot;Testing Results:&quot;</span>

    <span class="n">floatfmt</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">metric</span><span class="p">,</span> <span class="n">val</span> <span class="ow">in</span> <span class="n">kwargs</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="k">if</span> <span class="s2">&quot;loss&quot;</span> <span class="ow">in</span> <span class="n">metric</span> <span class="ow">or</span> <span class="s2">&quot;recall&quot;</span> <span class="ow">in</span> <span class="n">metric</span> <span class="ow">or</span> <span class="s2">&quot;alarm&quot;</span> <span class="ow">in</span> <span class="n">metric</span> <span class="ow">or</span> <span class="s2">&quot;prec&quot;</span> <span class="ow">in</span> <span class="n">metric</span><span class="p">:</span>
            <span class="n">floatfmt</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s2">&quot;.4f&quot;</span><span class="p">)</span>
        <span class="k">elif</span> <span class="s2">&quot;accuracy&quot;</span> <span class="ow">in</span> <span class="n">metric</span> <span class="ow">or</span> <span class="s2">&quot;acc&quot;</span> <span class="ow">in</span> <span class="n">metric</span><span class="p">:</span>
            <span class="n">floatfmt</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s2">&quot;.2f&quot;</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">floatfmt</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s2">&quot;.6f&quot;</span><span class="p">)</span>

        <span class="k">if</span> <span class="n">writer</span> <span class="ow">and</span> <span class="n">curr_epoch</span><span class="p">:</span>
            <span class="n">writer</span><span class="o">.</span><span class="n">add_scalar</span><span class="p">(</span><span class="n">metric</span><span class="p">,</span> <span class="n">val</span><span class="p">,</span> <span class="n">curr_epoch</span><span class="p">)</span>
        <span class="k">elif</span> <span class="n">writer</span> <span class="ow">and</span> <span class="n">batch_i</span><span class="p">:</span>
            <span class="n">writer</span><span class="o">.</span><span class="n">add_scalar</span><span class="p">(</span><span class="n">metric</span><span class="p">,</span> <span class="n">val</span><span class="p">,</span> <span class="n">batch_i</span> <span class="o">*</span> <span class="p">(</span><span class="n">curr_epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">))</span>

    <span class="n">out</span> <span class="o">+=</span> <span class="s2">&quot;</span><span class="se">\n</span><span class="s2">&quot;</span> <span class="o">+</span> <span class="n">tabulate</span><span class="p">(</span><span class="n">kwargs</span><span class="o">.</span><span class="n">items</span><span class="p">(),</span> <span class="n">headers</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;Metric&quot;</span><span class="p">,</span> <span class="s2">&quot;Value&quot;</span><span class="p">],</span> <span class="n">tablefmt</span><span class="o">=</span><span class="s1">&#39;github&#39;</span><span class="p">,</span> <span class="n">floatfmt</span><span class="o">=</span><span class="n">floatfmt</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">msg</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">out</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">out</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">summary</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">input_dim</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">torchsummary</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
        <span class="k">raise</span><span class="p">(</span><span class="n">ModuleNotFoundError</span><span class="p">,</span> <span class="s2">&quot;TorchSummary was not found!&quot;</span><span class="p">)</span>
    <span class="n">torchsummary</span><span class="o">.</span><span class="n">summary</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">input_dim</span><span class="p">)</span>
</pre></div>



### Dataloader
Let's define our dataset and dataloader using Pytorch's Imagefolder. This was used also in our CNN workshop. The built in Imagefolder dataset will load in images from all subfolders of the given path, and pass it through our transforms without the need for creating a custom dataset. For time purposes, we will set a variable called `thanos_level` that will cut our dataset in half, thirds, fourths etc so we can train on a subset of the 200,000 images. For 5 Epochs, the whole dataset will take about a half hour to train, half will be 15 minutes, and a fourth will be about 6 minutes. For best results, use the whole dataset! 

For transforms, we use a resize down to our image size (keeping it small for speed purposes), center crop the image so the face is centered in the image, convert it to a tensor and normalize it with a STD and mean of 0.5. When this is all done, our RGB scalar values will be betweenn -1 and 1, inclusive, the same as what our generator output will be.

Its important to visualize our data before building the model, so lets take a look and plot some images from the dataset.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">image_size</span> <span class="o">=</span> <span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="mi">64</span><span class="p">)</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">128</span>
<span class="n">num_workers</span> <span class="o">=</span> <span class="mi">4</span>

<span class="c1"># I&#39;m sorry little one</span>
<span class="n">thanos_level</span> <span class="o">=</span> <span class="mi">4</span>

<span class="n">dataset</span> <span class="o">=</span> <span class="n">ImageFolder</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">DATA_DIR</span><span class="p">),</span> <span class="n">transform</span><span class="o">=</span><span class="n">transforms</span><span class="o">.</span><span class="n">Compose</span><span class="p">([</span>
                               <span class="n">transforms</span><span class="o">.</span><span class="n">Resize</span><span class="p">(</span><span class="n">image_size</span><span class="p">),</span>
                               <span class="n">transforms</span><span class="o">.</span><span class="n">CenterCrop</span><span class="p">(</span><span class="n">image_size</span><span class="p">),</span>
                               <span class="n">transforms</span><span class="o">.</span><span class="n">ToTensor</span><span class="p">(),</span>
                               <span class="n">transforms</span><span class="o">.</span><span class="n">Normalize</span><span class="p">((</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">),</span> <span class="p">(</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">)),</span>
                           <span class="p">]))</span>

<span class="c1"># comment out if you want to use whole dataset</span>
<span class="n">dataset</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">random_split</span><span class="p">(</span><span class="n">dataset</span><span class="p">,</span> <span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="p">)</span> <span class="o">/</span> <span class="n">thanos_level</span><span class="p">),</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="p">)</span> <span class="o">-</span> <span class="nb">int</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="p">)</span> <span class="o">/</span> <span class="n">thanos_level</span><span class="p">))])</span>

<span class="c1"># TODO:Create the dataloader from our dataset above</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">dataloader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">dataset</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">num_workers</span><span class="o">=</span><span class="n">num_workers</span><span class="p">)</span>
<span class="c1">### END SOLUTION</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Length of dataset: </span><span class="si">{}</span><span class="s2">, dataloader: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="p">),</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="p">)))</span>

<span class="c1"># Plot some training images</span>
<span class="n">real_batch</span> <span class="o">=</span> <span class="nb">next</span><span class="p">(</span><span class="nb">iter</span><span class="p">(</span><span class="n">dataloader</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s2">&quot;off&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Training Images&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="n">vutils</span><span class="o">.</span><span class="n">make_grid</span><span class="p">(</span><span class="n">real_batch</span><span class="p">[</span><span class="mi">0</span><span class="p">][:</span><span class="mi">64</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">(),(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">0</span><span class="p">)))</span>
</pre></div>



### Generator
Here we will define our generator model. I've created a simple function that will give us a block of the network, which includes the Convolution Tranpose (which convolves and upsamples in one layer), a batch normalization, and our activation function ReLU. I've also included the the `get_padding` helper function we used before, which calculators the required padding needed. You can use the function when building the model or just run it below manually and hardcode the padding.

We start with our input size and want to upsample and reduce the number filters until the final layer has 3 channels for RGB, and 64x64, our output size.

It is important as you build the model to keep track of the size of the feature maps as the network gets deeper, as we need to make sure our output size matches the size we set above! If you want larger sizes we can add more layers to the generator. Try doing 128x128 images after the workshop!



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">get_padding</span><span class="p">(</span><span class="n">output_dim</span><span class="p">,</span> <span class="n">input_dim</span><span class="p">,</span> <span class="n">kernel_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Calculates padding given in output and input dim, and parameters of the convolutional layer</span>

<span class="sd">    Arguments should all be integers. Use this function to calculate padding for 1 dimesion at a time.</span>
<span class="sd">    Output dimensions should be the same or bigger than input dimensions</span>

<span class="sd">    Returns 0 if invalid arguments were passed, otherwise returns an int or tuple that represents the padding.</span>
<span class="sd">    &quot;&quot;&quot;</span>

    <span class="n">padding</span> <span class="o">=</span> <span class="p">(((</span><span class="n">output_dim</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span> <span class="o">*</span> <span class="n">stride</span><span class="p">)</span> <span class="o">-</span> <span class="n">input_dim</span> <span class="o">+</span> <span class="n">kernel_size</span><span class="p">)</span> <span class="o">//</span> <span class="mi">2</span>

    <span class="k">if</span> <span class="n">padding</span> <span class="o">&lt;</span> <span class="mi">0</span><span class="p">:</span>
        <span class="k">return</span> <span class="mi">0</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">padding</span>

<span class="nb">print</span><span class="p">(</span><span class="n">get_padding</span><span class="p">(</span><span class="mi">32</span><span class="p">,</span> <span class="mi">64</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">))</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">gen_block</span><span class="p">(</span><span class="n">input_channels</span><span class="p">,</span> <span class="n">output_channels</span><span class="p">,</span> <span class="n">kernel_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">,</span> <span class="n">padding</span><span class="p">):</span>
    <span class="n">layers</span> <span class="o">=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">ConvTranspose2d</span><span class="p">(</span><span class="n">input_channels</span><span class="p">,</span> <span class="n">output_channels</span><span class="p">,</span> <span class="n">kernel_size</span><span class="p">,</span> <span class="n">stride</span><span class="o">=</span><span class="n">stride</span><span class="p">,</span> <span class="n">padding</span><span class="o">=</span><span class="n">padding</span><span class="p">,</span> <span class="n">bias</span><span class="o">=</span><span class="kc">False</span><span class="p">)]</span>
    <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">BatchNorm2d</span><span class="p">(</span><span class="n">output_channels</span><span class="p">)]</span>
    <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">(</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)]</span>
    
    <span class="k">return</span> <span class="n">layers</span>
    
<span class="k">class</span> <span class="nc">Generator</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">channels</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">input_size</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span> <span class="n">output_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">Generator</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">channels</span> <span class="o">=</span> <span class="n">channels</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">input_size</span> <span class="o">=</span> <span class="n">input_size</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">output_size</span> <span class="o">=</span> <span class="n">output_dim</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">layers</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">build_layers</span><span class="p">()</span>
        
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">layers</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>
    
    <span class="k">def</span> <span class="nf">build_layers</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="n">layers</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">input_size</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_size</span> <span class="o">*</span> <span class="mi">8</span>
        
        <span class="c1"># dim: out_c x 4 x 4</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="n">gen_block</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_size</span> <span class="o">*</span> <span class="mi">4</span>
        
        <span class="c1"># TODO: Create the next two blocks the same way the above one is created</span>
        <span class="c1"># Use kernel size of 4 and a stride of 2. Whats the padding?</span>
        <span class="c1">### BEGIN SOLUTION</span>
        <span class="c1"># dim: out_c x 8 x 8</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="n">gen_block</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_size</span> <span class="o">*</span> <span class="mi">2</span>
        
        <span class="c1"># dim: out_c x 16 x 16</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="n">gen_block</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_size</span>
        <span class="c1">### END SOLUTION</span>
        <span class="c1"># dim: out_c x 32 x 32</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="n">gen_block</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">channels</span>
        
        <span class="c1"># dim: out_c x 64 x 64</span>
        <span class="c1"># don&#39;t use batch norm in the last layer since its the output.</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">ConvTranspose2d</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span> <span class="n">nn</span><span class="o">.</span><span class="n">Tanh</span><span class="p">()]</span>
        
        <span class="k">return</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span><span class="o">*</span><span class="n">layers</span><span class="p">)</span>
</pre></div>



### Discriminator
Now for the discriminator. This will be a simple CNN that we have seen before. The few differences is that we are going to use [LeakyReLU](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7) which is a adapatation to ReLU. LeakyReLU has a chance to "leak" negative values from the function output, instead of zeroing out **all** negative values. This has shown to give better results for the discriminator and help avoid the issues mentioned at the end of the slides.

Instead of max pooling, we use larger strides to halve our input size down until 1 node, which will be our output for the discriminator of either Real or Fake. Pooling in GANs is usually never used as it almost always creates models that don' train. Its better to have a larger stride to reduce size of the feature maps. Since we want the generator to produce images representing the input, it needs context of the whole image, so max pooling would not help here.

Another important note is to not use batch normalization in the first or last block of the discriminator, it can cause the model to not train.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">discrim_block</span><span class="p">(</span><span class="n">input_channels</span><span class="p">,</span> <span class="n">output_channels</span><span class="p">,</span> <span class="n">kernel_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">,</span> <span class="n">padding</span><span class="p">):</span>
    <span class="n">layers</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="n">input_channels</span><span class="p">,</span> <span class="n">output_channels</span><span class="p">,</span> <span class="n">kernel_size</span><span class="p">,</span> <span class="n">stride</span><span class="o">=</span><span class="n">stride</span><span class="p">,</span> <span class="n">padding</span><span class="o">=</span><span class="n">padding</span><span class="p">,</span> <span class="n">bias</span><span class="o">=</span><span class="kc">False</span><span class="p">)]</span>
    <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">BatchNorm2d</span><span class="p">(</span><span class="n">output_channels</span><span class="p">)]</span>
    <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">LeakyReLU</span><span class="p">(</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)]</span>
    
    <span class="k">return</span> <span class="n">layers</span>

<span class="k">class</span> <span class="nc">Discriminator</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">channels</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">input_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">Discriminator</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">channels</span> <span class="o">=</span> <span class="n">channels</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">input_dim</span> <span class="o">=</span> <span class="n">input_dim</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">layers</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">build_layers</span><span class="p">()</span>
        
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">layers</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>
    
    <span class="k">def</span> <span class="nf">build_layers</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="n">layers</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">channels</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">input_dim</span>
        
        <span class="c1"># dim: out_c x 32 x 32</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">bias</span><span class="o">=</span><span class="kc">False</span><span class="p">),</span> <span class="n">nn</span><span class="o">.</span><span class="n">LeakyReLU</span><span class="p">(</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)]</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">input_dim</span> <span class="o">*</span> <span class="mi">2</span>
        <span class="c1"># TODO: Create the next 2 blocks for the discriminator. Kernel size of 4 and a stride of 2</span>
        <span class="c1"># this is quite similar to the generator...</span>
        <span class="c1">### BEGIN SOLUTION</span>
        <span class="c1"># dim: out_c x 16 x 16</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="n">discrim_block</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">input_dim</span> <span class="o">*</span> <span class="mi">4</span>
        
        <span class="c1"># dim: out_c x 8 x 8</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="n">discrim_block</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">input_dim</span> <span class="o">*</span> <span class="mi">8</span>
        <span class="c1">### END SOLUTION</span>
        <span class="c1"># dim: out_c x 4 x 4</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="n">discrim_block</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">in_c</span> <span class="o">=</span> <span class="n">out_c</span>
        <span class="n">out_c</span> <span class="o">=</span> <span class="mi">1</span>
        
        <span class="c1"># dim: 1</span>
        <span class="n">layers</span> <span class="o">+=</span> <span class="p">[</span><span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="n">in_c</span><span class="p">,</span> <span class="n">out_c</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">),</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sigmoid</span><span class="p">()]</span>
        
        <span class="k">return</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span><span class="o">*</span><span class="n">layers</span><span class="p">)</span>
        
</pre></div>



### Define function for initalizing weights
Lets define a function to initalize our weights a certain way, this is following the DCGAN paper and their parameters they used. Convolution weights are randomized from a normal distrubution with a mean of 0 and STD of 0.02, with batch normalization weights randomized from a nnormal distrubution with a mean of 1 and a STD of 0.02.

This is how the DCGAN paper had it, so it should *hopefully* help produce the best results.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">weights_init</span><span class="p">(</span><span class="n">m</span><span class="p">):</span>
    <span class="n">classname</span> <span class="o">=</span> <span class="n">m</span><span class="o">.</span><span class="vm">__class__</span><span class="o">.</span><span class="vm">__name__</span>
    <span class="k">if</span> <span class="n">classname</span><span class="o">.</span><span class="n">find</span><span class="p">(</span><span class="s1">&#39;Conv&#39;</span><span class="p">)</span> <span class="o">!=</span> <span class="o">-</span><span class="mi">1</span><span class="p">:</span>
        <span class="n">nn</span><span class="o">.</span><span class="n">init</span><span class="o">.</span><span class="n">normal_</span><span class="p">(</span><span class="n">m</span><span class="o">.</span><span class="n">weight</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="mf">0.0</span><span class="p">,</span> <span class="mf">0.02</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">classname</span><span class="o">.</span><span class="n">find</span><span class="p">(</span><span class="s1">&#39;BatchNorm&#39;</span><span class="p">)</span> <span class="o">!=</span> <span class="o">-</span><span class="mi">1</span><span class="p">:</span>
        <span class="n">nn</span><span class="o">.</span><span class="n">init</span><span class="o">.</span><span class="n">normal_</span><span class="p">(</span><span class="n">m</span><span class="o">.</span><span class="n">weight</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">,</span> <span class="mf">0.02</span><span class="p">)</span>
        <span class="n">nn</span><span class="o">.</span><span class="n">init</span><span class="o">.</span><span class="n">constant_</span><span class="p">(</span><span class="n">m</span><span class="o">.</span><span class="n">bias</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
</pre></div>



### Create models, optimizers, and loss function
Now let's create our generator and discriminator models and apply our weight initalization function to them.

We also need to define the optimizer, device, and loss function. From the DCGAN paper, we will use the Adam optimizer with different betas parameters. These betas define how aggresive the optimizer is in reducing the learning rate on a plateau. The GAN suffers if the optimizer is too agreesive, so we reduce this behavior. **We need two optimizers, one for the generator and one for the discriminator.**

Our loss function will be Binary Cross Entropy since we have binary labels.

For purposes of visualizing our model lets define some fixed noise which we will generate examples on each batch iteration, so we can see how the model improves throughout training.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">gen_input</span> <span class="o">=</span> <span class="mi">100</span>
<span class="n">gen_output</span> <span class="o">=</span> <span class="mi">64</span>

<span class="n">gen</span> <span class="o">=</span> <span class="n">Generator</span><span class="p">(</span><span class="n">input_size</span><span class="o">=</span><span class="n">gen_input</span><span class="p">,</span> <span class="n">output_dim</span><span class="o">=</span><span class="n">gen_output</span><span class="p">)</span>
<span class="n">gen</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">weights_init</span><span class="p">)</span>
<span class="n">discrim</span> <span class="o">=</span> <span class="n">Discriminator</span><span class="p">(</span><span class="n">channels</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">input_dim</span><span class="o">=</span><span class="n">gen_output</span><span class="p">)</span>
<span class="n">discrim</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">weights_init</span><span class="p">)</span>

<span class="n">device</span> <span class="o">=</span> <span class="s1">&#39;cuda&#39;</span> <span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span> <span class="k">else</span> <span class="s1">&#39;cpu&#39;</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Using device: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">device</span><span class="p">))</span>
<span class="n">gen</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="n">discrim</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

<span class="c1"># hyperparameters from DCGAN paper</span>
<span class="n">learn_rate</span> <span class="o">=</span> <span class="mf">0.0002</span>

<span class="n">optG</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">gen</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="n">learn_rate</span><span class="p">,</span> <span class="n">betas</span><span class="o">=</span><span class="p">(</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.999</span><span class="p">))</span>
<span class="n">optD</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">discrim</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="n">learn_rate</span><span class="p">,</span> <span class="n">betas</span><span class="o">=</span><span class="p">(</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.999</span><span class="p">))</span>

<span class="c1"># TODO: Define our criterion (loss function)</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">criterion</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">BCELoss</span><span class="p">()</span>
<span class="c1">### END SOLUTION</span>
<span class="n">fixed_noise</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">gen_output</span><span class="p">,</span> <span class="n">gen_input</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>

<span class="n">real_label</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">fake_label</span> <span class="o">=</span> <span class="mi">0</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Generator:&quot;</span><span class="p">)</span>
<span class="n">summary</span><span class="p">(</span><span class="n">gen</span><span class="p">,</span> <span class="p">(</span><span class="n">gen_input</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="se">\n</span><span class="s2">Discriminator:&quot;</span><span class="p">)</span>
<span class="n">summary</span><span class="p">(</span><span class="n">discrim</span><span class="p">,</span> <span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="n">gen_output</span><span class="p">,</span> <span class="n">gen_output</span><span class="p">))</span>
</pre></div>



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



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">start_time</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>

<span class="n">epochs</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">print_step</span> <span class="o">=</span> <span class="mi">50</span>

<span class="n">gen_imgs</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">epochs</span><span class="p">):</span>
    <span class="n">g_train_loss</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">d_train_loss</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">e_time</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>
    
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">dataloader</span><span class="p">):</span>

        <span class="c1"># Train Discriminator</span>
        
        <span class="c1"># only need images from data, don&#39;t care about class from ImageFolder</span>
        <span class="n">images</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
        <span class="n">b_size</span> <span class="o">=</span> <span class="n">images</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
        <span class="n">labels</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">full</span><span class="p">((</span><span class="n">b_size</span><span class="p">,),</span> <span class="n">real_label</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
        
        <span class="c1"># train on real</span>
        <span class="n">discrim</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
        <span class="n">d_output</span> <span class="o">=</span> <span class="n">discrim</span><span class="p">(</span><span class="n">images</span><span class="p">)</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">loss_real</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">d_output</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span>
        <span class="n">loss_real</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
      
        <span class="c1"># get fake data from generator</span>
        <span class="n">noise</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">b_size</span><span class="p">,</span> <span class="n">gen_input</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
        <span class="n">fake_images</span> <span class="o">=</span> <span class="n">gen</span><span class="p">(</span><span class="n">noise</span><span class="p">)</span>
        <span class="c1"># this replaces all values in labels with fake_label, which is zero in this case</span>
        <span class="n">labels</span><span class="o">.</span><span class="n">fill_</span><span class="p">(</span><span class="n">fake_label</span><span class="p">)</span>
        
        <span class="c1"># calculate loss and update gradients on fake</span>
        <span class="c1"># must detach the fake images from the computational graph of the generator, so that gradients arent updated for the generator</span>
        <span class="n">d_output</span> <span class="o">=</span> <span class="n">discrim</span><span class="p">(</span><span class="n">fake_images</span><span class="o">.</span><span class="n">detach</span><span class="p">())</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">loss_fake</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">d_output</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span>
        <span class="n">loss_fake</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        
        <span class="c1"># add up real and fake loss</span>
        <span class="n">d_loss</span> <span class="o">=</span> <span class="n">loss_real</span> <span class="o">+</span> <span class="n">loss_fake</span>
        
        <span class="c1"># optimize weights after calculating real and fake loss then backprogating on each</span>
        <span class="n">optD</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
        
        <span class="n">d_train_loss</span> <span class="o">+=</span> <span class="n">d_loss</span><span class="o">.</span><span class="n">item</span><span class="p">()</span>
        
        <span class="c1"># Train Generator</span>
        <span class="n">gen</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
        <span class="n">labels</span><span class="o">.</span><span class="n">fill_</span><span class="p">(</span><span class="n">real_label</span><span class="p">)</span>
        <span class="c1"># get new output from discriminator for fake images, which is now updated from our above step</span>
        <span class="n">d_output</span> <span class="o">=</span> <span class="n">discrim</span><span class="p">(</span><span class="n">fake_images</span><span class="p">)</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>
        <span class="c1"># calculate the Generator&#39;s loss based on this, use real_labels since fake images should be real for generator</span>
        <span class="c1"># i.e the generator wants the discriminator to output real for it&#39;s fake images, so thats the target for generator</span>
        <span class="n">g_loss</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">d_output</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span>
        <span class="n">g_loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">optG</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
        
        <span class="n">g_train_loss</span> <span class="o">+=</span> <span class="n">g_loss</span><span class="o">.</span><span class="n">item</span><span class="p">()</span>
        
        <span class="k">if</span> <span class="n">i</span> <span class="o">%</span> <span class="n">print_step</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">print_iter</span><span class="p">(</span><span class="n">curr_epoch</span><span class="o">=</span><span class="n">e</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="n">epochs</span><span class="p">,</span> <span class="n">batch_i</span><span class="o">=</span><span class="n">i</span><span class="p">,</span> <span class="n">num_batches</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="p">),</span> <span class="n">d_loss</span><span class="o">=</span><span class="n">d_train_loss</span><span class="o">/</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">),</span> <span class="n">g_loss</span><span class="o">=</span><span class="n">g_train_loss</span><span class="o">/</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">))</span>
            <span class="c1"># save example images</span>
            <span class="n">gen</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>
            <span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
                <span class="n">fake_images</span> <span class="o">=</span> <span class="n">gen</span><span class="p">(</span><span class="n">fixed_noise</span><span class="p">)</span><span class="o">.</span><span class="n">detach</span><span class="p">()</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span>
                <span class="n">gen</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
                <span class="n">gen_imgs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">vutils</span><span class="o">.</span><span class="n">make_grid</span><span class="p">(</span><span class="n">fake_images</span><span class="p">,</span> <span class="n">padding</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">))</span>
                
    <span class="n">print_iter</span><span class="p">(</span><span class="n">curr_epoch</span><span class="o">=</span><span class="n">e</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="n">epochs</span><span class="p">,</span> <span class="n">d_loss</span><span class="o">=</span><span class="n">d_train_loss</span><span class="o">/</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">),</span> <span class="n">g_loss</span><span class="o">=</span><span class="n">g_train_loss</span><span class="o">/</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">))</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="se">\n</span><span class="s2">Epoch </span><span class="si">{}</span><span class="s2"> took </span><span class="si">{:.2f}</span><span class="s2"> minutes.</span><span class="se">\n</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">e</span><span class="o">+</span><span class="mi">1</span><span class="p">,</span> <span class="p">(</span><span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">e_time</span><span class="p">)</span> <span class="o">/</span> <span class="mi">60</span><span class="p">))</span>
    
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Model took </span><span class="si">{:.2f}</span><span class="s2"> minutes to train.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">((</span><span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">start_time</span><span class="p">)</span> <span class="o">/</span> <span class="mi">60</span><span class="p">))</span>
</pre></div>



### View Results
This segment of code will create a small animation that goes through the generator's output through training. Notice how the features become more clearer as time goes on. Its able to produce a human face in RGB, amazing!



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s2">&quot;off&quot;</span><span class="p">)</span>
<span class="n">ims</span> <span class="o">=</span> <span class="p">[[</span><span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="n">i</span><span class="p">,(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">0</span><span class="p">)),</span> <span class="n">animated</span><span class="o">=</span><span class="kc">True</span><span class="p">)]</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">gen_imgs</span><span class="p">]</span>
<span class="n">ani</span> <span class="o">=</span> <span class="n">animation</span><span class="o">.</span><span class="n">ArtistAnimation</span><span class="p">(</span><span class="n">fig</span><span class="p">,</span> <span class="n">ims</span><span class="p">,</span> <span class="n">interval</span><span class="o">=</span><span class="mi">1000</span><span class="p">,</span> <span class="n">repeat_delay</span><span class="o">=</span><span class="mi">1000</span><span class="p">,</span> <span class="n">blit</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="n">HTML</span><span class="p">(</span><span class="n">ani</span><span class="o">.</span><span class="n">to_jshtml</span><span class="p">())</span>
</pre></div>



### Final Results
This will show the last epoch's results, which hopefully will be our best.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Grab a batch of real images from the dataloader</span>
<span class="n">real_batch</span> <span class="o">=</span> <span class="nb">next</span><span class="p">(</span><span class="nb">iter</span><span class="p">(</span><span class="n">dataloader</span><span class="p">))</span>

<span class="c1"># Plot the real images</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">15</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">subplot</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s2">&quot;off&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Real Images&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="n">vutils</span><span class="o">.</span><span class="n">make_grid</span><span class="p">(</span><span class="n">real_batch</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)[:</span><span class="mi">64</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">(),(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">0</span><span class="p">)))</span>

<span class="c1"># Plot the fake images from the last epoch</span>
<span class="n">plt</span><span class="o">.</span><span class="n">subplot</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s2">&quot;off&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Fake Images&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="n">gen_imgs</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">],(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">0</span><span class="p">)))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>



# Closing Thoughts
Now that we've built a GAN, the possibilities are endless for what you can apply this too! Getting this model to train is another story though, it'll be lot of playing around and trial/error, but for a very amazing result. I suggest you find datasets of cats or some images that you can try to use this model to train on. You can also try your hand on implementing a cGAN or InfoGAN model, using this as a base. Take the time to explore what you can do and try it out!

For this dataset, try increasing the size of the model to generate larger image sizes, like 128, 128. You would need to add a layer to the generator and discriminator, and probably reduce your batch size and such. You can also try training on the whole dataset for a longer time and see what you get!
