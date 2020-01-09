<img src="https://ucfai.org/core/fa19/2019-10-02-nns/nns/banner.png">

<div class="col-12">
    <span class="btn btn-success btn-block">
        Meeting in-person? Have you signed in?
    </span>
</div>

<div class="col-12">
    <h1> Getting Started with Neural Networks </h1>
    <hr>
</div>

<div style="line-height: 2em;">
    <p>by: 
        <strong> Liam Jarvis</strong>
        (<a href="https://github.com/jarviseq">@jarviseq</a>)
     on 2019-10-02</p>
</div>



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># This is a bit of code to make things work on Kaggle</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">from</span> <span class="nn">pathlib</span> <span class="kn">import</span> <span class="n">Path</span>

<span class="k">if</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">exists</span><span class="p">(</span><span class="s2">&quot;/kaggle/input&quot;</span><span class="p">):</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;/kaggle/input&quot;</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s2">&quot;We don&#39;t know this machine.&quot;</span><span class="p">)</span>
</pre></div>



## Before we get started

We need to import some non-sense

Pytorch is imported as just torch, otherwise we've seen everything else before.



<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">torch</span> 
<span class="kn">import</span> <span class="nn">torch.nn</span> <span class="k">as</span> <span class="nn">nn</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="k">as</span> <span class="nn">F</span>
<span class="kn">from</span> <span class="nn">torch.utils.data</span> <span class="kn">import</span> <span class="n">Dataset</span><span class="p">,</span> <span class="n">DataLoader</span>
<span class="kn">from</span> <span class="nn">torch</span> <span class="kn">import</span> <span class="n">optim</span>
<span class="kn">import</span> <span class="nn">time</span>
</pre></div>



## Tensors

Tensors live at the heart of Pytorch.

You can think of tensors as an Nth-dimensional data container similar to the containers that exist in numpy. 

Below we have some *magical* tensor stuff 
going on to show you how to make some tensors using the built-in tensor generating functions. 



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># create a tensor</span>
<span class="n">new_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span> <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">]])</span>

<span class="c1"># create a 2 x 3 tensor with random values</span>
<span class="n">empty_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>

<span class="c1"># create a 2 x 3 tensor with random values between -1and 1</span>
<span class="n">uniform_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span><span class="o">.</span><span class="n">uniform_</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>

<span class="c1"># create a 2 x 3 tensor with random values from a uniform distribution on the interval [0, 1)</span>
<span class="n">rand_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">rand</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>

<span class="c1"># create a 2 x 3 tensor of zeros</span>
<span class="n">zero_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>
</pre></div>



To see what's inside of the tensor, put the name of the tensor into a code block and run it. 

These notebook environments are meant to be easy for you to debug your code, 
so this will not work if you are writing a python script and running it in a command line.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">new_tensor</span>
</pre></div>



You can replace elements in tensors with indexing. 
It works a lot like arrays you will see in many programming languages. 




<div class=" highlight hl-ipython3"><pre><span></span><span class="n">new_tensor</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">new_tensor</span>
</pre></div>



How the tensor is put together is going to be important, so there are some 
built-in commands in torch that allow you to find out some information about the tensor you are working with.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># type of a tensor</span>
<span class="nb">print</span><span class="p">(</span><span class="n">new_tensor</span><span class="o">.</span><span class="n">type</span><span class="p">())</span>  

<span class="c1"># shape of a tensor</span>
<span class="nb">print</span><span class="p">(</span><span class="n">new_tensor</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>    
<span class="nb">print</span><span class="p">(</span><span class="n">new_tensor</span><span class="o">.</span><span class="n">size</span><span class="p">())</span>   

<span class="c1"># dimension of a tensor</span>
<span class="nb">print</span><span class="p">(</span><span class="n">new_tensor</span><span class="o">.</span><span class="n">dim</span><span class="p">())</span>
</pre></div>



## Coming from Numpy

Much of your data manipulation will be done in either pandas or numpy. 
To feed that manipulated data into a tensor for use in torch, you will have to use the `.from_numpy` command.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">np_ndarray</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
<span class="n">np_ndarray</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># NumPy ndarray to PyTorch tensor</span>
<span class="n">to_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">from_numpy</span><span class="p">(</span><span class="n">np_ndarray</span><span class="p">)</span>

<span class="n">to_tensor</span>
</pre></div>



## Checking for CUDA

CUDA will speed up the training of your Neural Network greatly.

Your notebook should already have CUDA enabled, but the following command can be used to check for it.

TL:DR: CUDA rock for NNs



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span>
</pre></div>



## Defining Networks

In the example below, we are going to make a simple example to show how you will go about 
building a Neural Network using a randomly generated dataset. This will be a simple network with one hidden layer.

First, we need to set some placeholder variables to define how we want the network to be set up.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">n_in</span><span class="p">,</span> <span class="n">n_h</span><span class="p">,</span> <span class="n">n_out</span><span class="p">,</span> <span class="n">batch_size</span> <span class="o">=</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">10</span>
</pre></div>



Next, we are going to generate our lovely randomised dataset.

We are not expecting any insights to come from this network as the data is generated randomly. 



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">n_in</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">([[</span><span class="mf">1.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">0.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">0.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">1.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">1.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">1.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">0.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">0.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">1.0</span><span class="p">],</span> <span class="p">[</span><span class="mf">1.0</span><span class="p">]])</span>
</pre></div>



Next, we are going to define what our model looks like. The `Linear()` part applies a linear transformation to the 
incoming data, with `Sigmoid()` being the activation function that we use for that layer. 

So, for this network, we have two fully connected layers with a sigmoid as the activation function. 
This looks a lot like the network we saw in the slide deck with one input layer, one hidden layer, and one output layer. 



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">n_in</span><span class="p">,</span> <span class="n">n_h</span><span class="p">),</span>
                     <span class="n">nn</span><span class="o">.</span><span class="n">Sigmoid</span><span class="p">(),</span>
                     <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">n_h</span><span class="p">,</span> <span class="n">n_out</span><span class="p">),</span>
                     <span class="n">nn</span><span class="o">.</span><span class="n">Sigmoid</span><span class="p">())</span>
</pre></div>



Next, let's define what the loss fucntion will be.

For this example, we are going to use Mean Squared Error, but there are a ton of different loss functions we can use.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">criterion</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">MSELoss</span><span class="p">()</span>
</pre></div>



Optimizer is how the network will be training. 

We are going to be using a standard gradient descent method in this example.

We will have a learning rate of 0.01, which is pretty standard too.
 You are going to want to keep this learning rate pretty low, as high learning rates cause problems in training.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">optimizer</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">optim</span><span class="o">.</span><span class="n">SGD</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="mf">0.01</span><span class="p">)</span>
</pre></div>



Now, let's train!

To train, we combine all the different parts that we defined into one for loop.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">50</span><span class="p">):</span>
    <span class="c1"># Forward Propagation</span>
    <span class="n">y_pred</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="c1"># Compute and print loss</span>
    <span class="n">loss</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">y_pred</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;epoch: &#39;</span><span class="p">,</span> <span class="n">epoch</span><span class="p">,</span><span class="s1">&#39; loss: &#39;</span><span class="p">,</span> <span class="n">loss</span><span class="o">.</span><span class="n">item</span><span class="p">())</span>
    <span class="c1"># Zero the gradients</span>
    <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
    
    <span class="c1"># perform a backward pass (backpropagation)</span>
    <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
    
    <span class="c1"># Update the parameters</span>
    <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
</pre></div>



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



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">DATA_DIR</span> <span class="o">/</span> <span class="s2">&quot;train.csv&quot;</span><span class="p">,</span> <span class="n">header</span><span class="o">=</span><span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">values</span>

<span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">dataset</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>



## What are we looking at?

This is a fairly small dataset that includes some basic information about an individual's health. 

Using this information, we should be able to make a model that will allow us to determine if a person has diabetes or not. 

The last column, `Outcome`, is a single digit that tells us if an individual has diabetes. 

We need to clean up the data a bit, so let's get rid of the first row with the labels on them.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">delete</span><span class="p">(</span><span class="n">dataset</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>

<span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">dataset</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>



Alright, now let's break up our data into test and train set.

Once we have those sets, we'll need to set them to be tensors. 

This bit of code below does just that!



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># split into x and y sets</span>

<span class="n">X</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[:,:</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>

<span class="n">Y</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[:,</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>

<span class="c1"># Needed to make PyTorch happy</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">expand_dims</span><span class="p">(</span><span class="n">Y</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>

<span class="c1"># Test-Train split</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>

<span class="n">xTrain</span><span class="p">,</span> <span class="n">xTest</span><span class="p">,</span> <span class="n">yTrain</span><span class="p">,</span> <span class="n">yTest</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.1</span><span class="p">)</span>

<span class="c1"># Here we&#39;re defining what component we&#39;ll use to train this model</span>
<span class="c1"># We want to use the GPU if available, if not we use the CPU</span>
<span class="c1"># If your device is not cuda, check the GPU option in the Kaggle Kernel</span>

<span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">device</span><span class="p">(</span><span class="s2">&quot;cuda:0&quot;</span> <span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span> <span class="k">else</span> <span class="s2">&quot;cpu&quot;</span><span class="p">)</span>
<span class="n">device</span>
</pre></div>



### PyTorch Dataset
Our next step is to create PyTorch **Datasets** for our training and validation sets. 
**torch.utils.data.Dataset** is an abstract class that represents a dataset and 
has several handy attributes we'll utilize from here on out. 

To create one, we simply need to create a class which inherits from PyTorch's Dataset class and 
override the constructor, as well as the __len__() and __getitem__() methods.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">class</span> <span class="nc">PyTorch_Dataset</span><span class="p">(</span><span class="n">Dataset</span><span class="p">):</span>
  
  <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">outputs</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">data</span> <span class="o">=</span> <span class="n">data</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">outputs</span> <span class="o">=</span> <span class="n">outputs</span>

  <span class="k">def</span> <span class="fm">__len__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="s1">&#39;Returns the total number of samples in this dataset&#39;</span>
        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">data</span><span class="p">)</span>

  <span class="k">def</span> <span class="fm">__getitem__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">index</span><span class="p">):</span>
        <span class="s1">&#39;Returns a row of data and its output&#39;</span>
      
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">data</span><span class="p">[</span><span class="n">index</span><span class="p">]</span>
        <span class="n">y</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">outputs</span><span class="p">[</span><span class="n">index</span><span class="p">]</span>

        <span class="k">return</span> <span class="n">x</span><span class="p">,</span> <span class="n">y</span>
</pre></div>



With the class written, we can now create our training and validation 
datasets by passing the corresponding data to our class



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train_dataset</span> <span class="o">=</span> <span class="n">PyTorch_Dataset</span><span class="p">(</span><span class="n">xTrain</span><span class="p">,</span> <span class="n">yTrain</span><span class="p">)</span>
<span class="n">val_dataset</span> <span class="o">=</span> <span class="n">PyTorch_Dataset</span><span class="p">(</span><span class="n">xTest</span><span class="p">,</span> <span class="n">yTest</span><span class="p">)</span>

<span class="n">datasets</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;Train&#39;</span><span class="p">:</span> <span class="n">train_dataset</span><span class="p">,</span> <span class="s1">&#39;Validation&#39;</span><span class="p">:</span> <span class="n">val_dataset</span><span class="p">}</span>
</pre></div>



### PyTorch Dataloaders

It's quite inefficient to load an entire dataset onto your RAM at once, so PyTorch uses **DataLoaders** to 
load up batches of data on the fly. We pass a batch size of 16,
 so in each iteration the loaders will load 16 rows of data and return them to us.

For the most part, Neural Networks are trained on **batches** of data so these DataLoaders greatly simplify 
the process of loading and feeding data to our network. The rank 2 tensor returned by the dataloader is of size (16, 8).



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataloaders</span> <span class="o">=</span> <span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">datasets</span><span class="p">[</span><span class="n">x</span><span class="p">],</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">num_workers</span> <span class="o">=</span> <span class="mi">4</span><span class="p">)</span>
              <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">&#39;Train&#39;</span><span class="p">,</span> <span class="s1">&#39;Validation&#39;</span><span class="p">]}</span>
</pre></div>



### PyTorch Model

We need to define how we want the neural network to be structured, 
so let's set those hyper-parameters and create our model.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">inputSize</span> <span class="o">=</span>  <span class="mi">8</span>         <span class="c1"># how many classes of input</span>
<span class="n">hiddenSize</span> <span class="o">=</span> <span class="mi">15</span>        <span class="c1"># Number of units in the middle</span>
<span class="n">numClasses</span> <span class="o">=</span> <span class="mi">1</span>         <span class="c1"># Only has two classes</span>
<span class="n">numEpochs</span> <span class="o">=</span> <span class="mi">20</span>         <span class="c1"># How many training cycles</span>
<span class="n">learningRate</span> <span class="o">=</span> <span class="o">.</span><span class="mi">01</span>     <span class="c1"># Learning rate</span>

<span class="k">class</span> <span class="nc">NeuralNet</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">input_size</span><span class="p">,</span> <span class="n">hidden_size</span><span class="p">,</span> <span class="n">num_classes</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">NeuralNet</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">input_size</span><span class="p">,</span> <span class="n">hidden_size</span><span class="p">)</span> 
        <span class="bp">self</span><span class="o">.</span><span class="n">fc2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">hidden_size</span><span class="p">,</span> <span class="n">num_classes</span><span class="p">)</span>  
    
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc1</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc2</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
</pre></div>



### PyTorch Training

Now we create an instance of this NeuralNet() class and define the loss function and optimizer 
we'll use to train our model. 
In our case we'll use Binary Cross Entropy Loss, a commonly used loss function binary classification problems.

For the optimizer we'll use Adam, an easy to apply but powerful optimizer which is an extension of the popular 
Stochastic Gradient Descent method. We need to pass it all of the parameters it'll train, 
which PyTorch makes easy with model.parameters(), and also the learning rate we'll use.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Creating our model</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">NeuralNet</span><span class="p">(</span><span class="n">inputSize</span><span class="p">,</span> <span class="n">hiddenSize</span><span class="p">,</span> <span class="n">numClasses</span><span class="p">)</span>
<span class="n">criterion</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">BCELoss</span><span class="p">()</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span> <span class="o">=</span> <span class="n">learningRate</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>
</pre></div>



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



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">run_epoch</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">phase</span><span class="p">):</span>
  
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

      <span class="c1"># Adjust weights through backpropagation if we&#39;re in training phase</span>
      <span class="k">if</span> <span class="n">phase</span> <span class="o">==</span> <span class="s1">&#39;Train&#39;</span><span class="p">:</span>
        <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
      
    <span class="c1"># Get binary predictions</span>
    <span class="n">preds</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="n">outputs</span><span class="p">)</span>

    <span class="c1"># Document statistics for the batch</span>
    <span class="n">running_loss</span> <span class="o">+=</span> <span class="n">loss</span><span class="o">.</span><span class="n">item</span><span class="p">()</span> <span class="o">*</span> <span class="n">inputs</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">running_corrects</span> <span class="o">+=</span> <span class="n">torch</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">preds</span> <span class="o">==</span> <span class="n">labels</span><span class="p">)</span>
    
  <span class="c1"># Calculate epoch statistics</span>
  <span class="n">epoch_loss</span> <span class="o">=</span> <span class="n">running_loss</span> <span class="o">/</span> <span class="n">datasets</span><span class="p">[</span><span class="n">phase</span><span class="p">]</span><span class="o">.</span><span class="fm">__len__</span><span class="p">()</span>
  <span class="n">epoch_acc</span> <span class="o">=</span> <span class="n">running_corrects</span><span class="o">.</span><span class="n">double</span><span class="p">()</span> <span class="o">/</span> <span class="n">datasets</span><span class="p">[</span><span class="n">phase</span><span class="p">]</span><span class="o">.</span><span class="fm">__len__</span><span class="p">()</span>
  
  <span class="k">return</span> <span class="n">epoch_loss</span><span class="p">,</span> <span class="n">epoch_acc</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">train</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">criterion</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">num_epochs</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">):</span>
    <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>

    <span class="n">best_model_wts</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">state_dict</span><span class="p">()</span>
    <span class="n">best_acc</span> <span class="o">=</span> <span class="mf">0.0</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;| Epoch</span><span class="se">\t</span><span class="s1"> | Train Loss</span><span class="se">\t</span><span class="s1">| Train Acc</span><span class="se">\t</span><span class="s1">| Valid Loss</span><span class="se">\t</span><span class="s1">| Valid Acc</span><span class="se">\t</span><span class="s1">|&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;-&#39;</span> <span class="o">*</span> <span class="mi">73</span><span class="p">)</span>
    
    <span class="c1"># Iterate through epochs</span>
    <span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">num_epochs</span><span class="p">):</span>
        
        <span class="c1"># Training phase</span>
        <span class="n">train_loss</span><span class="p">,</span> <span class="n">train_acc</span> <span class="o">=</span> <span class="n">run_epoch</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="s1">&#39;Train&#39;</span><span class="p">)</span>
        
        <span class="c1"># Validation phase</span>
        <span class="n">val_loss</span><span class="p">,</span> <span class="n">val_acc</span> <span class="o">=</span> <span class="n">run_epoch</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="s1">&#39;Validation&#39;</span><span class="p">)</span>
           
        <span class="c1"># Print statistics after the validation phase</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;| </span><span class="si">{}</span><span class="se">\t</span><span class="s2"> | </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">| </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">| </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">| </span><span class="si">{:.4f}</span><span class="se">\t</span><span class="s2">|&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">train_loss</span><span class="p">,</span> <span class="n">train_acc</span><span class="p">,</span> <span class="n">val_loss</span><span class="p">,</span> <span class="n">val_acc</span><span class="p">))</span>

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



Now, let's train the model!



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">train</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">criterion</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">numEpochs</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Function which generates predictions, given a set of inputs</span>
<span class="k">def</span> <span class="nf">test</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">inputs</span><span class="p">,</span> <span class="n">device</span><span class="p">):</span>
  <span class="n">model</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>
  <span class="n">inputs</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">inputs</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
  
  <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">inputs</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">detach</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span>
  
  <span class="n">preds</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">outputs</span> <span class="o">&gt;</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
  
  <span class="k">return</span> <span class="n">preds</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="n">preds</span> <span class="o">=</span> <span class="n">test</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">xTest</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>
</pre></div>



Now that our model has made some predictions, let's find the mathew's 



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># import functions for matthews and confusion matrix</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">matthews_corrcoef</span>

<span class="n">matthews_corrcoef</span><span class="p">(</span><span class="n">preds</span><span class="p">,</span> <span class="n">yTest</span><span class="p">)</span>
</pre></div>



Let's check the confusion matrix



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">preds</span><span class="p">,</span> <span class="n">yTest</span><span class="p">)</span>
</pre></div>



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



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#TODO, make a better model!</span>

<span class="c1">### BEGIN SOLUTION</span>

<span class="n">inputSize</span> <span class="o">=</span>  <span class="mi">8</span>         <span class="c1"># how many classes of input</span>
<span class="n">hiddenSize</span> <span class="o">=</span> <span class="mi">15</span>        <span class="c1"># Number of units in the middle</span>
<span class="n">numClasses</span> <span class="o">=</span> <span class="mi">1</span>         <span class="c1"># Only has two classes</span>
<span class="n">numEpochs</span> <span class="o">=</span> <span class="mi">69</span>         <span class="c1"># How many training cycles</span>
<span class="n">learningRate</span> <span class="o">=</span> <span class="o">.</span><span class="mi">01</span>     <span class="c1"># Learning rate</span>

<span class="k">class</span> <span class="nc">NeuralNet</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">input_size</span><span class="p">,</span> <span class="n">hidden_size</span><span class="p">,</span> <span class="n">num_classes</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">NeuralNet</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">input_size</span><span class="p">,</span> <span class="n">hidden_size</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">hidden_size</span><span class="p">,</span> <span class="n">hidden_size</span><span class="p">)</span> 
        <span class="bp">self</span><span class="o">.</span><span class="n">fc3</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">hidden_size</span><span class="p">,</span> <span class="n">num_classes</span><span class="p">)</span>  
    
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc1</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc2</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc3</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">train</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">criterion</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">numEpochs</span><span class="p">,</span> <span class="n">dataloaders</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>

<span class="n">predictions</span> <span class="o">=</span> <span class="n">test</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">xTest</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>

<span class="c1">### END SOLUTION</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Run this to generate the submission file for the competition!</span>
<span class="c1">### Make sure to name your model variable &quot;model&quot; ###</span>

<span class="c1"># load in test data:</span>
<span class="n">test_data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">DATA_DIR</span> <span class="o">/</span> <span class="s2">&quot;test.csv&quot;</span><span class="p">,</span> <span class="n">header</span><span class="o">=</span><span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">values</span>
<span class="c1"># remove row with column labels:</span>
<span class="n">test_data</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">delete</span><span class="p">(</span><span class="n">test_data</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>

<span class="c1"># convert to float32 values</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">test_data</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>
<span class="c1"># get indicies for each entry in test data</span>
<span class="n">indicies</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">))]</span>

<span class="c1"># generate predictions</span>
<span class="n">preds</span> <span class="o">=</span> <span class="n">test</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>

<span class="c1"># create our pandas dataframe for our submission file. Squeeze removes dimensions of 1 in a numpy matrix Ex: (161, 1) -&gt; (161,)</span>
<span class="n">preds</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;Id&#39;</span><span class="p">:</span> <span class="n">indicies</span><span class="p">,</span> <span class="s1">&#39;Class&#39;</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">preds</span><span class="p">)})</span>

<span class="c1"># save submission csv</span>
<span class="n">preds</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;submission.csv&#39;</span><span class="p">,</span> <span class="n">header</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Id&#39;</span><span class="p">,</span> <span class="s1">&#39;Class&#39;</span><span class="p">],</span> <span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>


