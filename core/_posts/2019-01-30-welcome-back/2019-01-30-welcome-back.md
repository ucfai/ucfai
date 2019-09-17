---
title: "Welcome back! Featuring Supercomputers"
categories: ["sp19"]
authors: ['ionlights']
description: >-
  "Welcome back to SIGAI! We'll be re-introducing AI@UCF for newcomers and refreshing it for veterans; following that, we'll cover some club logistics, revealing our plans for the semester, and finish off with setting everyone up on the UCF Supercomputer, from which we'll be streaming all future meetings!"
---

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Description:</strong> Welcome back to SIGAI! We'll be re-introducing AI@UCF for newcomers and
    refreshing it for veterans; following that, we'll cover some club logistics,
    reveal our plans for the semester, and finish off with setting everyone
    up on the UCF Supercomputer, from which we'll be streaming all future
    meetings!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><a href="https://docs.google.com/presentation/d/1ao6UhsjFQwxcP2Go5j7fv6c8RmsnnEL7AyPwHZKcPng/edit"><strong>View the Slide Deck</strong></a></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Notebook-Overhead">Notebook Overhead<a class="anchor-link" href="#Notebook-Overhead">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As normal, we'll import the majority of our needed libraries here. <em>Don't worry if this looks like gibberish, you'll become familiar with this over the course of the semester.</em></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">tensorflow</span> <span class="k">as</span> <span class="nn">tf</span>

<span class="kn">from</span> <span class="nn">tensorflow</span> <span class="k">import</span> <span class="n">keras</span>
<span class="kn">from</span> <span class="nn">tensorflow.keras.layers</span> <span class="k">import</span> <span class="n">Dense</span>
<span class="kn">from</span> <span class="nn">tensorflow.keras</span> <span class="k">import</span> <span class="n">Sequential</span>

<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">display</span><span class="p">(</span><span class="n">f</span><span class="s2">&quot;TensorFlow: v</span><span class="si">{tf.VERSION}</span><span class="s2">; Keras: v</span><span class="si">{keras.__version__}</span><span class="s2">&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_text output_subarea ">
<pre>&#39;TensorFlow: v1.12.0; Keras: v2.1.6-tf&#39;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now, we'll setup a relatively straightforward Neural Network takes in random data and classifies it – ultimately, this isn't the purpose of this notebook. (This is notebook's primarily used as a tool to introduce Colab, JupyterHub, and self-supported usage of our lectures.)</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Creating-the-Model">Creating the Model<a class="anchor-link" href="#Creating-the-Model">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Below, we're creating a <code>Sequential</code> model using Keras. Just think of this as an assembly line of "Layers" – no worries, you'll be learning about these concepts in the next month from the different Neural Network lectures.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">Sequential</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s2">&quot;relu&quot;</span><span class="p">))</span>
<span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s2">&quot;relu&quot;</span><span class="p">))</span>
<span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s2">&quot;softmax&quot;</span><span class="p">))</span>

<span class="n">model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">optimizer</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">AdamOptimizer</span><span class="p">(</span><span class="mf">0.001</span><span class="p">),</span>
              <span class="n">loss</span><span class="o">=</span><span class="s1">&#39;categorical_crossentropy&#39;</span><span class="p">,</span>
              <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;accuracy&#39;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>All Machine Learning is based on data... so, we have some randomly generated data to demo the Network and make sure that everything works as expected.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">data</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">random</span><span class="p">((</span><span class="mi">1000</span><span class="p">,</span> <span class="mi">32</span><span class="p">))</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">random</span><span class="p">((</span><span class="mi">1000</span><span class="p">,</span> <span class="mi">10</span><span class="p">))</span>

<span class="n">val_data</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">random</span><span class="p">((</span><span class="mi">100</span><span class="p">,</span> <span class="mi">32</span><span class="p">))</span>
<span class="n">val_labels</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">random</span><span class="p">((</span><span class="mi">100</span><span class="p">,</span> <span class="mi">10</span><span class="p">))</span>

<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">,</span>
          <span class="n">validation_data</span><span class="o">=</span><span class="p">(</span><span class="n">val_data</span><span class="p">,</span> <span class="n">val_labels</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Train on 1000 samples, validate on 100 samples
Epoch 1/10
1000/1000 [==============================] - 1s 848us/step - loss: 11.5737 - acc: 0.1020 - val_loss: 11.5987 - val_acc: 0.1000
Epoch 2/10
1000/1000 [==============================] - 0s 91us/step - loss: 11.5305 - acc: 0.1030 - val_loss: 11.5905 - val_acc: 0.1000
Epoch 3/10
1000/1000 [==============================] - 0s 88us/step - loss: 11.5215 - acc: 0.1080 - val_loss: 11.5893 - val_acc: 0.0900
Epoch 4/10
1000/1000 [==============================] - 0s 96us/step - loss: 11.5149 - acc: 0.1090 - val_loss: 11.5949 - val_acc: 0.1100
Epoch 5/10
1000/1000 [==============================] - 0s 98us/step - loss: 11.5113 - acc: 0.1150 - val_loss: 11.5923 - val_acc: 0.0900
Epoch 6/10
1000/1000 [==============================] - 0s 90us/step - loss: 11.5047 - acc: 0.1330 - val_loss: 11.5956 - val_acc: 0.1300
Epoch 7/10
1000/1000 [==============================] - 0s 84us/step - loss: 11.4998 - acc: 0.1290 - val_loss: 11.5921 - val_acc: 0.1000
Epoch 8/10
1000/1000 [==============================] - 0s 81us/step - loss: 11.4943 - acc: 0.1380 - val_loss: 11.5950 - val_acc: 0.1000
Epoch 9/10
1000/1000 [==============================] - ETA: 0s - loss: 11.4959 - acc: 0.15 - 0s 88us/step - loss: 11.4920 - acc: 0.1550 - val_loss: 11.5925 - val_acc: 0.1300
Epoch 10/10
1000/1000 [==============================] - 0s 85us/step - loss: 11.4881 - acc: 0.1650 - val_loss: 11.5967 - val_acc: 0.1100
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;tensorflow.python.keras.callbacks.History at 0x143febc88&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>It's also paramount that you evaluate your model to determine how well it's doing and this will also become the deliverable of your work when you do research and build projects of your own.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">data</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">random</span><span class="p">((</span><span class="mi">1000</span><span class="p">,</span> <span class="mi">32</span><span class="p">))</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">random</span><span class="p">((</span><span class="mi">1000</span><span class="p">,</span> <span class="mi">10</span><span class="p">))</span>

<span class="n">model</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>1000/1000 [==============================] - 0s 42us/step
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[11.54239518737793, 0.089]</pre>
</div>

</div>

</div>
</div>

</div>
 

