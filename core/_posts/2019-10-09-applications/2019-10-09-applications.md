---
layout: "meeting"
title: "Applications"
date: "2019-10-09"
authors:
    - "jarviseq"
    
categories: ['fa19']
tags: ['neural-nets', 'applications', 'random-forest', 'svms']
description: >-
    You know what they are, but "how do?" In this meeting, we let you loose on a    dataset to help you apply your newly developed or honed data science skills.  Along the way, we go over the importance of visulisations and why it is  important to be able to pick apart a dataset.
---
{% raw %}  <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

  

  <!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration -->
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># This is a bit of code to make things work on Kaggle</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">from</span> <span class="nn">pathlib</span> <span class="k">import</span> <span class="n">Path</span>

<span class="k">if</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">exists</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-applications&quot;</span><span class="p">):</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-applications&quot;</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;./&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Dataset-for-the-day:-Suicide-Preventation">Dataset for the day: Suicide Preventation<a class="anchor-link" href="#Dataset-for-the-day:-Suicide-Preventation">&#182;</a></h1><h2 id="Slides"><a href="https://docs.google.com/presentation/d/1fzw2j1BJuP3Z-Y1noB4bcEkjFUak_PxIKjHBC9_vp6E/edit?usp=sharing">Slides</a><a class="anchor-link" href="#Slides">&#182;</a></h2><p>The <a href="https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016">dataset</a> we will be using today is socio-economic data alongside suicide rates per country from 1985 to 2016. It is your task today to try to predict the suicide rate per 100,000 people in a give country. Building a good model for this can help find areas where there might be a high suicide rate so that prevention measures can be put in place to help people before it happens.</p>
<p>We cannot natively use a SVM, Logistic Regression, or RF because they predict on categorical data, while today we will be making predictions on continuous data. Check out Regression Trees if you want to see these types of models applied to regression problems.</p>
<p>However, this problem can be changed to a categorical one so that you can use a RF or SVM. To do so, after the data analysis we will get some statistics on mean, min, max etc of suicide rate per 100,000 people column. Then, you can define ranges and assign them to a integer class. For example, assigning 0-5 suicides/100k as "Low", 5-50 as "High" etc. You can make as many ranges as you want, then train on those classes. In this case, we want to focus on producing actual values for this, so we will stick with regression, but this is something really cool you can try on your own! (Hint: use pandas dataframe apply function for this!)</p>
<p>Linear regression can work, although is a bit underwhelming for this task. So instead we will be using a Neural Network!</p>
<p>Let's dive in!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># import all the libraries you need</span>

<span class="c1"># torch for NNs</span>
<span class="kn">import</span> <span class="nn">torch</span> 
<span class="kn">import</span> <span class="nn">torch.nn</span> <span class="k">as</span> <span class="nn">nn</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="k">as</span> <span class="nn">F</span>
<span class="kn">from</span> <span class="nn">torch.utils.data</span> <span class="k">import</span> <span class="n">Dataset</span><span class="p">,</span> <span class="n">DataLoader</span>
<span class="kn">from</span> <span class="nn">torch</span> <span class="k">import</span> <span class="n">optim</span>

<span class="c1"># general imports</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Load-in-data-and-process">Load in data and process<a class="anchor-link" href="#Load-in-data-and-process">&#182;</a></h2><p>The data contains many different datatypes, such as floats, integers, strings, dates etc. We need to load this in and transform it all properly to something the models can understand. Once this is done, its up to you to build a model to solve this problem!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">DATA_DIR</span> <span class="o">/</span> <span class="s2">&quot;master.csv&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The first thing to notice is all the NaNs for HDI, meaning that there is missing data for those row entries. There is also a possibly useless row country-year as well. Lets see how many entires are missing HDI data first.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Total entries: </span><span class="si">{}</span><span class="s2">, null entries: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;HDI for year&quot;</span><span class="p">]),</span> <span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;HDI for year&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()))</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As you can see, most entires are null, so lets remove this column and the country-year column.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;HDI for year&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;country-year&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now that looks much better. We need to transform the categories that are objects (like sex, country, age ranges) to number representations. For example, sex will become <code>Male = 0</code> and <code>Female = 1</code>. The countries and age-ranges will be similiarly encoded to integer values. Then we can describe our data again and see the full stats.</p>
<p>This is done using dictionaries that map's these keys to values and apply that to the dataframe. The gdp_for_year however has commas in the numbers, so we need a function that can strip these and convert them to integers.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">country_set</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span><span class="nb">set</span><span class="p">(</span><span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;country&quot;</span><span class="p">]))</span>
<span class="n">country_map</span> <span class="o">=</span> <span class="p">{</span><span class="n">country</span> <span class="p">:</span> <span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">country</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">country_set</span><span class="p">)}</span>

<span class="n">sex_map</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;male&#39;</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span> <span class="s1">&#39;female&#39;</span><span class="p">:</span> <span class="mi">1</span><span class="p">}</span>

<span class="n">age_set</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span><span class="nb">set</span><span class="p">(</span><span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]))</span>
<span class="n">age_map</span> <span class="o">=</span> <span class="p">{</span><span class="n">age</span><span class="p">:</span> <span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">age</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">age_set</span><span class="p">)}</span>

<span class="n">gen_set</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span><span class="nb">set</span><span class="p">(</span><span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;generation&quot;</span><span class="p">]))</span>
<span class="n">gen_map</span> <span class="o">=</span> <span class="p">{</span><span class="n">gen</span><span class="p">:</span> <span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">gen</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">gen_set</span><span class="p">)}</span>

<span class="k">def</span> <span class="nf">gdp_fix</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="n">x</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">&quot;,&quot;</span><span class="p">,</span> <span class="s2">&quot;&quot;</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">x</span>

<span class="n">dataset</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">replace</span><span class="p">({</span><span class="s2">&quot;country&quot;</span><span class="p">:</span> <span class="n">country_map</span><span class="p">,</span> <span class="s2">&quot;sex&quot;</span><span class="p">:</span> <span class="n">sex_map</span><span class="p">,</span> <span class="s2">&quot;generation&quot;</span><span class="p">:</span> <span class="n">gen_map</span><span class="p">,</span> <span class="s2">&quot;age&quot;</span><span class="p">:</span> <span class="n">age_map</span><span class="p">})</span>
<span class="n">dataset</span><span class="p">[</span><span class="s2">&quot; gdp_for_year ($) &quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">row</span><span class="p">:</span> <span class="n">gdp_fix</span><span class="p">(</span><span class="n">row</span><span class="p">[</span><span class="s2">&quot; gdp_for_year ($) &quot;</span><span class="p">]),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now that is looking much better! However, as you can see the values vary pretty different, such as the year can be 1985-2016 and suicide rate could be from 0 to about 225. While you can train on this, its better if all of your data is within the same range. To do this, you would need to divide each value in the column, subtract its minimum value, then divide by its max value. This isn't required but sometimes can make your model train a lot faster and converge on a lower loss. For example on changing the range of year:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">((</span><span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;year&quot;</span><span class="p">]</span> <span class="o">-</span> <span class="mi">1985</span><span class="p">)</span> <span class="o">/</span> <span class="mi">31</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


  
<div class="output_area">

    


<div class="output_subarea output_text output_error">
<pre>
<span class="ansi-red-fg">---------------------------------------------------------------------------</span>
<span class="ansi-red-fg">NameError</span>                                 Traceback (most recent call last)
<span class="ansi-green-fg">&lt;ipython-input-1-e44eea9aecc7&gt;</span> in <span class="ansi-cyan-fg">&lt;module&gt;</span>
<span class="ansi-green-fg">----&gt; 1</span><span class="ansi-red-fg"> </span><span class="ansi-blue-fg">(</span>dataset<span class="ansi-blue-fg">[</span><span class="ansi-blue-fg">&#34;year&#34;</span><span class="ansi-blue-fg">]</span> <span class="ansi-blue-fg">-</span> <span class="ansi-cyan-fg">1985</span><span class="ansi-blue-fg">)</span> <span class="ansi-blue-fg">/</span> <span class="ansi-cyan-fg">31</span>

<span class="ansi-red-fg">NameError</span>: name &#39;dataset&#39; is not defined</pre>
</div>
</div>


</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here you need to split the data for input into your NN model.</p>
<p>If you using an NN, you need to use <code>torch.from_numpy()</code> to get torch tensors and use that to build a simple dataset class and dataloader. You'll also need to define the device to use GPU if you are using pytorch, check the previous lecture for how that works. The <a href="https://pytorch.org/docs/stable/index.html">pytorch documentation</a> is also a great resource!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;suicides/100k pop&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">values</span><span class="p">,</span> <span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;suicides/100k pop&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Split data here using train_test_split</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">xtrain</span><span class="p">,</span> <span class="n">xtest</span><span class="p">,</span> <span class="n">ytrain</span><span class="p">,</span> <span class="n">ytest</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>
<span class="c1">### END SOLUTION</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;X shape: </span><span class="si">{}</span><span class="s2">, Y shape: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="n">Y</span><span class="o">.</span><span class="n">shape</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># run this if you are using torch and a NN</span>
<span class="k">class</span> <span class="nc">Torch_Dataset</span><span class="p">(</span><span class="n">Dataset</span><span class="p">):</span>
  
  <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">outputs</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">data</span> <span class="o">=</span> <span class="n">data</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">outputs</span> <span class="o">=</span> <span class="n">outputs</span>

  <span class="k">def</span> <span class="nf">__len__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="c1">#&#39;Returns the total number of samples in this dataset&#39;</span>
        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">data</span><span class="p">)</span>

  <span class="k">def</span> <span class="nf">__getitem__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">index</span><span class="p">):</span>
        <span class="c1">#&#39;Returns a row of data and its output&#39;</span>
      
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">data</span><span class="p">[</span><span class="n">index</span><span class="p">]</span>
        <span class="n">y</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">outputs</span><span class="p">[</span><span class="n">index</span><span class="p">]</span>

        <span class="k">return</span> <span class="n">x</span><span class="p">,</span> <span class="n">y</span>

<span class="c1"># use the above class to create pytorch datasets and dataloader below</span>
<span class="c1"># REMEMBER: use torch.from_numpy before creating the dataset! Refer to the NN lecture before for examples</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">


<div class="inner_cell">
    <div class="input_area">
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Lets get this model!</span>
<span class="c1"># for your output, it will be one node, that outputs the predicted value. What would the output activation function be?</span>
<span class="c1">### BEGIN SOLUTION</span>


<span class="c1">### END SOLUTION</span>
</pre></div>

    </div>
</div>
</div>

</div>
  
 


{% endraw %}