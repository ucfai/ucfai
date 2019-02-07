---
title: "Starting with the Basics, Regression"
categories: ["sp19"]
authors: ['JarvisEQ', 'causallycausal', 'ionlights']
description: >-
  "You always start with the basics, and with Data Science it's no different! We'll be getting our feet wet with some simple, but powerful, models and  demonstrate their power by applying them to real world data."
---

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Die-Pr&#228;sentation"><strong><a href="https://docs.google.com/presentation/d/12MvqRMZlKL3DwqAX1XqQMaL21UUV29hLkEcYO3liHfs/edit?usp=sharing">Die Pr&#228;sentation</a></strong><a class="anchor-link" href="#Die-Pr&#228;sentation">&#182;</a></h2><hr>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Linear-Regression"><strong>Linear Regression</strong><a class="anchor-link" href="#Linear-Regression">&#182;</a></h2><hr>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>First thing first, we to get some packages</p>
<ul>
<li>matplotlib allows us to graph </li>
<li>numpy is powerful package for data manipulation</li>
<li>pandas is a tool for allowing us to interact with large datasets</li>
<li>sklearn is what we'll use for making the models</li>
<li>!wget grabs the data set we'll be using later</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#import some stuff</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">scipy.stats</span> <span class="k">as</span> <span class="nn">st</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="k">import</span> <span class="n">datasets</span><span class="p">,</span> <span class="n">linear_model</span>
<span class="o">!</span>wget <span class="s2">&quot;https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data&quot;</span> 
<span class="o">!</span>wget <span class="s2">&quot;http://jse.amstat.org/v19n3/decock/AmesHousing.txt&quot;</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>--2019-02-06 19:53:49--  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
Resolving archive.ics.uci.edu (archive.ics.uci.edu)... 128.195.10.249
Connecting to archive.ics.uci.edu (archive.ics.uci.edu)|128.195.10.249|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 3974305 (3.8M) [text/plain]
Saving to: ‘adult.data.6’

adult.data.6        100%[===================&gt;]   3.79M  4.91MB/s    in 0.8s    

2019-02-06 19:53:50 (4.91 MB/s) - ‘adult.data.6’ saved [3974305/3974305]

--2019-02-06 19:53:51--  http://jse.amstat.org/v19n3/decock/AmesHousing.txt
Resolving jse.amstat.org (jse.amstat.org)... 107.180.48.28
Connecting to jse.amstat.org (jse.amstat.org)|107.180.48.28|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 960807 (938K) [text/plain]
Saving to: ‘AmesHousing.txt.6’

AmesHousing.txt.6   100%[===================&gt;] 938.29K  1.24MB/s    in 0.7s    

2019-02-06 19:53:52 (1.24 MB/s) - ‘AmesHousing.txt.6’ saved [960807/960807]

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The data for this example is arbitrary (we'll use real data in a bit), but there is a clear linear relationship here</p>
<p>Graphing the data will make this relationship clear to see</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Get some data </span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">7</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">9</span><span class="p">])</span> 
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">7</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">9</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">12</span><span class="p">])</span>

<span class="c1">#Let&#39;s plot the data to see what it looks like</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s2">&quot;black&quot;</span><span class="p">,</span> 
               <span class="n">marker</span> <span class="o">=</span> <span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="mi">30</span><span class="p">)</span> 
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here's the meat of the calculations</p>
<p>This is using least squares estimation, which tries to minimize the squared error of the function vs. the training data</p>
<p>SS_xy is the cross deviation about x, and SS_xx is the deviation about x</p>
<p><a href="https://www.amherst.edu/system/files/media/1287/SLR_Leastsquares.pdf">It's basically some roundabout algebra methods to optimize a function</a></p>
<p>The concept isn't super complicated but it gets hairy when you do it by hand</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#calculating the coefficients</span>

<span class="c1"># number of observations/points </span>
<span class="n">n</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> 

<span class="c1"># mean of x and y vector </span>
<span class="n">m_x</span><span class="p">,</span> <span class="n">m_y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">y</span><span class="p">)</span> 

<span class="c1"># calculating cross-deviation and deviation about x </span>
<span class="n">SS_xy</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">y</span><span class="o">*</span><span class="n">x</span> <span class="o">-</span> <span class="n">n</span><span class="o">*</span><span class="n">m_y</span><span class="o">*</span><span class="n">m_x</span><span class="p">)</span> 
<span class="n">SS_xx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">x</span><span class="o">*</span><span class="n">x</span> <span class="o">-</span> <span class="n">n</span><span class="o">*</span><span class="n">m_x</span><span class="o">*</span><span class="n">m_x</span><span class="p">)</span> 

<span class="c1"># calculating regression coefficients </span>
<span class="n">b_1</span> <span class="o">=</span> <span class="n">SS_xy</span> <span class="o">/</span> <span class="n">SS_xx</span> 
<span class="n">b_0</span> <span class="o">=</span> <span class="n">m_y</span> <span class="o">-</span> <span class="n">b_1</span><span class="o">*</span><span class="n">m_x</span>

<span class="c1">#var to hold the coefficients</span>
<span class="n">b</span> <span class="o">=</span> <span class="p">(</span><span class="n">b_0</span><span class="p">,</span> <span class="n">b_1</span><span class="p">)</span>

<span class="c1">#print out the estimated coefficients</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Estimated coefficients:</span><span class="se">\n</span><span class="s2">b_0 = </span><span class="si">{}</span><span class="s2"> </span><span class="se">\n</span><span class="s2">b_1 = </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">b</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">b</span><span class="p">[</span><span class="mi">1</span><span class="p">]))</span> 
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Estimated coefficients:
b_0 = -0.0586206896552 
b_1 = 1.45747126437
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>But, we don't need to directly program all of the maths everytime we do linear regression</p>
<p>sklearn has built in functions that allows you to quickly do Linear Regression with just a few lines of code</p>
<p>We're going to use sklearn to make a model and then plot it using matplotlib</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#we need to reshape the array to make the sklearn gods happy</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">y</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>

<span class="c1">#making the model</span>
<span class="n">regress</span> <span class="o">=</span> <span class="n">linear_model</span><span class="o">.</span><span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">regress</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
<span class="n">y_sk_pred</span> <span class="o">=</span> <span class="n">regress</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mi">6</span><span class="p">]])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>And now lets see what the model looks like</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plotting the actual points as scatter plot </span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s2">&quot;black&quot;</span><span class="p">,</span> 
           <span class="n">marker</span> <span class="o">=</span> <span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="mi">30</span><span class="p">)</span> 

<span class="c1"># predicted response vector </span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">b</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">+</span> <span class="n">b</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">*</span><span class="n">x</span> 

<span class="c1"># plotting the regression line </span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s2">&quot;blue&quot;</span><span class="p">)</span> 

<span class="c1"># putting labels </span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;x&#39;</span><span class="p">)</span> 
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;y&#39;</span><span class="p">)</span> 

<span class="c1"># function to show plot </span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAe0AAAFYCAYAAAB+s6Q9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4zLCBo
dHRwOi8vbWF0cGxvdGxpYi5vcmcvIxREBQAAIABJREFUeJzt3XlgVOXd9vHv7GcmoCAEKS641eUB
N0Q2AVE2cQepQIxia6tWJCxBCDGKSpWABEKCChXhqbKUFhGwWlEKSF5NwAhFcQcUERQChC2ZmSwz
7x9TeaQqm5lzZrk+f9UYz/3zdso11zlnztjC4XAYERERiXl2qwcQERGRY6PQFhERiRMKbRERkTih
0BYREYkTCm0REZE4odAWERGJE06rBziSsrIDdX7Mhg19lJdX1vlx5XDaZ3Non82hfTaH9jkiNbX+
z/69pGvaTqfD6hGSgvbZHNpnc2ifzaF9PrqkC20REZF4pdAWERGJEwptERGROKHQFhERiRMKbRER
kTih0BYREYkTCm0REZE4odAWERGJEwptERGROKHQFhERiRMKbRERkROwebONJUuchMPmrRnTXxgi
IiISa2prYfp0F+PGeaiuhs8+O8jJJ5uztkJbRETkGG3aZCMjw8t77zlo3DhEXl7QtMAGnR4XERE5
qlAo0q6vuSaF995zcOut1RQVVdKrV42pc6hpi4iIHMHmzTaGDjUoKXHSqFGIZ54JcNNN5ob199S0
RUREfkIoBDNmRNp1SYmTG2+sZtWqSssCG9S0RUREfuSrryLt+t13nZxySogpUwLccksNNpu1c6lp
i4iI/EcoBDNnuujSJYV333Vy/fWRdn3rrdYHNqhpi4iIAPD11zaGDTMoKnLSoEGYvDw/ffrERlh/
T01bRESSWjgMf/mLi6uvTqGoyEnPnjUUFVVw222xFdigpi0iIkls69ZIu161ysnJJ4eZOtXPb34T
e2H9PYW2iIgknXAY5sxx8eijHg4etNGtWw2TJgVo2tTEZ5KeAIW2iIgkle3bI+16xQon9euHKSjw
069f7LbrH1Joi4hIUgiH4a9/dZKTY3DggI1rr42062bNYrtd/5BCW0REEt6339rIzDRYtsxJvXph
Jk8OkJZWHRft+ocU2iIikrDCYfjb35w8/LDB/v02OneuIT8/wOmnx0+7/iGFtoiIJKQdO2yMGGGw
dKmTlJQwEycGuPPO+GvXP6TQFhGRhBIOw8svO8nONti710anTjVMnhzgzDPjs13/kEJbREQSxs6d
Nh56yMM//+nC5wuTmxvg7rursSfIo8QU2iIiEvfCYVi0yMno0R727LHToUPk2vVZZ8V/u/6hqL73
+Pzzz+nWrRuzZ88G4Ntvv+Xuu+8mPT2du+++m7KysmguLyIiSaCszMY99xjcd5+XQMDGuHEBFi70
Rz2wS0vXMHVqPqWla6K6zg9FLbQrKysZO3Ys7du3P/Sz/Px8br/9dmbPnk337t2ZNWtWtJYXEZEk
sGSJk86dffzjHy7atq1h+fIK7rkn+qfDs7Iy6dPnJp544lH69LmJrKzM6C74H1H713K73Tz//PM0
adLk0M/GjBlDz549AWjYsCF79+6N1vIiIpLAdu+28Yc/GPz+914qK22MHRtg8WI/55wT/dPhpaWr
mTt3NoGAH4BAwM+8eXNMadxRu6btdDpxOg8/vM/nA6C2tpa5c+cyaNCgIx6jYUMfTqejzmdLTa1f
58eUH9M+m0P7bA7tszmOZZ8XLoQ//hF27oQOHWDWLBvnn28ARvQHBDZsWHcosL/n91eyYcNaevXq
GtW1Tb8Rrba2lpEjR9KuXbvDTp3/lPLyyjpfPzW1PmVlB+r8uHI47bM5tM/m0D6b42j7vGcPZGcb
LFzowuMJ8/jjQe69txqHA8y8Raply1YYhvew4PZ6fbRs2apOXidHeuNi+k3wo0ePpnnz5jz44INm
Ly0iInHqn/900qlTCgsXurjiilqWL6/kj3+MBLbZWrduQ1paOobhBSKBPWDAHbRu3Sbqa5vatJcs
WYLL5SIjI8PMZUVEJE6Vl8PDDxssWBBp1488EuSBB6osCesfys3No2/ffqxeXUzbtu1NCWwAWzgc
jspV+w0bNjB+/Hi2bduG0+nk1FNPZffu3Xg8HurVqwfAueeey2OPPfazx4jG6Sid5jKH9tkc2mdz
aJ/N8d/7/OabDjIzDXbssHP55bUUFAS44IKQhROa40inx6PWtFu2bMlLL70UrcOLiEiC2rcPcnIM
5s934XKFefjhIIMGVeHU48D0RDQREYkdy5Y5GD7c4Lvv7Fx6aaRdX3RR4rfrY5UgT2MVEZF4tn8/
3HMPpKX52L3bRlZWkNdfr1Rg/xc1bRERsdSKFQ6GDTPYvh1atqylsDBAixYK65+i0BYREUscOACP
PebhpZfcOJ1hxoyBe++txOWyerLYpdAWERHTvf12pF1/842d//mfSLu+9toUUx+SEo8U2iIiYpqD
B+Hxxz385S9uHI4ww4cHGT68Crfb6snig0JbRERM8f/+n4OhQw2+/trOhRdG2vWll+ra9fFQaIuI
SFQdPAh/+pOHmTPd2O1hhg4NkplZhcdj9WTxR6EtIiJRU1zsICPDYMsWOxdcEPnc9eWXq12fKIW2
iIjUuYoKeOopD88/H2nXgwcHeeihKgxzvj0zYSm0RUSkTpWUOBgyxODLL+2cd16kXbdurXZdFxTa
IiJSJ/z+SLv+858jH7R+4IEqRo0K4vVaPFgCUWiLiMgv9t57djIyvGzaZOecc0IUFPhp00btuq4p
tEVE5IT5/TB+vIdp01yEw3DffVWMHh3E57N6ssSk0BYRkRPy/vt2MjIMvvjCwVlnhSgoCNCuXa3V
YyU0hbaIiByXYBCeftrN1KluQiEbf/hDFdnZQVJSrJ4s8Sm0RUTkmK1bF2nXn33moHnzEFOm+OnQ
Qe3aLAptERE5qmAQJk1yU1DgprbWxu9+V0VOTpB69ayeLLkotEVE5Ig++MDO4MEGn3zi4IwzQuTn
++nUSe3aCnarBxARkdhUVQXjx7vp2dPHJ584GDiwirffrlBgW0hNW0REfuTDDyPXrj/6yMFpp4WY
PNlPly4Ka6upaYuIyCHV1TBxYqRdf/SRg/T0KlatqlBgxwg1bRERAeDjjyPXrj/80EGzZiEmTfJz
7bUK61iipi0ikuRqamDyZDfdu/v48EMHaWmRdq3Ajj1q2iIiSezTTyPtev16B02bRtp1t24K61il
pi0ikoRqaqCgwE23bj7Wr3dw++3VrFpVocCOcWraIiJJ5vPPI3eGr13roEmTEHl5fnr2VFjHAzVt
EZEkUVsLU6e66NrVx9q1Dvr2raaoqEKBHUfUtEVEksDGjTYyMryUljpo3DjE9OkBrr++xuqx5Dip
aYuIJLDaWnjuORfXXptCaamD3r2rKSqqVGDHKTVtEZEEtXmzjYwMgzVrnDRuHOKZZwLcdJPCOp6p
aYuIJJhQCP78ZxfXXJPCmjVObr65mlWrKhXYCSCqof3555/TrVs3Zs+eDcC3337LnXfeSVpaGkOG
DKGqqiqay4uIJJ0vv7TRu7eXnBwDrzfM88/7mTEjQOPGYatHkzoQtdCurKxk7NixtG/f/tDPCgoK
SEtLY+7cuTRv3pwFCxZEa3kRkaQSCsELL0TadXGxk+uvj7TrW26Jj3ZdWrqGp59+mtLSNVaPEtOi
Ftput5vnn3+eJk2aHPrZ6tWr6dq1KwDXXHMNxcXF0VpeRCRpbNli47bbvIwebeDxwPTpfmbNCtCk
SXy066ysTPr0uYmRI0fSp89NZGVlWj1SzIpaaDudTgzDOOxnfr8ft9sNQKNGjSgrK4vW8iIiCS8U
glmzXFx9dQrvvOPkuusiTzXr3bsGm83q6Y5Naelq5s6dTSDgByAQ8DNv3hw17p9h2d3j4fDR3wE2
bOjD6XTU+dqpqfXr/JjyY9pnc2ifzRFr+7xlC9xzD/zrX9CwIUyfDnfc4cJmc1k92nHZsGHdocD+
nt9fyYYNa+nVq6tFU8UuU0Pb5/MRCAQwDIMdO3Ycdur8p5SXV9b5DKmp9SkrO1Dnx5XDaZ/NoX02
RyztczgMs2e7ePRRDxUVNnr0qGHixABNm4bZtcvq6Y5fy5atMAzvYcHt9fpo2bJVzOy52Y70BtHU
j3x16NCBpUuXAvDmm2/SqVMnM5cXEYlr27bZ6NfPS2amgcMBhYV+XnrJT9Om8XHt+qe0bt2GtLR0
DMMLRAJ7wIA7aN26jcWTxSZb+FjOU5+ADRs2MH78eLZt24bT6eTUU09l4sSJZGVlEQwGadasGePG
jcPl+vlTOdF4lxVL75gTmfbZHNpnc1i9z+EwzJ0badcHDtjo2rWGSZMC/OpX8RvW/620dA0bNqyl
ZctWSR/YR2raUQvtuqDQjl/aZ3Non81h5T5v325j+HCD5cud1K8fZuzYAAMGxM+NZsdDr+eII4W2
HmMqIhKDwmGYP99JTo7B/v02unSpYfLkAKedFrM9S0yg0BYRiTHffWcjM9Pgrbec1KsXJi8vQHp6
dUK2azk+Cm0RkRgRDsOCBU6ysw327bPRqVMN+fkBzjhD7VoiFNoiIjFgxw4bDz3k4Y03XPh8YSZM
CDBwoNq1HE6hLSJioXAYXnnFyejRBuXlNjp2jFy7bt5c7Vp+TKEtImKRsjIbI0d6eO21SLseNy7A
b39bjV1fmiw/Q6EtImKBxYudZGV52L3bTrt2NUyZEuDss9Wu5cgU2iIiJtq1y0ZWloclS1x4vWGe
fDLAPfeoXcuxUWiLiJjk1VedjBrlYdcuO23a1FBQEOCcc9Su5dgptEVEomzPHhg92uCVV1wYRpjH
Hw9w773VOOr+SwwlwSm0RUSi6PXXnTz0kIeyMjtXXFFLYaGf885Tu5YTo9AWEYmC8nLIzjZ4+WUX
Hk+YMWMC3H+/2rX8MgptEZE6tnSpg8xMg5077bRqVUtBQYDzzw9ZPZYkAIW2iEgd2bsXcnIM/vY3
F253mJycIA88UIXT5D9pS0vXUFLyLu3adUj6r7lMNAptEZE6sGyZg+HDDb77zs6ll9ZSWBjgwgvN
b9dZWZnMnTubQMCPYXhJS0snNzfP9DkkOvTJQBGRX2D/fhgyxCAtzcfu3TZGjw7y+uuVlgR2aenq
Q4ENEAj4mTdvDqWla0yfRaJDoS0icoKWL3fQuXMK8+a5uPjiWt58s5Jhw6pwuayZp6Sk+FBgf8/v
r2T16mJrBpI6p9AWETlOBw7A8OEe+vf3sXOnjZEjg7zxRiUtWlh7s1m7dh0wDO9hP/N6fbRt296i
iaSuKbRFRI7D229H2vXs2W7+539qWbq0khEjrGvXP9S6dRvS0tIPBbfX62PAgDt0M1oC0Y1oIiLH
4OBBeOwxDy++6MbhCJOZGWTYsCrcbqsnO1xubh59+/Zj9epi2rZtr8BOMAptEZGjWL4c7r47ha1b
7Vx0UeRz15deGrufu27duo3COkHp9LiIyM84eBCysjx07Qrbt9sYNizIm29WxnRgS2JT0xYR+Qnv
vusgI8Pg66/tXHQR5OdXcvnlCmuxlpq2iMgPVFTAww97uPVWH998Y2Pw4CBr16LAlpigpi0i8h8l
JZF2/dVXdn7968i16yuuCGEYHg4csHo6EYW2iAiVlTBunIc//znyua1Bg6oYOTKI13uUf1DEZApt
EUlqa9bYycjwsnmznXPOCVFQ4KdNG50Kl9ik0BaRpOT3w/jxHp57LtKu77uvitGjg/h8Fg8mcgQK
bRFJOqWldjIyDDZudHD22SGmTAnQrl2t1WOJHJVCW0SSRiAATz/t5pln3IRCNu69t4rsbLVriR8K
bRFJCuvWRdr1Z585aN48xJQpfjp0ULuW+KLQFpGEFgxCXp6bwkI3tbU2fve7KnJygtSrZ/VkIsdP
oS0iCWv9+ki7/uQTB2eeGSI/30/HjmrXEr9MDe2KigpGjRrFvn37qK6uZtCgQXTq1MnMEUTEQqWl
aygpeZd27TpE9Qstqqpg0iQ3U6ZE2vXAgVWMGXNi7bq0dA0bNqylZctW+hIOsZypof3KK69w9tln
k5mZyY4dOxg4cCBvvPGGmSOIiEWysjKZO3c2gYAfw/CSlpZObm5ena/z4Yd2Bg82+PhjB6efHmLy
ZD9XX31i7dqsmUWOlanPHm/YsCF79+4FYP/+/TRs2NDM5UXEIqWlqw+FH0Ag4GfevDmUlq6pszWq
qyN3hvfs6ePjjx3ceWcVb79dccKBbcbMIsfL1KZ9ww03sHDhQrp3787+/fuZPn36EX+/YUMfTqej
zudITa1f58eUH9M+myMe9nnDhnWHwu97fn8lGzaspVevrr/4+B98AAMHwr//DaefDjNmQM+ebsB9
wseM9szy0+Lh9WwlU0N78eLFNGvWjBdeeIFPP/2U7OxsFi5c+LO/X15eWeczpKbWp6xMT/6PNu2z
OeJln1u2bIVheA8LQa/XR8uWrX7R/NXVUFjoJi/PTXW1jbS0Kp54IshJJ0FZWWzOLD8vXl7P0Xak
Ny6mnh5fu3YtHTt2BODCCy9k586d1NbqTk6RRNe6dRvS0tIxjMg3cHi9PgYMuOMX3dj1ySd2rr/e
R26uh0aNwsydW0l+fiSwY3VmkV/K1KbdvHlz1q9fT8+ePdm2bRspKSk4HHV/+ltEYk9ubh59+/Zj
9epi2rZtf8LhV1MDzzzj5umn3VRV2ejXr5qxYwM0aFDHA/N/M+vucYkVtnA4HDZrsYqKCrKzs9m9
ezc1NTUMGTKE9u3b/+zvR+M0iU6/mEP7bI5k2+fPPot87nrdOgennhoiLy9Ajx7RP1uXbPtsFe1z
xJFOj5vatFNSUpgyZYqZS4pIAqithWefdTNhgptg0EbfvtU8+WQAfQBFko2eiCYiMe2LLyLt+v33
HaSmhpg4MUCvXjVWjyViCVNvRBMROVaRdu3i2mt9vP++gz59qikqqlBgS1JT0xaRmLNpk42MDC/v
veegceMQzz0X4MYbFdYiatoiEjNCIZg+3cU116Tw3nsObrmlmlWrKhXYIv+hpi0iMWHzZhtDhxqU
lDhp1CjE1KkBbr5ZYS3yQ2raImKpUAhmzIi065ISJzfeGGnXCmyRH1PTFhHLfPVVpF2/+66Thg3D
5Of7ufXWGmw2qycTiU1q2iJiulAIZs500aVLCu++6+S666pZtaqC3r0V2CJHoqYtIqb6+msbw4YZ
FBU5adAgzMSJfm67TWEtcizUtEXEFOEw/OUvLq6+OoWiIic9e9ZQVFRB374KbJFjpaYtIlH3zTeR
dv32205OOilMYaGf229XWIscL4W2iERNOAxz5rh49FEPBw/a6Nathry8AL/6lWnfUySSUBTaInGq
tHRNTH9l5PbtkXa9YoWT+vXDTJnip39/tWuRX0KhLRKHsrIymTt3NoGAH8PwkpaWTm5untVjAZF2
/de/OnnkEYP9+21cc00NkyYFOO00tWuRX0o3oonEmdLS1YcCGyAQ8DNv3hxKS9dYPBl8+62NO+7w
MmSIl1AIJk0K8Ne/+hXYInVEoS0SZ0pKig8F9vf8/kpWry62aKJIu54/30nnziksW+akc+caVq2q
ID29WqfDReqQTo+LxJl27TpgGN7Dgtvr9dG2bXtL5tmxw8aIEQZLlzpJSQnz9NMB7rpLYS0SDWra
InGmdes2pKWlYxheIBLYAwbcYfrNaOEwLFjgpFOnFJYuddKxYw1vv13BwIEKbJFoUdMWiUO5uXn0
7dvPsrvHd+608dBDHv75Txc+X5jc3AB3312NXTVAJKoU2iJxqnXrNvTq1ZWysgOmrRkOw+LFTrKy
POzZY6dDhxry8wOcdZZuNBMxg0JbRI5JWZmNUaM8/OMfLrzeME89FeB3v1O7FjGTQltEjmrJEiej
RnnYvdtO27Y1TJkS4Jxz1K5FzKbQFpGftXu3jdGjPSxa5MIwwowdG+D3v6/G4bB6MpHkdNQTW6tW
rTJjDhGJMa+95qRTJx+LFrm48spaVqyo4L77FNgiVjpqaL/00kt0796dgoICtm3bZsZMImKhPXvg
/vsNfvtbLwcO2HjssQBLllRy7rk6HS5itaOeHn/++efZt28fb731Fo899hgAffr0oUePHjj0llsk
obzxhoPMTIOyMjtXXFFLQUGAX/86ZPVYIvIfx3Tf58knn8wNN9zAjTfeyIEDB5g5cya33HIL//73
v6M9n4iYYO9eGDTI4K67fOzbZ+ORR4K8+mqlAlskxhy1ab/33nssXLiQ1atX0717d5588knOPfdc
vvnmGx588EEWLVpkxpwiEiVvvhlp1zt22Ln88ki7vuAChbVILDpqaE+aNIn+/fvz+OOP43a7D/38
9NNPp1evXlEdTkSiZ98+yMkxmD/fhcsV5uGHgwwaVIVTnykRiVlH/b/nvHnzfvbv3XfffXU6jIiY
41//cjB8uMG339q55JJaCgsDXHSR2rVIrNN7apEksn8/jBnjYc4cNy5XmKysIIMHV+FyWT2ZiBwL
hbZIklixwsGwYQbbt9tp2TLSrlu0ULsWiSemPzV4yZIl3HzzzfTp04eVK1eavbxI0jl4EDIzPfTr
5/vPt3MFWbq0UoEtEodMbdrl5eU888wzvPzyy1RWVlJYWEiXLl3MHEEkqaxaFWnXW7faueiiWqZO
DXDxxQprkXhlamgXFxfTvn176tWrR7169Rg7dqyZy4skjYMH4YknPPzv/7pxOMIMHx5k+PAqfvAB
EBGJQ6aG9jfffEMgEOD+++9n//79DB48mPbt25s5gkjCe+cdB0OGGHz9tZ0LL4xcu770UrVrkURg
+o1oe/fuZerUqWzfvp277rqLFStWYLPZfvJ3Gzb04XTW/aNSU1Pr1/kx5ce0z+b4fp8rKiArC6ZO
BbsdsrPh0UcdeDwpFk+YGPR6Nof2+chMDe1GjRpx+eWX43Q6OfPMM0lJSWHPnj00atToJ3+/vLyy
zmdITa1PWdmBOj+uHE77bI7v97m42EFGhsGWLXbOPz/yVLNWrULs32/1hIlBr2dzaJ8jjvTGxdS7
xzt27EhJSQmhUIjy8nIqKytp2LChmSOIJJTKSsjJ8XDrrV62brUxeHCQZcsqadVKp8NFEpGpTfvU
U0+lZ8+e3H777QDk5ORgt5v+qTORhLB6tYNhw2DjRjfnnRdp161bK6xFEpnp17T79+9P//79zV5W
JGH4/TBunIfp0yOPMXvggSpGjQri9Vo8mIhEnZ6IJhJH3nvPTkaGl02b7JxzTogXX7Rx/vlBq8cS
EZPo3LRIHAgE4PHHPdx0k4/Nm23cd18Vy5dXcNVVVk8mImZS0xaJcWvX2snIMPj8cwdnnRWioCBA
u3a1Vo8lIhZQaIvEqGAQnn7azdSpbkIhG7//fRUPPxwkRR+7FklaCm2RGPTvf0fa9aefOjjzzBBT
pvi56iq1a5Fkp9AWiSHBIEya5KagwE1trY3f/a6KnJwg9epZPZmIxAKFtkiM+OADO4MHG3zyiYMz
zgiRn++nUye1axH5P7p7XMRiVVUwfryb667z8cknDu66q4q3365QYIvIj6hpi1how4ZIu/7oIwen
nRZi8mQ/XboorEXkp6lpi1iguhry8tz06OHjo48cpKdXsWpVhQJbRI5ITVvEZB9/HLkz/IMPHPzq
V5F2fe21CmsROTo1bRGT1NTA5Mluunf38cEHDgYMqGbVqgoFtogcMzVtEaC0dA0lJe/Srl0HWrdu
U+fH//TTSLv+978dNG0aYtIkP926KaxF5PgotCXpZWVlMnfubAIBP4bhJS0tndzcvDo5dk0NPPus
mwkT3FRV2bj99mr+9KcADRrUyeFFJMno9LgktdLS1YcCGyAQ8DNv3hxKS9f84mN//rmdG2/08ac/
eWjQIMxLL1UydaoCW0ROnEJbklpJSfGhwP6e31/J6tXFJ3zM2lqYOtVF164+1q51cNtt1RQVVdCz
p06Hi8gvo9PjktTateuAYXgPC26v10fbtu1P6HgbN9rIyPBSWuqgceMQ06YFuOGGmroaV0SSnJq2
JLXWrduQlpaOYXiBSGAPGHDHcd+MVlsL06a5uPbaFEpLHfTuXU1RUaUCW0TqlJq2JL3c3Dz69u3H
6tXFtG3b/rgDe/NmGxkZBmvWOGncOMQzzwS46SaFtYjUPYW2CJHGfbxhHQrBjBkunnzSg99v46ab
qhk/PkjjxuEoTSkiyU6hLXICvvzSxtChBsXFTk45JURBQYBbblG7FpHo0jVtkeMQCsELL7i45poU
ioudXH99NatWVSqwRcQUatoix2jLlki7fucdJw0ahJk0yU/v3jXYbFZPJiLJQk1b5ChCIZg1y8XV
V6fwzjtOrrsu8rnrPn0U2CJiLjVtkSPYujXSrouKnJx8cphnnvHTt6/CWkSsodAW+QnhMMye7WLM
GA8HD9ro3r2GvLwATZvqznARsY5CW+S/bNtmY9gwg5UrnZx0UpiCAj/9+qldi4j1FNoi/xEOw9y5
Lh591MOBAzauvbaGSZMCNGumdi0isUGhLQJ8+62N4cMN/vUvJ/Xrh8nP9zNggNq1iMQWhbYktXAY
5s93kpNjsH+/jauvrmHy5ACnn652LSKxR6EtSeu772xkZhq89ZaTlJQwEycGuPPOarVrEYlZCm1J
OuEwLFjgJDvbYN8+G5061ZCfH+CMM9SuRSS2WfJwlUAgQLdu3Vi4cKEVy0sS27HDxsCBBoMGeamu
hgkTAixY4Fdgi0hcsKRpP/fcc5x88slWLC1JKhyGV15xMnq0QXm5jauuirTr5s0V1iISP0wP7U2b
NrFx40a6dOli9tKSpMrKbIwc6eG111z4fGHGjQvw299WY9dDfEUkzpj+x9b48ePJysoye1lJUosX
O+nc2cdrr7lo166GFSsquOceBbaIxCdTm/aiRYu47LLLOOOMM47p9xs29OF0Oup8jtTU+nV+TPkx
K/e5rAwGDYK//x28XpgyBR580IndXs+ymaJFr2dzaJ/NoX0+MlNDe+XKlWzdupWVK1fy3Xff4Xa7
adq0KR06dPjJ3y8vr6zzGVJT61NWdqDOjyuHs3KfX33VyahRHnbtstOmTQ0FBQHOOSfM7t2WjBNV
ej2bQ/tsDu1zxJHeuJga2vl/oHwtAAARtUlEQVT5+Yf+d2FhIaeddtrPBrbI8dqzB0aPNnjlFReG
EebxxwPce281jro/WSMiYgl9TlsSwuuvO3noIQ9lZXauuKKWwkI/552nO8NFJLFYFtqDBw+2amlJ
IOXlkJ1t8PLLLjyeMGPGBLj/frVrEUlMatoSt5YudZCZabBzp51WrWopKAhw/vkhq8cSEYkahbbE
nb17ISfH4G9/c+F2h8nJCfLAA1U49WoWkQSnP+Ykrixb5mD4cIPvvrNz6aW1FBYGuPBCtWsRSQ4K
bYkL+/fDI48YzJvnwuUKk50d5MEH1a5FJLnojzyJecuXR9r19u12Lr44cu26RQu1axFJPgptiVkH
DsCYMR5mz3bjdIYZOTLIkCFVuFxWTyYiYg2FtsSklSsdDBtmsG2bnRYtIu364ovVrkUkuSm0JaYc
PAiPPebhxRfdOBxhMjODDBtWhdtt9WQiItZTaEvMKCpyMHSowdatdi66KHJn+CWXqF2LiHxPoS2W
O3gQxo71MGtWpF0PGxZk+PAqPB6rJxMRiS0KbbHUu+86yMgw+PprOxdcEGnXl12mdi0i8lMU2mKJ
igp46ikPzz/vxm4Pk5ERZMSIKgzD6slERGKXQltMV1ISaddffWXn17+O3Bl+xRVq1yIiR2O3egA5
stLSNUydmk9p6RqrR/nFKivhkUc83HKLl6+/tjFoUBX/+lelAltE5BipacewrKxM5s6dTSDgxzC8
pKWlk5ubZ/VYJ2TNGjsZGV42b7Zz7rkhCgr8XHmlwlpE5Hioaceo0tLVhwIbIBDwM2/enLhr3H5/
5HPXN93k48svbdx/fxXLl1cosEVEToBCO0aVlBQfCuzv+f2VrF5dbNFEx+/99+107erj2WfdnHVW
mMWL/TzxRBCv1+rJRETik0I7RrVr1wHDODzdvF4fbdu2t2iiYxcIQFYW3HCDj40bHdx7bxUrVlTQ
rl2t1aOJiMQ1hXaMat26DWlp6YeC2+v1MWDAHbRu3cbiyY5s3To73bv7GD8ezjgjzKJFlfzpT0F8
PqsnExGJf7oRLYbl5ubRt28/Vq8upm3b9jEd2MEg5OW5KSx0U1trY9AgGDGigpQUqycTEUkcCu0Y
17p1m5gOa4D16+1kZBh88omDM88MkZ/vp3dvH2VlVk8mIpJYdHpcTlhVFeTmurnuOh+ffOLg7rur
WLmygo4dde1aRCQa1LTlhHz4oZ3Bgw0+/tjB6aeHmDzZz9VXK6xFRKJJTVuOS3U1PP20m549fXz8
sYM776zi7bcrFNgiIiZQ05Zj9tFHkWvXH37ooFmzSLu+5hqFtYiIWdS05aiqq2HSJDc9evj48EMH
aWlVrFpVocAWETGZmrYc0SefRNr1+vUOmjaNtOuuXRXWIiJWUNOWn1RTA1OmuOne3cf69Q769atm
1aoKBbaIiIXUtOVHPvss0q7XrXNw6qkh8vL89OihsBYRsZqathxSWwtTp7ro1s3HunUO+vaNtGsF
tohIbFDTFgA2brQxeLCX9993kJoaYuLEAL161Vg9loiI/ICadpKrrYVnn3Vx7bUpvP++gz59qikq
qlBgi4jEIDXtJLZ5c6Rdv/eeg8aNQzz7bIAbb1RYi4jEKtNDe8KECbz//vvU1NRw33330aNHD7NH
SHqhEMyY4eLJJz34/TZuuaWaceOCNG4ctno0ERE5AlNDu6SkhC+++IL58+dTXl5O7969Fdom27zZ
xtChBiUlTho1ClFYGODmm+u2XZeWrmHDhrW0bNkq5r+hTEQknpga2ldeeSWXXHIJACeddBJ+v5/a
2locDoeZYySlUAhmzXIxdqyHykobN9xQzYQJQVJT67ZdZ2VlMnfubAIBP4bhJS0tndzcvDpdQ0Qk
WZl6I5rD4cDn8wGwYMECOnfurMA2wZYtNm67zcvo0QYeD0yf7mfmzECdB3Zp6epDgQ0QCPiZN28O
paVr6nQdEZFkZcmNaMuWLWPBggXMnDnziL/XsKEPp7PuQz01tX6dHzMWhUIwfTo89BBUVMAtt8C0
aTaaNvVGZb0NG9YdCuzv+f2VbNiwll69ukZlTUme17PVtM/m0D4fmemhXVRUxLRp05gxYwb16x/5
P055eWWdr5+aWp+ysgN1ftxY8/XXNoYNMygqctKgQZhnnw1w22012GxQVhadNVu2bIVheA8Lbq/X
R8uWrZJiz62QLK9nq2mfzaF9jjjSGxdTT48fOHCACRMmMH36dBo0aGDm0kkjHIYXX3Rx9dUpFBU5
6dGjhlWrKujbNxLY0dS6dRvS0tIxjEiT93p9DBhwh25GExGpI6Y27ddff53y8nKGDh166Gfjx4+n
WbNmZo6RsL75JtKu337byUknhSks9HP77dEP6x/Kzc2jb99+untcRCQKbOFwOGY/nBuN0ySJePol
HIa5c1088oiHgwdtdO1aw6RJAX71K+v+0ybiPsci7bM5tM/m0D5HHOn0uJ6IFue2b7cxfLjB8uVO
6tcPk5/vZ8AAc9u1iIiYQ6Edp8JhmD/fSU6Owf79Nrp0qWHy5ACnnRazJ05EROQXUmjHoe++s5GZ
afDWW07q1QuTlxcgPb1a7VpEJMEptONIOAx//7uThx822LfPRqdONeTnBzjjDLVrEZFkoNCOEzt2
2HjoIQ9vvOHC5wszYUKAgQPVrkVEkolCO8aFw7BwoZPsbIPychsdO0auXTdvrnYtIpJsFNoxbOdO
GyNHenj99Ui7HjcuwG9/W43d1EfiiIhIrFBox6jFi52MGuVhzx477dtHrl2ffbbatYhIMlNox5hd
u2yMGuXh1VddeL1hnnwywD33qF2LiIhCO6a8+mqkXe/aZadNmxoKCgKcc47atYiIRCi0Y8Du3TZG
j/awaJELwwjzxBMB/vCHavRV4yIi8kMKbYu99pqThx6KtOvWrWspKPBz3nlq1yIi8mNJFdqlpWti
5tun9uyB7GyDhQtdeDxhxowJcP/9atciIvLzkia0s7IymTt3NoGAH8PwkpaWTm5uniWzvPGGgxEj
DHbutNOqVS0FBQHOPz9kySwiIhI/kuKe5NLS1YcCGyAQ8DNv3hxKS9eYOsfevTBokMFdd/nYu9dG
Tk6Qf/yjUoEtIiLHJClCu6Sk+FBgf8/vr2T16mLTZnjrLQedO6fw97+7uOyyWpYtqyQjowpn0pzr
EBGRXyopQrtduw4Yhvewn3m9Ptq2bR/1tfftgyFDDO64w8fu3Tays4O8/nolF16odi0iIscnKUK7
des2pKWlHwpur9fHgAF3RP1mtOXLI+163jwXl1xSy1tvVTJ0qNq1iIicmKSJj9zcPPr27WfK3eP7
98OYMR7mzHHjdIYZNSpIRkYVLlfUlhQRkSSQNKENkcbdq1dXysoORG2NlSsdDBtmsG2bnRYtIneG
X3yxToWLiMgvl1ShHU0HD8Jjj3l48cVIux4xIsjQoVW43VZPJiIiiUKhXQdWrYq0661b7Vx0US2F
hQEuuUTtWkRE6pZC+xc4eBDGjvUwa5YbhyPM8OFBhg9XuxYRkehQaJ+gd95xMGSIwddf27nggki7
vuwytWsREYkehfZxqqiAJ5/0MGOGG7s9zJAhQUaMqMLjsXoyERFJdArt41BS4iAjw+Crr+z8+teR
dt2qldq1iIiYQ6F9DCorYdw4D3/+swubDR58MMjIkVUYhtWTiYhIMlFoH8WaNXYyMrxs3mzn3HND
FBT4ufJKtWsRETGfQvtn+P2Qm+th2rTIY8z++McqsrKCeL1H+QdFRESiRKH9E0pL7WRkGGzc6ODs
s0MUFARo27bW6rFERCTJKbR/IBCACRPcPPusm3AY7ruvitGjg/h8Vk8mIiKi0D5k3To7gwcbfP65
g+bNI+26fXu1axERiR2mh/ZTTz3F+vXrsdlsZGdnc8kll5g9wmGCQZg40U1hoZtQyMbvf1/Fww8H
SUmxdCwREZEfMTW016xZw5YtW5g/fz6bNm0iOzub+fPnmznCYdavj7TrTz91cOaZIaZM8XPVVWrX
IiISm+xmLlZcXEy3bt0AOPfcc9m3bx8HDx40cwQAqqogN9fNddf5+PRTB3ffXcXKlRUKbBERiWmm
Nu1du3bRokWLQ399yimnUFZWRr169UybYd06SE/38fHHDk4/PUR+vp/OnRXWIiIS+yy9ES0cDh/x
7zds6MPpdNTZetOmweDBUFPj4N574emn7Zx0km4Nj5bU1PpWj5AUtM/m0D6bQ/t8ZKaGdpMmTdi1
a9ehv965cyepqak/+/vl5ZV1uv7ixQbNmrl4+ulKrrmmlmAQysrqdAn5j9TU+pSVHbB6jISnfTaH
9tkc2ueII71xMfWa9lVXXcXSpUsB+Oijj2jSpImpp8Znzgzw1VdwzTU6HS4iIvHH1KbdqlUrWrRo
Qf/+/bHZbIwZM8bM5XE6wWYzdUkREZE6Y/o17REjRpi9pIiISEIw9fS4iIiInDiFtoiISJxQaIuI
iMQJhbaIiEicUGiLiIjECYW2iIhInFBoi4iIxAmFtoiISJxQaIuIiMQJhbaIiEicsIWP9v2YIiIi
EhPUtEVEROKEQltERCROKLRFRETihEJbREQkTii0RURE4oRCW0REJE4kVWg/9dRT9OvXj/79+/PB
Bx9YPU7CmjBhAv369eO2227jzTfftHqchBYIBOjWrRsLFy60epSEtWTJEm6++Wb69OnDypUrrR4n
IVVUVPDggw9y55130r9/f4qKiqweKWY5rR7ALGvWrGHLli3Mnz+fTZs2kZ2dzfz5860eK+GUlJTw
xRdfMH/+fMrLy+nduzc9evSweqyE9dxzz3HyySdbPUbCKi8v55lnnuHll1+msrKSwsJCunTpYvVY
CeeVV17h7LPPJjMzkx07djBw4EDeeOMNq8eKSUkT2sXFxXTr1g2Ac889l3379nHw4EHq1atn8WSJ
5corr+SSSy4B4KSTTsLv91NbW4vD4bB4ssSzadMmNm7cqBCJouLiYtq3b0+9evWoV68eY8eOtXqk
hNSwYUM+++wzAPbv30/Dhg0tnih2Jc3p8V27dh32QjjllFMoKyuzcKLE5HA48Pl8ACxYsIDOnTsr
sKNk/PjxZGVlWT1GQvvmm28IBALcf//9pKWlUVxcbPVICemGG25g+/btdO/enfT0dEaNGmX1SDEr
aZr2f9PTW6Nr2bJlLFiwgJkzZ1o9SkJatGgRl112GWeccYbVoyS8vXv3MnXqVLZv385dd93FihUr
sNlsVo+VUBYvXkyzZs144YUX+PTTT8nOztZ9Gj8jaUK7SZMm7Nq169Bf79y5k9TUVAsnSlxFRUVM
mzaNGTNmUL9+favHSUgrV65k69atrFy5ku+++w63203Tpk3p0KGD1aMllEaNGnH55ZfjdDo588wz
SUlJYc+ePTRq1Mjq0RLK2rVr6dixIwAXXnghO3fu1GW1n5E0p8evuuoqli5dCsBHH31EkyZNdD07
Cg4cOMCECROYPn06DRo0sHqchJWfn8/LL7/M3/72N37zm9/wwAMPKLCjoGPHjpSUlBAKhSgvL6ey
slLXW6OgefPmrF+/HoBt27aRkpKiwP4ZSdO0W7VqRYsWLejfvz82m40xY8ZYPVJCev311ykvL2fo
0KGHfjZ+/HiaNWtm4VQiJ+bUU0+lZ8+e3H777QDk5ORgtydN1zFNv379yM7OJj09nZqaGh577DGr
R4pZ+mpOERGROKG3jCIiInFCoS0iIhInFNoiIiJxQqEtIiISJxTaIiIicUKhLSIiEicU2iIiInFC
oS0ih8yaNYucnBwANm/ezHXXXcfBgwctnkpEvqfQFpFDBg4cyJdffsn777/P448/zhNPPKHH/YrE
ED0RTUQOs2XLFtLT07nuuut4+OGHrR5HRH5ATVtEDrNv3z58Ph/ffvut1aOIyH9RaIvIIcFgkDFj
xjBt2jRcLheLFi2yeiQR+QGdHheRQyZMmEBKSgqDBg1i165d9OvXjzlz5tC0aVOrRxMRFNoiIiJx
Q6fHRURE4oRCW0REJE4otEVEROKEQltERCROKLRFRETihEJbREQkTii0RURE4oRCW0REJE78fwBv
rSeAgAtyAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>So now we can make predictions with new points based off our data</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#here we can try out any data point</span>
<span class="nb">print</span><span class="p">(</span><span class="n">regress</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mi">6</span><span class="p">]]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[[8.25454545]]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Applied-Linear-Regression">Applied Linear Regression<a class="anchor-link" href="#Applied-Linear-Regression">&#182;</a></h2><hr>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="The-Ames-Housing-Dataset">The Ames Housing Dataset<a class="anchor-link" href="#The-Ames-Housing-Dataset">&#182;</a></h3><blockquote><p>Ames is a city located in Iowa.</p>
<ul>
<li>This data set consists of all property sales
collected by the Ames City Assessor’s Office between the years
of 2006 and 2010.</li>
<li>Originally contained 113 variables and 3970 property sales
pertaining to the sale of stand-alone garages, condos, storage
areas, and of course residential property.</li>
<li>Distributed to the public as a means to replace the old Boston
Housing 1970’s data set.  </li>
<li><a href="http://lib.stat.cmu.edu/datasets/boston">Link to Original</a> </li>
<li>The "cleaned" version of this dataset contains 2930 observations along with 80
predictor variables and two identification variables.</li>
</ul>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="What-was-the-original-purpose-of-this-data-set?">What was the original purpose of this data set?<a class="anchor-link" href="#What-was-the-original-purpose-of-this-data-set?">&#182;</a></h3><p>That is, why did the Amess City Assessor's Office decide to collect this data?</p>
<blockquote><ul>
<li><strong>Answer</strong>: To update the assessment model used by the Ames
City Assessor’s Office.</li>
</ul>
</blockquote>
<p>Now you may ask, what is an assessment model?</p>
<blockquote><ul>
<li><strong>Answer</strong>: In short, an assessment model is used to assign dollar value to a property that reflects the true market value of that property.</li>
</ul>
</blockquote>
<p>Now according to the Iowa Department of Revenue’s website, primary beneficiaries of the revenue generated by
property taxes include but are not limited to:</p>
<blockquote><ul>
<li>K-12 Schools, Hospitals, Assessors, Townships, and Agricultural
Extension Districts.</li>
</ul>
</blockquote>
<p><strong>Moral of this story</strong>: We will be using the modified Ames housing dataset to predict housing price.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="What's-inside?">What's inside?<a class="anchor-link" href="#What's-inside?">&#182;</a></h3><p>This ”new” data set contains 2930 (n=2930) observations along with 80
predictor variables and two identification variables.</p>
<p><a href="http://jse.amstat.org/v19n3/decock.pdf">Paper linked to dataset</a></p>
<p>An exhaustive variable breakdown can be found
<a href="http://jse.amstat.org/v19n3/decock/DataDocumentation.txt">here</a></p>
<h3 id="Quick-Summary"><strong>Quick Summary</strong><a class="anchor-link" href="#Quick-Summary">&#182;</a></h3><hr>
<p>Of the 80 predictor variables we have:</p>
<blockquote><ul>
<li>20 continuous variables (area dimension)<ul>
<li>Garage Area, Wood Deck Area, Pool Area</li>
</ul>
</li>
<li>14 discrete variables (items occurring)<ul>
<li>Remodeling Dates, Month and Year Sold</li>
</ul>
</li>
<li>23 nominal and 23 ordinal <ul>
<li>Nominal: Condition of the Sale, Type of Heating and
Foundation</li>
<li>Ordinal: Fireplace and Kitchen Quality, Overall
Condition of the House</li>
</ul>
</li>
</ul>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong><em>Question to Answer</em></strong>: What is the linear relationship between sale price on above ground
living room area?</p>
<p>But first lets visually investigate what we are trying to predict.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We shall start our analysis with summary statistics.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">housing_data</span> <span class="o">=</span>  <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;AmesHousing.txt&quot;</span><span class="p">,</span> <span class="n">delimiter</span><span class="o">=</span><span class="s2">&quot;</span><span class="se">\t</span><span class="s2">&quot;</span><span class="p">)</span> 

<span class="c1">#Mean Sales price </span>
<span class="n">mean_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Mean Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">mean_price</span><span class="p">))</span>

<span class="c1">#Variance of the Sales Price </span>
<span class="n">var_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">var</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">],</span> <span class="n">ddof</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Variance of Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">var_price</span><span class="p">))</span>

<span class="c1">#Median of Sales Price </span>
<span class="n">median_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Median Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">median_price</span><span class="p">))</span>

<span class="c1">#Skew of Sales Price </span>
<span class="n">skew_price</span> <span class="o">=</span> <span class="n">st</span><span class="o">.</span><span class="n">skew</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Skew of Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">skew_price</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Mean Price : 180796.0600682594
Variance of Sales Price : 6381883615.6884365
Median Sales Price : 160000.0
Skew of Sales Price : 1.74260737195
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;Price&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Frequency&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Histogram of Housing Prices&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfgAAAFnCAYAAABKGFvpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4zLCBo
dHRwOi8vbWF0cGxvdGxpYi5vcmcvIxREBQAAIABJREFUeJzt3XlY1XXe//HXgQMRigUEpraOlTaF
W6a5kImKS3XnnjLg2NjiiEsTpmg62mi5ZWOppWmmt+ZyS6ZYCo6l5iRyW8wPl8nUpmZc4SAgyiKI
398fXp5bculAHuB8fD6uy+vifM93eb/POV6v8/18l2OzLMsSAAAwildVFwAAAK4/Ah4AAAMR8AAA
GIiABwDAQAQ8AAAGIuABADAQAQ9cRYMGDXTixIky09asWaOBAwdKkpYtW6ZZs2Zdcx3p6enav3+/
u0p0q9LSUg0YMEARERH6/vvvyzx36etwqZiYGK1bt+661zJz5kytWLHiuq2vQYMG6tSpk7p06aLO
nTurV69eSklJueK8GRkZeuqpp67btoHKYq/qAgBPFR0d/YvzfPLJJ3rkkUfUsGHDSqjo+srMzNSu
Xbu0e/du+fj4VGktcXFx132dS5cu1e233y5J+vbbb/XHP/5RSUlJCgoKKjNf7dq19dlnn1337QPu
xh48UEGzZ8/Wa6+9JknauHGjnnrqKXXt2lVPP/20UlNTtWLFCq1bt04zZszQRx99pPPnz+uvf/2r
unTpoi5duig+Pl4FBQWSpH379ikyMlKRkZGaM2eOcx1HjhxR27Zt9eabbzq/UHzxxRd6+umn1blz
Z/Xs2VPfffedJCk1NVXPPvus3njjDXXo0EE9e/ZUenq6YmJi1KZNG7377rtX7GP//v3q16+funTp
omeeeUbbt29XaWmpYmJidP78eT399NMVGoU4duyYBg0apM6dO+upp57S2rVrnXV26tTJOd+ljw8c
OKBnn31WTz75pCIjI7Vs2TJJUnx8vN577z1JUkREhFauXKnevXurbdu2mjp1qnNd8+bNU6tWrdSr
Vy99/PHHioiIcKnWRx55RHfddZf+8Y9/XPaaHzlyRL/97W8lSZZlacqUKYqIiFDnzp21cOFC5/Q5
c+aoc+fOat++vSZPnqzS0lJJV/5sAJWBgAeug9dff13z58/Xxo0bNWHCBH355Zfq37+/GjVqpFdf
fVXPPfecNm7cqK+++kpr1qzR559/rry8PC1evFiSNH78eA0cOFCbNm1SzZo19dNPPznXnZubqwcf
fFDLli3TuXPnFB8fr0mTJik5OVkRERGaNm2ac959+/apY8eO2rx5s7y8vPSXv/xFH3zwgT766CPN
nz9fZ8+eLVP3+fPn9corryg6OlpJSUmaPHmy4uLiVFhYqMWLF8vb21tJSUkVGoEYP368WrRooeTk
ZM2fP1+TJ0/WkSNHrrnMnDlz1K9fP33++edauXKlduzYoeLi4svm27Vrl1atWqVPPvlEy5Yt04kT
J3Tw4EEtXLhQ69at0/Lly5WUlFSues+dOydfX19JZV/zSyUmJmr37t1KTk52bnv37t1at26dkpKS
lJCQoL/97W86fPiw85DClT4bQGUg4IFriImJce5xd+nSRW+//fYV5wsODtbKlSt19OhRNW/eXGPG
jLlsnq1bt6p79+7y9/eXt7e3evbsqa+//lpFRUXat2+f8zjv7373O116B+mSkhLnHq7dbteOHTvU
pEkTSVLz5s11+PBh57y1atVSy5YtZbPZdP/996tFixa6+eabdf/996u0tFTZ2dllajpy5IiysrL0
5JNPSpLCwsJUt25d7dmz5xdfm//3//5fmdemS5cu2r17t7PmHTt2KCoqSpJUr149tWzZUjt37rzm
OoODg5WcnKx9+/YpMDBQ7733njN0L/X000/L29tbtWvXVnBwsI4fP65du3apRYsWCg0N1U033aRe
vXr9Yg8Xbdu2TVlZWWrWrJmz/ktHGS766quv1LlzZ/n4+KhmzZrasGGDwsLCtGXLFvXq1UsBAQGy
2+3q06ePNm3a5Ozplz4bgDtwDB64hkuP00oXTi5LTEy8bL73339f77//vnr27Kk6depo7NixatGi
RZl5srOzdcsttzgf33LLLTp58qROnTolm82mWrVqSZJ8fHwUHBzsnM/b21s1a9YsU9Onn36q4uJi
FRcXy2azOZ+rUaOG828vLy/5+/tLkmw2m7y8vJzDxpfWFBAQUGYdtWrVUnZ2tu68885rvjZNmjRx
jkBcFBMTI+nCHrBlWQoICCjXekeOHKn58+fr5Zdf1tmzZ/XSSy/pd7/73WXzXfp6eHt7q7S0VHl5
eWVe39q1a1+z/piYGHl7e8uyLNWrV08LFixQjRo1lJOTc9lrflFOTo7zfZLkfH1Pnz6tDz/8UKtW
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
AMBABDwAAAaq1pfJvfnmm0pPT5fNZtPYsWPVqFGjqi4JAACPUG0D/n//93/173//W6tWrdIPP/yg
sWPHatWqVVVdFgAPVN2v05e4Vh/XX7UN+JSUFHXs2FGSVL9+fZ06dUpnzpxRzZo1q7gyALj++BKC
663aBnxWVpYeeugh5+OgoCA5HA4CHgCqiCd8CanuKvNLUrUN+J+zLOsX5wkJCajQuq+23PqZz1Ro
fQAAVLVqexZ9aGiosrKynI8zMzMVEhJShRUBAOA5qm3At2nTRsnJyZKkffv2KTQ0lOF5AABcVG2H
6Js1a6aHHnpI/fr1k81m04QJE6q6JAAAPIbNcuXgNgAA8CjVdogeAABUHAEPAICBqu0x+MrgCbfC
PXDggIYMGaKBAwcqOjpax48f16hRo1RaWqqQkBDNmDFDvr6+SkxM1JIlS+Tl5aW+ffuqT58+Kikp
UXx8vI4dOyZvb29NmTJFd955p/bv36+JEydKkho0aKDXX39dkrRw4UIlJSXJZrNp6NChateundv6
mj59ur799ludO3dOL730ksLCwozoq7CwUPHx8Tp58qTOnj2rIUOGqGHDhkb0JklFRUV66qmnNGTI
ELVq1cqIvlJTUzVixAjdf//9kqQHHnhAzz//vBG9SVJiYqIWLlwou92u4cOHq0GDBh7f2+rVq5WY
mOh8vHfvXq1YscLlmk6fPq24uDidPn1a/v7+mjlzpm699Vbt2LFDb7/9try9vfX4448rNjZWkmdk
xRVZN6jU1FTrxRdftCzLsg4dOmT17du3iiu6XH5+vhUdHW2NGzfOWrp0qWVZlhUfH29t2LDBsizL
mjlzpvXxxx9b+fn5VmRkpJWXl2cVFhZaTz75pJWTk2OtWbPGmjhxomVZlrV9+3ZrxIgRlmVZVnR0
tJWenm5ZlmW98sor1tatW63//Oc/Vo8ePayzZ89aJ0+etDp37mydO3fOLX2lpKRYzz//vGVZlpWd
nW21a9fOiL4sy7I+//xz64MPPrAsy7KOHDliRUZGGtObZVnW22+/bfXs2dP65JNPjOlr586d1rBh
w8pMM6W37OxsKzIy0jp9+rSVkZFhjRs3zpjeLkpNTbUmTpxYrppmz55tLViwwLIsy1q5cqU1ffp0
y7Isq2vXrtaxY8es0tJSq3///tbBgwc9Iiuu5oYdor/arXCrE19fXy1YsEChoaHOaampqerQoYMk
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Another way we can view our data is with a box and whisker plot.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Sales Price&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
TkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now we shall look at sales price on above ground living room area.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;Gr Liv Area&quot;</span><span class="p">],</span> <span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Sales Price&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Finally, lets generate our model and see how it predicts Sales Price!!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#we need to reshape the array to make the sklearn gods happy</span>
<span class="n">area_reshape</span> <span class="o">=</span> <span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;Gr Liv Area&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="n">price_reshape</span> <span class="o">=</span> <span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>

<span class="c1">#Generate the Model</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">linear_model</span><span class="o">.</span><span class="n">LinearRegression</span><span class="p">(</span><span class="n">fit_intercept</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">area_reshape</span><span class="p">,</span> <span class="n">price_reshape</span><span class="p">)</span>
<span class="n">price_prediction</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">area_reshape</span><span class="p">)</span>

<span class="c1"># plotting the actual points as scatter plot </span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">area_reshape</span><span class="p">,</span> <span class="n">price_reshape</span><span class="p">)</span> 

<span class="c1"># plotting the regression line </span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">area_reshape</span><span class="p">,</span> <span class="n">price_prediction</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s2">&quot;red&quot;</span><span class="p">)</span> 

<span class="c1"># putting labels </span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Above Ground Living Area&#39;</span><span class="p">)</span> 
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Sales Price&#39;</span><span class="p">)</span> 

<span class="c1"># function to show plot </span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
  &#34;&#34;&#34;Entry point for launching an IPython kernel.
/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
  
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
kxwF3sSN2QAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Applied-Logistic-Regression"><strong>Applied Logistic Regression</strong><a class="anchor-link" href="#Applied-Logistic-Regression">&#182;</a></h2><hr>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#we&#39;re going to need a different model, so let&#39;s import it</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegression</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This data set was provided by UCI's Machine Learning Repository:</p>
<ul>
<li><a href="https://archive.ics.uci.edu/ml/datasets/Adult">Adult Data Set (Also know as Census Income)</a></li>
</ul>
<p>We already downloaded the dataset at the begining of the notebook, so now let's mess around with it.</p>
<p>But before that, we need to read in the data, and pandas has the functions we need to do this</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#read_csv allow us to easily import a whole dataset</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;adult.data&quot;</span><span class="p">,</span> <span class="n">names</span> <span class="o">=</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">,</span><span class="s2">&quot;workclass&quot;</span><span class="p">,</span><span class="s2">&quot;fnlwgt&quot;</span><span class="p">,</span><span class="s2">&quot;education&quot;</span><span class="p">,</span><span class="s2">&quot;education-num&quot;</span><span class="p">,</span><span class="s2">&quot;marital-status&quot;</span><span class="p">,</span><span class="s2">&quot;occupation&quot;</span><span class="p">,</span><span class="s2">&quot;relationship&quot;</span><span class="p">,</span><span class="s2">&quot;race&quot;</span><span class="p">,</span><span class="s2">&quot;sex&quot;</span><span class="p">,</span><span class="s2">&quot;capital-gain&quot;</span><span class="p">,</span><span class="s2">&quot;capital-loss&quot;</span><span class="p">,</span><span class="s2">&quot;hours-per-week&quot;</span><span class="p">,</span><span class="s2">&quot;native-country&quot;</span><span class="p">,</span><span class="s2">&quot;income&quot;</span><span class="p">])</span>

<span class="c1">#this tells us whats in it </span>
<span class="nb">print</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">info</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 15 columns):
age               32561 non-null int64
workclass         32561 non-null object
fnlwgt            32561 non-null int64
education         32561 non-null object
education-num     32561 non-null int64
marital-status    32561 non-null object
occupation        32561 non-null object
relationship      32561 non-null object
race              32561 non-null object
sex               32561 non-null object
capital-gain      32561 non-null int64
capital-loss      32561 non-null int64
hours-per-week    32561 non-null int64
native-country    32561 non-null object
income            32561 non-null object
dtypes: int64(6), object(9)
memory usage: 3.7+ MB
None
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># data.head() gives us some the the first 5 sets of the data</span>
<span class="nb">print</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>

<span class="c1">#this is the function that give us some quick info about continous data in the dataset</span>
<span class="nb">print</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">describe</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>   age          workclass  fnlwgt   education  education-num  \
0   39          State-gov   77516   Bachelors             13   
1   50   Self-emp-not-inc   83311   Bachelors             13   
2   38            Private  215646     HS-grad              9   
3   53            Private  234721        11th              7   
4   28            Private  338409   Bachelors             13   

        marital-status          occupation    relationship    race      sex  \
0        Never-married        Adm-clerical   Not-in-family   White     Male   
1   Married-civ-spouse     Exec-managerial         Husband   White     Male   
2             Divorced   Handlers-cleaners   Not-in-family   White     Male   
3   Married-civ-spouse   Handlers-cleaners         Husband   Black     Male   
4   Married-civ-spouse      Prof-specialty            Wife   Black   Female   

   capital-gain  capital-loss  hours-per-week  native-country  income  
0          2174             0              40   United-States   &lt;=50K  
1             0             0              13   United-States   &lt;=50K  
2             0             0              40   United-States   &lt;=50K  
3             0             0              40   United-States   &lt;=50K  
4             0             0              40            Cuba   &lt;=50K  
                age        fnlwgt  education-num  capital-gain  capital-loss  \
count  32561.000000  3.256100e+04   32561.000000  32561.000000  32561.000000   
mean      38.581647  1.897784e+05      10.080679   1077.648844     87.303830   
std       13.640433  1.055500e+05       2.572720   7385.292085    402.960219   
min       17.000000  1.228500e+04       1.000000      0.000000      0.000000   
25%       28.000000  1.178270e+05       9.000000      0.000000      0.000000   
50%       37.000000  1.783560e+05      10.000000      0.000000      0.000000   
75%       48.000000  2.370510e+05      12.000000      0.000000      0.000000   
max       90.000000  1.484705e+06      16.000000  99999.000000   4356.000000   

       hours-per-week  
count    32561.000000  
mean        40.437456  
std         12.347429  
min          1.000000  
25%         40.000000  
50%         40.000000  
75%         45.000000  
max         99.000000  
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Now here is the Qustion:</strong></p>
<blockquote><p><em>Which one of these parameters are best in figuring out if someone is going to be making more then 50k a year?</em></p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#put the name of the parameter you want to test</span>
<span class="n">test</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Ploting will help with visualising the data</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#little baby helper function</span>
<span class="k">def</span> <span class="nf">incomeFixer</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
  <span class="k">if</span> <span class="n">x</span> <span class="o">==</span> <span class="s2">&quot; &lt;=50K&quot;</span><span class="p">:</span>
    <span class="k">return</span> <span class="mi">0</span>
  <span class="k">else</span><span class="p">:</span>
    <span class="k">return</span> <span class="mi">1</span>

<span class="c1">#change the income data into 0&#39;s and 1&#39;s</span>
<span class="n">data</span><span class="p">[</span><span class="s2">&quot;income&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">row</span><span class="p">:</span> <span class="n">incomeFixer</span><span class="p">(</span><span class="n">row</span><span class="p">[</span><span class="s1">&#39;income&#39;</span><span class="p">]),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="c1">#ploting </span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">test</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;income&#39;</span><span class="p">],</span> <span class="n">color</span><span class="o">=</span> <span class="s2">&quot;black&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
vl0zZmQEv0Ec7lvQnflb0rGxsZox45aQ2xq/xs5szJgUxcSEvusSE+OMyrek+/cfoJEjR4XcNnLk
KL4lDaCJsFfCqampGj16tDIzM+VwOJSTk6PCwkK53W5NnTpVubm5mjdvnqSvvimdmHhx3mpLSEjQ
rbd+9e3oxnbv3t/st6M7u/OvZePGDSotPaqEhKGaMSPjgtfYmZWUHGj229HRsmHDu81+OxoAGgv7
R5QiLZKX+Pw54chp6fYLf064db4+Q/6ccNtxK9U+ZmhfR96O7tIRlnjDRQpztI8Z2scM7WOG9nWq
z4QBAEB0EGEAAAwhwgAAGEKEAQAwhAgDAGAIEQYAwBAiDACAIUQYAABDiDAAAIYQYQAADCHCAAAY
0uH/7WgAAPAVroQBADCECAMAYAgRBgDAECIMAIAhRBgAAEOIMAAAhrhML8CO/Px8FRcXy+FwKDs7
WykpKaaX1Ck8++yz+uijj1RfX6/7779fY8aM0eOPP66GhgZ5vV4tWbJEPXv21Pr16/WHP/xBMTEx
uuOOO3T77bfr7Nmzmj9/vkpLS+V0OrVo0SINGzZMn3zyiXJzcyVJw4cP1y9+8QuzL7IDnD59Wrfe
eqsefPBBjR8/nhm20fr16/XSSy/J5XLpZz/7mYYPH84M26i2tlZPPPGEqqqqdPbsWc2dO1derzfk
DF566SVt2rRJDodDP/nJTzRp0iRVV1dr3rx5qq6uVmxsrJYtW6Z+/fpp+/btWr58uZxOpyZOnKi5
c+cafJXRsX//fj344IOaM2eOsrKy9Pnnn0ft/Rdq9q1mdVEffPCBdd9991mWZVkHDx607rjjDsMr
6hyKioqse++917Isyzp+/Lg1adIka/78+daGDRssy7KsZcuWWWvWrLFqa2utadOmWSdPnrROnTpl
3XLLLVZlZaVVWFho5ebmWpZlWdu2bbMeeughy7IsKysryyouLrYsy7IeeeQRa+vWrQZeXcdavny5
9b3vfc968803mWEbHT9+3Jo2bZpVXV1tlZWVWQsXLmSG7fDqq69aS5cutSzLsr744gtr+vTpIWfw
2WefWd/97netM2fOWBUVFdb06dOt+vp6a8WKFdaqVassy7KstWvXWs8++6xlWZY1Y8YMq7S01Gpo
aLDuvPNO68CBA2ZeYJTU1tZaWVlZ1sKFC61XX33Vsiwrau+/5mbfWl32dnRRUZHS09MlSUlJSaqq
qlJNTY3hVZk3btw4Pf/885KkPn366NSpU/rggw80ZcoUSdJNN92koqIiFRcXa8yYMXK73erVq5dS
U1O1a9cuFRUVaerUqZKk66+/Xrt27dKXX36pY8eOBe80nD9Gd3bo0CEdPHhQN954oyQxwzYqKirS
+PHjFRcXJ5/Pp1/+8pfMsB08Ho9OnDghSTp58qT69esXcgYffPCBbrjhBvXs2VPx8fEaMmSIDh48
2GSO5/c9cuSI+vbtq8GDBysmJkaTJk3qdnPs2bOnVq1aJZ/PF3wuWu+/5mbfWl02woFAQB6PJ/g4
Pj5efr/f4Io6B6fTqdjYWEnSunXrNHHiRJ06dUo9e/aUJPXv319+v1+BQEDx8fHBnzs/v8bPx8TE
yOFwKBAIqE+fPsF9zx+jO3vmmWc0f/784GNm2DZHjx7V6dOn9cADD+gHP/iBioqKmGE73HLLLSot
LdXUqVOVlZWlxx9/POQMWjPH/v37q7y8XH6/P+S+3YnL5VKvXr2aPBet919zx2j1Wtv1Cjshi//6
ZhN///vftW7dOv3ud7/TtGnTgs83N6e2PN/dZ/3WW2/p6quv1rBhw0JuZ4atc+LECb3wwgsqLS3V
7Nmzm7xmZtg6b7/9thISErR69Wp98sknmjt3rtxud3A782qfaL7/2jrnLnsl7PP5FAgEgo/Ly8vl
9XoNrqjz2LZtm37zm99o1apVcrvdio2N1enTpyVJZWVl8vl8Ied3/vnzv4s7e/asLMuS1+sN3hJr
fIzuauvWrXr33Xd1xx136I033tCLL77IDNuof//+uuaaa+RyuXTppZeqd+/e6t27NzNso127dmnC
hAmSpBEjRujMmTOqrKwMbm9ujo2fPz/HcPt2d9H637DdeXbZCKelpWnz5s2SpL1798rn8ykuLs7w
qsyrrq7Ws88+q5UrV6pfv36SvvpM4/ys/va3v+mGG27QVVddpT179ujkyZOqra3Vrl27NHbsWKWl
pWnTpk2SpC1btui6665Tjx499M1vflMffvhhk2N0V88995zefPNNvf7667r99tv14IMPMsM2mjBh
gt5//32dO3dOlZWVqqurY4bt8I1vfEPFxcWSpGPHjql3795KSkq6YAbf+ta3tHXrVn355ZcqKytT
eXm5Lr/88iZzPL/v0KFDVVNTo6NHj6q+vl5btmxRWlqasdfYUaL1/mtu9q3Vpf8WpaVLl+rDDz+U
w+FQTk6ORowYYXpJxhUUFGjFihVKTEwMPrd48WItXLhQZ86cUUJCghYtWqQePXpo06ZNWr16tRwO
h7KysnTbbbepoaFBCxcu1KeffqqePXtq8eLFGjx4sA4ePKinnnpK586d01VXXaUnn3zS4KvsOCtW
rNCQIUM0YcIEPfHEE8ywDdauXat169ZJkn784x9rzJgxzLCNamtrlZ2drYqKCtXX1+uhhx6S1+sN
OYNXX31Vf/nLX+RwOPTwww9r/Pjxqq2t1WOPPaYTJ06oT58+WrJkidxut3bu3KmlS5dKkqZNm6Z7
7rnH5MuMuJKSEj3zzDM6duyYXC6XBg4cqKVLl2r+/PlRef+Fmn1rdekIAwDQlXXZ29EAAHR1RBgA
AEOIMAAAhhBhAAAMIcIAABhChAEAMIQIAwBgCBEGAMCQ/w9jVgBBb0p75wAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#but before we make our model, we need to modify our data a bit</span>


<span class="c1">#get the data we are going to make the model with </span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">test</span><span class="p">])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="s2">&quot;income&quot;</span><span class="p">])</span>

<span class="c1">#again, lets make the scikitlearn gods happy</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>

<span class="c1">#Making the test-train split</span>
<span class="n">x_train</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">x</span> <span class="p">,</span><span class="n">y</span> <span class="p">,</span><span class="n">test_size</span><span class="o">=</span><span class="mf">0.25</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#now make data model!</span>
<span class="n">logreg</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">solver</span><span class="o">=</span><span class="s1">&#39;liblinear&#39;</span><span class="p">)</span>
<span class="n">logreg</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[34]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class=&#39;warn&#39;,
          n_jobs=None, penalty=&#39;l2&#39;, random_state=None, solver=&#39;liblinear&#39;,
          tol=0.0001, verbose=0, warm_start=False)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#now need to test the model&#39;s performance</span>
<span class="nb">print</span><span class="p">(</span><span class="n">logreg</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">x_test</span><span class="p">,</span><span class="n">y_test</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>0.802972607787741
</pre>
</div>
</div>

</div>
</div>

</div>
 

