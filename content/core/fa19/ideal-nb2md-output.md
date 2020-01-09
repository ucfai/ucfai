---
title: "Getting Started, Regression"
linktitle: "Getting Started, Regression"
date: "2019-09-18T00:00:00Z"
lastmod: "2019-09-18T00:00:00Z"
draft: false # Is this a draft? true/false
toc: true # Show table of contents? true/false
type: docs # Do not modify.

menu:
  core_fa19:
    parent: Fall 2019
    weight: 2

weight: 2

authors: ["Liam Jarvis", "John Muchovej"]

urls:
  youtube: "#"
  slides:  "#"
  github:  "#"
  kaggle:  "#"
  colab:   "#"

categories: ["fa19"]
tags: ["Regresssion", "Linear Regression", "Logistic Regression", "non-nn"]
description: >-
  You always start with the basics, and in this meeting we are doing just that! We will be going over what Machine Learning consists of and how we can use models to do awesome stuff!
---

First thing first, we to get some packages

- matplotlib allows us to graph
- numpy is powerful package for data manipulation
- pandas is a tool for allowing us to interact with large datasets
- sklearn is what we'll use for making the models
- !wget grabs the data set we'll be using later

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># import some important stuff</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">scipy.stats</span> <span class="k">as</span> <span class="nn">st</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">datasets</span><span class="p">,</span> <span class="n">linear_model</span>
</pre></div>

## Basic Example

The data for this example is arbitrary (we'll use real data in a bit), but there is a clear linear relationship here

Graphing the data will make this relationship clear to see

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Get some data </span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">7</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">9</span><span class="p">])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">7</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">9</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">12</span><span class="p">])</span>

<span class="c1"># Let&#39;s plot the data to see what it looks like</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s2">&quot;black&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

</pre></div>

Here's the meat of the calculations

This is using least squares estimation, which tries to minimize the squared error of the function vs. the training data

SS_xy is the cross deviation about x, and SS_xx is the deviation about x

[It's basically some roundabout algebra methods to optimize a function](https://www.amherst.edu/system/files/media/1287/SLR_Leastsquares.pdf)

The concept isn't super complicated but it gets hairy when you do it by hand

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># calculating the coefficients</span>

<span class="c1"># number of observations/points </span>
<span class="n">n</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>

<span class="c1"># mean of x and y vector </span>
<span class="n">m_x</span><span class="p">,</span> <span class="n">m_y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">y</span><span class="p">)</span>

<span class="c1"># calculating cross-deviation and deviation about x </span>
<span class="n">SS*xy</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">y</span><span class="o">*</span><span class="n">x</span> <span class="o">-</span> <span class="n">n</span><span class="o">_</span><span class="n">m_y</span><span class="o">_</span><span class="n">m*x</span><span class="p">)</span>
<span class="n">SS_xx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">x</span><span class="o">*</span><span class="n">x</span> <span class="o">-</span> <span class="n">n</span><span class="o">_</span><span class="n">m_x</span><span class="o">_</span><span class="n">m_x</span><span class="p">)</span>

<span class="c1"># calculating regression coefficients </span>
<span class="n">b_1</span> <span class="o">=</span> <span class="n">SS_xy</span> <span class="o">/</span> <span class="n">SS_xx</span>
<span class="n">b_0</span> <span class="o">=</span> <span class="n">m_y</span> <span class="o">-</span> <span class="n">b_1</span><span class="o">\*</span><span class="n">m_x</span>

<span class="c1">#var to hold the coefficients</span>
<span class="n">b</span> <span class="o">=</span> <span class="p">(</span><span class="n">b_0</span><span class="p">,</span> <span class="n">b_1</span><span class="p">)</span>

<span class="c1">#print out the estimated coefficients</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Estimated coefficients:</span><span class="se">\n</span><span class="s2">b_0 = </span><span class="si">{}</span><span class="s2"> </span><span class="se">\n</span><span class="s2">b_1 = </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">b</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">b</span><span class="p">[</span><span class="mi">1</span><span class="p">]))</span>

</pre></div>

But, we don't need to directly program all of the maths everytime we do linear regression

sklearn has built in functions that allows you to quickly do Linear Regression with just a few lines of code

We're going to use sklearn to make a model and then plot it using matplotlib

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Sklearn learn require this shape</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">y</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>

<span class="c1"># making the model</span>
<span class="n">regress</span> <span class="o">=</span> <span class="n">linear_model</span><span class="o">.</span><span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">regress</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>

</pre></div>

### And now, lets see what the model looks like

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plotting the actual points as scatter plot </span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s2">&quot;black&quot;</span><span class="p">,</span>
           <span class="n">marker</span> <span class="o">=</span> <span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="mi">30</span><span class="p">)</span>

<span class="c1"># predicted response vector </span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">b</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">+</span> <span class="n">b</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">\*</span><span class="n">x</span>

<span class="c1"># plotting the regression line </span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s2">&quot;blue&quot;</span><span class="p">)</span>

<span class="c1"># putting labels </span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;x&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;y&#39;</span><span class="p">)</span>

<span class="c1"># function to show plot </span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

</pre></div>

### So now we can make predictions with new points based off our data

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># here we can try out any data point</span>
<span class="nb">print</span><span class="p">(</span><span class="n">regress</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mi">6</span><span class="p">]]))</span>
</pre></div>

---

## Applied Linear Regression

---

### The Ames Housing Dataset

> Ames is a city located in Iowa.
>
> - This data set consists of all property sales
>   collected by the Ames City Assessor’s Office between the years
>   of 2006 and 2010.
> - Originally contained 113 variables and 3970 property sales
>   pertaining to the sale of stand-alone garages, condos, storage
>   areas, and of course residential property.
> - Distributed to the public as a means to replace the old Boston
>   Housing 1970’s data set.
> - [Link to Original](http://lib.stat.cmu.edu/datasets/boston)
> - The "cleaned" version of this dataset contains 2930 observations along with 80
>   predictor variables and two identification variables.

### What was the original purpose of this data set?

Why did the City of Ames decide to collect this data?

What does the prices of houses affect?

### What's inside?

This ”new” data set contains 2930 (n=2930) observations along with 80
predictor variables and two identification variables.

[Paper linked to dataset](http://jse.amstat.org/v19n3/decock.pdf)

An exhaustive variable breakdown can be found
[here](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)

### _Quick Summary_

---

Of the 80 predictor variables we have:

> - 20 continuous variables (area dimension)

- Garage Area, Wood Deck Area, Pool Area
  > - 14 discrete variables (items occurring)
- Remodeling Dates, Month and Year Sold
  > - 23 nominal and 23 ordinal
- Nominal: Condition of the Sale, Type of Heating and
  Foundation
- Ordinal: Fireplace and Kitchen Quality, Overall
  Condition of the House

### _Question to Answer:_

What is the linear relationship between sale price on above ground
living room area?

But first lets visually investigate what we are trying to predict.

We shall start our analysis with summary statistics.

<div class=" highlight hl-ipython3"><pre><span></span><span class="n">housing_data</span> <span class="o">=</span>  <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;AmesHousing.txt&quot;</span><span class="p">,</span> <span class="n">delimiter</span><span class="o">=</span><span class="s2">&quot;</span><span class="se">\t</span><span class="s2">&quot;</span><span class="p">)</span>

<span class="c1"># Mean Sales price </span>
<span class="n">mean_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Mean Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">mean_price</span><span class="p">))</span>

<span class="c1"># Variance of the Sales Price </span>
<span class="n">var_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">var</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">],</span> <span class="n">ddof</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Variance of Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">var_price</span><span class="p">))</span>

<span class="c1"># Median of Sales Price </span>
<span class="n">median_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Median Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">median_price</span><span class="p">))</span>

<span class="c1"># Skew of Sales Price </span>
<span class="n">skew_price</span> <span class="o">=</span> <span class="n">st</span><span class="o">.</span><span class="n">skew</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Skew of Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">skew_price</span><span class="p">))</span>

<span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>

</pre></div>

<div class=" highlight hl-ipython3"><pre><span></span><span class="n">housing_data</span> <span class="o">=</span>  <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;AmesHousing.txt&quot;</span><span class="p">,</span> <span class="n">delimiter</span><span class="o">=</span><span class="s2">&quot;</span><span class="se">\t</span><span class="s2">&quot;</span><span class="p">)</span>

<span class="c1"># Mean Sales price </span>
<span class="n">mean_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Mean Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">mean_price</span><span class="p">))</span>

<span class="c1"># Variance of the Sales Price </span>
<span class="n">var_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">var</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">],</span> <span class="n">ddof</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Variance of Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">var_price</span><span class="p">))</span>

<span class="c1"># Median of Sales Price </span>
<span class="n">median_price</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Median Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">median_price</span><span class="p">))</span>

<span class="c1"># Skew of Sales Price </span>
<span class="n">skew_price</span> <span class="o">=</span> <span class="n">st</span><span class="o">.</span><span class="n">skew</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Skew of Sales Price : &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">skew_price</span><span class="p">))</span>

<span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>

</pre></div>

### Another way we can view our data is with a box and whisker plot.

<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Sales Price&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

### Now we shall look at sales price on above ground living room area.

<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;Gr Liv Area&quot;</span><span class="p">],</span> <span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Sales Price&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

### Finally, lets generate our model and see how it predicts Sales Price!!

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># we need to reshape the array to make the sklearn gods happy</span>
<span class="n">area_reshape</span> <span class="o">=</span> <span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;Gr Liv Area&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="n">price_reshape</span> <span class="o">=</span> <span class="n">housing_data</span><span class="p">[</span><span class="s2">&quot;SalePrice&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>

<span class="c1"># Generate the Model</span>
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

---

## **Applied Logistic Regression**

---

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># we&#39;re going to need a different model, so let&#39;s import it</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="kn">import</span> <span class="n">LogisticRegression</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
</pre></div>

for Logistic Regression, we're going to be using a real dataset

This data set was provided by UCI's Machine Learning Repository:

- [Adult Data Set (Also know as Census Income)](https://archive.ics.uci.edu/ml/datasets/Adult)

We already downloaded the dataset at the begining of the notebook, so now let's mess around with it.

but before that, we need to read in the data. pandas has the functions we need to do this

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># read_csv allow us to easily import a whole dataset</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;adult.data&quot;</span><span class="p">,</span> <span class="n">names</span> <span class="o">=</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">,</span><span class="s2">&quot;workclass&quot;</span><span class="p">,</span><span class="s2">&quot;fnlwgt&quot;</span><span class="p">,</span><span class="s2">&quot;education&quot;</span><span class="p">,</span><span class="s2">&quot;education-num&quot;</span><span class="p">,</span><span class="s2">&quot;marital-status&quot;</span><span class="p">,</span><span class="s2">&quot;occupation&quot;</span><span class="p">,</span><span class="s2">&quot;relationship&quot;</span><span class="p">,</span><span class="s2">&quot;race&quot;</span><span class="p">,</span><span class="s2">&quot;sex&quot;</span><span class="p">,</span><span class="s2">&quot;capital-gain&quot;</span><span class="p">,</span><span class="s2">&quot;capital-loss&quot;</span><span class="p">,</span><span class="s2">&quot;hours-per-week&quot;</span><span class="p">,</span><span class="s2">&quot;native-country&quot;</span><span class="p">,</span><span class="s2">&quot;income&quot;</span><span class="p">])</span>

<span class="c1"># this tells us whats in it </span>
<span class="nb">print</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">info</span><span class="p">())</span>

</pre></div>

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># data.head() gives us some the the first 5 sets of the data</span>
<span class="n">data</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

The code below will show us some information about the _continunous_ parameters that our dataset contains.

- Age is Age

- fnlwgt is final weight, or the number of people that are represented in this group relative to the overall population of this dataset.

- Education-num is a numerical way of representing Education level

- Capital Gain is the money made investments

- Capital Loss is the loss from investments

- Hours-per-Week is the number of hours worked during a week

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># this is the function that give us some quick info about continous data in the dataset</span>
<span class="n">data</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

Now here is the Qustion:

- Which one of these parameters are best in figuring out if someone is going to be making more then 50k a year?
- Make sure you choose a continunous parameter, as categorical stuff isn't going to work

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># put the name of the parameter you want to test</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">test</span> <span class="o">=</span> <span class="s2">&quot;capital-gain&quot;</span>
<span class="c1">### END SOLUTION</span>
</pre></div>

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># but before we make our model, we need to modify our data a bit</span>

<span class="c1"># little baby helper function</span>
<span class="k">def</span> <span class="nf">incomeFixer</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
<span class="k">if</span> <span class="n">x</span> <span class="o">==</span> <span class="s2">&quot; &lt;=50K&quot;</span><span class="p">:</span>
<span class="k">return</span> <span class="mi">0</span>
<span class="k">else</span><span class="p">:</span>
<span class="k">return</span> <span class="mi">1</span>

<span class="c1"># change the income data into 0&#39;s and 1&#39;s</span>
<span class="n">data</span><span class="p">[</span><span class="s2">&quot;income&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">row</span><span class="p">:</span> <span class="n">incomeFixer</span><span class="p">(</span><span class="n">row</span><span class="p">[</span><span class="s1">&#39;income&#39;</span><span class="p">]),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="c1"># get the data we are going to make the model with </span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">test</span><span class="p">])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="s2">&quot;income&quot;</span><span class="p">])</span>

<span class="c1"># again, lets make the scikitlearn gods happy</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>

<span class="c1"># Making the test-train split</span>
<span class="n">x_train</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">x</span> <span class="p">,</span><span class="n">y</span> <span class="p">,</span><span class="n">test_size</span><span class="o">=</span><span class="mf">0.25</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>

</pre></div>

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># now make data model!</span>
<span class="n">logreg</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">solver</span><span class="o">=</span><span class="s1">&#39;liblinear&#39;</span><span class="p">)</span>
<span class="n">logreg</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>
</pre></div>

<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># now need to test the model&#39;s performance</span>
<span class="nb">print</span><span class="p">(</span><span class="n">logreg</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">x_test</span><span class="p">,</span><span class="n">y_test</span><span class="p">))</span>
</pre></div>

## Thank You

We hope that you enjoyed being here today.

Please fill out [this questionaire](https://docs.google.com/forms/d/e/1FAIpQLSe8kucGh3_2Dcm7BFPv89qy-F4_BZKF-_Jm0nie37Ty6FuL9g/viewform?usp=sf_link) so we can get some feedback about tonight's meeting.

We hope to see you here again next week for our core meeting on _Random Forests and Support Vector Machines_.

### Live in Virtue
