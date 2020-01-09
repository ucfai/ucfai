<img src="https://ucfai.org/core/fa19/2019-09-25-rf-svm/rf-svm/banner.png">

<div class="col-12">
    <span class="btn btn-success btn-block">
        Meeting in-person? Have you signed in?
    </span>
</div>

<div class="col-12">
    <h1> A Walk Through the Random Forest </h1>
    <hr>
</div>

<div style="line-height: 2em;">
    <p>by: 
        <strong> Liam Jarvis</strong>
        (<a href="https://github.com/jarviseq">@jarviseq</a>)
     on 2019-09-25</p>
</div>



<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">os</span>
<span class="kn">from</span> <span class="nn">pathlib</span> <span class="kn">import</span> <span class="n">Path</span>

<span class="k">if</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">exists</span><span class="p">(</span><span class="s2">&quot;/kaggle/input&quot;</span><span class="p">):</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;/kaggle/input&quot;</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s2">&quot;We don&#39;t know this machine.&quot;</span><span class="p">)</span>
</pre></div>



## Overview

Before getting going on more complex examples, 
we're going to take a look at a very simple example using the Iris Dataset. 

The final example deals with credit card fraud, 
and how to identify if fraud is taking place based a dataset of over 280,000 entries. 




<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Importing the important stuff</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">svm</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">RandomForestClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">matthews_corrcoef</span>
</pre></div>



## Iris Data Set

This is a classic dataset of flowers. The goal is to have the model classify the types of flowers based on 4 factors. 
Those factors are sepal length, sepal width, petal length, and petal width, which are all measured in cm. 
The dataset is very old in comparison to many of the datasets we use, coming from a 
[1936 paper about taxonomy](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x).

### Getting the Data Set

Sklearn has the dataset built into the the library, so getting the data will be easy.
Once we do that, we'll do a test-train split.




<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="kn">import</span> <span class="n">load_iris</span>

<span class="n">iris</span> <span class="o">=</span> <span class="n">load_iris</span><span class="p">()</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">,</span> <span class="n">Y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.1</span><span class="p">)</span>
</pre></div>



### Making the Model

Making and Random Forests model is very easy, taking just two lines of code! 
Training times can take a second, but in this example is small so the training time is minimal.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">trees</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">150</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
<span class="n">trees</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">)</span>
</pre></div>



sklearn has a few parameters that we can tweak to tune our model. 
We won't be going into those different parameters in this notebook, 
but if you want to give it a look, 
[here is the documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  

### We need to Figure out how well this model does

There are a few ways we are going to test for accuracy using a Confusion Matrix and Matthews Correlation Coefficient . 

#### Confusion Matrix

A Confusion Matrix shows us where the model is messing up. Below is an example from dataschool.io. The benefit of a confusion matrix is that it is a very easy way to visualise the performance of the model. 

![alt text](https://www.dataschool.io/content/images/2015/01/confusion_matrix_simple2.png)



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">predictions</span> <span class="o">=</span> <span class="n">trees</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">confusion_matrix</span><span class="p">(</span><span class="n">Y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">)</span>
</pre></div>



#### Matthews correlation coefficient

This is used to find the quality of binary classification. 
It is based on the values found in the Confusion Matrix and tries to take those values and boil it down to one number. 
It is generally considered one of the better measures of quality for classification. 
MCC does not rely on class size, so in cases where we have very different class sizes, 
we can get a realiable measure of how well it did. 


Matthews correlation coefficient ranges from -1 to 1. 
-1 represents total disagreement between the prediction and the observation, 
while 1 represents prefect prediction. 
In other words, the closer to 1 we get the better the model is considered. 



<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">matthews_corrcoef</span><span class="p">(</span><span class="n">Y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>



### Now, what about SVMs?

We want to see how well SVMs can work on the Iris, so let's see it in action.

First, let's define the models; one for linear, ploy and rbf.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># SVM regularization parameter, we&#39;ll keep it simple for now</span>
<span class="n">C</span> <span class="o">=</span> <span class="mf">1.0</span> 

<span class="n">models</span> <span class="o">=</span> <span class="p">(</span><span class="n">svm</span><span class="o">.</span><span class="n">SVC</span><span class="p">(</span><span class="n">kernel</span><span class="o">=</span><span class="s1">&#39;linear&#39;</span><span class="p">,</span> <span class="n">C</span><span class="o">=</span><span class="n">C</span><span class="p">),</span>
          <span class="n">svm</span><span class="o">.</span><span class="n">SVC</span><span class="p">(</span><span class="n">kernel</span><span class="o">=</span><span class="s1">&#39;poly&#39;</span><span class="p">,</span> <span class="n">degree</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">C</span><span class="o">=</span><span class="n">C</span><span class="p">),</span>
          <span class="n">svm</span><span class="o">.</span><span class="n">SVC</span><span class="p">(</span><span class="n">kernel</span><span class="o">=</span><span class="s1">&#39;rbf&#39;</span><span class="p">,</span> <span class="n">gamma</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span> <span class="n">C</span><span class="o">=</span><span class="n">C</span><span class="p">))</span>
</pre></div>



So you know what the parameters mean:
* degree refers to the degree of the polynomial
* gamma refer to the influence of a single training example reaches
> * with low values meaning ‘far’ and high values meaning ‘close’

Once we have the models defined, let's train them!



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">models</span> <span class="o">=</span> <span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">)</span> <span class="k">for</span> <span class="n">clf</span> <span class="ow">in</span> <span class="n">models</span><span class="p">)</span>
</pre></div>



Now we are see how the confusion matrices look like:



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">for</span> <span class="n">clf</span> <span class="ow">in</span> <span class="n">models</span><span class="p">:</span>
    <span class="n">predictions</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">Y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>



The confusion matrix is all nice and dandy, 
but let's check out what the Matthews Coefficient has to say about our models.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">for</span> <span class="n">clf</span> <span class="ow">in</span> <span class="n">models</span><span class="p">:</span>
    <span class="n">predictions</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">matthews_corrcoef</span><span class="p">(</span><span class="n">Y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>



That wasn't too bad was it? 
Both Random Forests and SVMs are very easy models to implement,
and its low training times means that the model can be used without the overheads associated with neural networks, 
which we will learn more about next week.

## Credit Card Fraud Dataset

As always, we are going to need a dataset to work on!
Credit card fraud detection is a serious issue, and as such is something that data scientists have looked into. This dataset is from a Kaggle competition with over 2,000 Kernals based on it. Let's see how well Random Forests can do with this dataset!

Lets read in the data and use *.info()* to find out some meta-data



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">DATA_DIR</span> <span class="o">/</span> <span class="s2">&quot;train.csv&quot;</span><span class="p">)</span>
<span class="n">data</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>



What's going on with this V stuff?
Credit Card information is a bit sensitive, and as such raw information had to be obscured in some way to protect that information.

In this case, the data provider used a method know as PCA transformation to hide those features that would be considered sensitive. 
PCA is a very useful technique for dimension reduction. 
For now, know that this technique allows us to take data and transform it in a way that maintains the patterns in that data.

If you are interested in learning more about PCA, 
[Consider checking out this article](https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-b836fb9c97e2). 
Unfortunately, there is a lot that we want to cover in Core and not enough time to do all of it. :(

Next, let's get some basic statistical info from this data.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">data</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>



### Some important points about this data 

For most of the features, there is not a lot we can gather since it's been obscured with PCA. 
However there are three features that have been left in for us to see. 

#### 1. Time

Time is the amount of time from first transaction in seconds. 
The max is 172792, so the data was collected for around 48 hours. 

#### 2. Amount

Amount is the amount that the transaction was for. 
The denomination of the transactions was not given, so we're going to be calling them "Simoleons" as a place holder. 
Some interesting points about this feature is the STD, or standard deviation, which is 250§. That's quite large, 
but makes sense when the min and max are 0§ and 25,691§ respectively. 
There is a lot of variance in this feature, which is to be expected. The 75% amount is only in 77§, 
which means that the whole set skews to the lower end.

#### 3. Class

This tells us if the transaction was fraud or not. 0 is for no fraud, 1 if it is fraud. 
If you look at the mean, it is .001727, which means that only .1727% of the 284,807 cases are fraud. 
There are only 492 fraud examples in this data set, so we're looking for a needle in a haystack in this case.

Now that we have that out of the way, let's start making this model! We need to split our data first



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="s1">&#39;Class&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Class&#39;</span><span class="p">]</span>

<span class="c1"># sklearn requires a shape with dimensions (N, 1), </span>
<span class="c1"># so we expand dimensions of x and y to put a 1 in the second dimension</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;X shape: </span><span class="si">{X.shape}</span><span class="s2"> Y shape: </span><span class="si">{Y.shape}</span><span class="s2">&quot;</span><span class="p">)</span>
</pre></div>



### Some Points about Training

With Random Forest and SVMs, training time is very quick, so we can finish training the model in realitively short order, 
even when our dataset contains 284,807 entries. This is done without the need of GPU acceleration, 
which Random Forests cannot take advantage of.

The area is left blank, 
but there's examples on how to make the models earlier in the notebook that can be used as an example if you need it. 
What model and the parameters you choose are up to you, so have fun!



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Make the magic happen!</span>
<span class="c1"># https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html</span>
<span class="n">X_test</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">DATA_DIR</span> <span class="o">/</span> <span class="s2">&quot;test.csv&quot;</span><span class="p">)</span>

<span class="c1"># to expedite things: pass `n_jobs=-1` so you can run across all available CPUs</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">rf</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
<span class="n">rf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">)</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">rf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="o">.</span><span class="n">values</span><span class="p">)</span>
<span class="c1">### END SOLUTION</span>
</pre></div>



### Submitting your Solution

To submit your solution to kaggle, you'll need to save you data.

Luckily, we got the code you need to do that below. 



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">predictions</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;Id&#39;</span><span class="p">:</span> <span class="n">Y_test</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="s1">&#39;Class&#39;</span><span class="p">:</span> <span class="n">predictions</span><span class="p">})</span>

<span class="n">predictions</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;submission.csv&#39;</span><span class="p">,</span> <span class="n">header</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Id&#39;</span><span class="p">,</span> <span class="s1">&#39;Class&#39;</span><span class="p">],</span> <span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>



## Thank You

We hope that you enjoyed being here today.

Please fill out [this questionaire](https://docs.google.com/forms/d/e/1FAIpQLSemUFE7YNrnKYT-KBUJcsWbmNkBIj_1aT0mtw3LszJLOMAXLA/viewform?usp=sf_link) so we can get some feedback about tonight's meeting.

We hope to see you here again next week for our Core meeting on *Neural Networks*.

### Vevu en Virtu 

