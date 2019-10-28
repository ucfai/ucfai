---
layout: "meeting"
title: "Answering the Important Question: Where's My Money? Part 2"
date: "2019-10-10"
authors:
    - "causallycausal"
    
categories: ['fa19']
tags: ['machine-learning', 'data science', 'finance', 'statistics']
description: >-
    We continue our deep dive into the wonderful world of loan defaults. Does the loan grading of creditors line up with what reality shows? Use machine learning to find out!
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="n">train</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;../input/ucfai-dsg-fa19-default/train.csv&quot;</span><span class="p">)</span>
<span class="n">test</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;../input/ucfai-dsg-fa19-default/test.csv&quot;</span><span class="p">)</span>
<span class="n">ID_test</span> <span class="o">=</span> <span class="n">test</span><span class="p">[</span><span class="s1">&#39;id&#39;</span><span class="p">]</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;GOOD_STANDING&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># So there are 9x as many good loans as bad (naturally, any reputable lender would avoid bad loans)</span>
<span class="c1"># This is problomatic, because most models will notice that most features are associated with good loans</span>
<span class="c1"># Therefore, they will most likely just predict all good loans. Why is this a problem?</span>

<span class="c1"># The score for this comp is an AUC ROC metric. In an oversimplified sense, this score is based on both</span>
<span class="c1"># how precise your positives are AND your negatives</span>
<span class="c1"># If you guess on either of them, you should expect the lowest score (0.5)</span>

<span class="c1"># There are almost 1 million examples, it is safe to undersample</span>
<span class="c1"># Undersampling is basically where we only use a subset of the training data so that our good loans/bad loans are equal</span>
<span class="c1"># The simple solution to this is just to randomly choose good loans to use until we are equal to bad loans</span>
<span class="c1"># Here is how we are going to undersample</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>

<span class="c1"># Give me the -length - of the subset of -train- made up of entries with GOOD_STANDING == 0 </span>
<span class="c1"># In otherwords, how many bad loans are there?</span>
<span class="n">bad_standing_len</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">train</span><span class="p">[</span><span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">])</span>

<span class="c1"># Give me the index of the subset of train where good_standing == 1 </span>
<span class="c1"># In otherwords, give me the index of all the good loans</span>
<span class="n">good_standing_index</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;GOOD_STANDING&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">index</span>

<span class="c1"># Randomly choose indices of good loans equal to the number of bad loans</span>
<span class="n">random_index</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">choice</span><span class="p">(</span><span class="n">good_standing_index</span><span class="p">,</span> <span class="n">bad_standing_len</span><span class="p">,</span> <span class="n">replace</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

<span class="c1"># Give me the index of all the bad loans in train</span>
<span class="n">bad_standing_index</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;GOOD_STANDING&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">index</span>

<span class="c1"># Concatonate the indices of bad loans, and our randomly sampled good loans</span>
<span class="n">under_sample_index</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">([</span><span class="n">bad_standing_index</span><span class="p">,</span> <span class="n">random_index</span><span class="p">])</span>

<span class="c1"># Create a new pandas dataframe made only of these indices </span>
<span class="n">under_sample</span> <span class="o">=</span> <span class="n">train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">under_sample_index</span><span class="p">]</span>

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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># Make sure it works, and make this undersampled dataframe our train</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;GOOD_STANDING&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">under_sample</span><span class="p">[</span><span class="s1">&#39;GOOD_STANDING&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">train</span> <span class="o">=</span> <span class="n">under_sample</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># As we did in Titanic, lets concatonate train and test</span>
<span class="n">train</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
<span class="n">train_len</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">train</span><span class="p">)</span>
<span class="n">dataset</span> <span class="o">=</span>  <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">(</span><span class="n">objs</span><span class="o">=</span><span class="p">[</span><span class="n">train</span><span class="p">,</span> <span class="n">test</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">nan</span><span class="p">)</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># There are a lot of columns and a lot of nulls, so I&#39;m going to just delete features that have more than 20% of the data missing and go from there</span>

<span class="n">null_list</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="k">for</span> <span class="n">column</span><span class="p">,</span> <span class="n">missing_num</span> <span class="ow">in</span> <span class="n">null_list</span><span class="o">.</span><span class="n">iteritems</span><span class="p">():</span>
    <span class="k">if</span> <span class="n">column</span> <span class="o">!=</span> <span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">missing_num</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">index</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mf">0.2</span><span class="p">:</span>
            <span class="n">dataset</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="n">column</span><span class="p">],</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># Since &quot;sub grade&quot; exists, grade is kind of redundant, let&#39;s just get rid of grade</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;grade&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># We&#39;re also going to remove issue date because, as we discussed last week, the issue date</span>
<span class="c1"># between the train and the test is unbalanced. Therefore, there is likely not much to learn from it</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;issue_d&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>


<span class="c1"># I&#39;m also going to remove employee title. This might seem problamatic, but consider two things</span>
<span class="c1"># We already have annual income, really how much more info can we gloss from this?</span>
<span class="c1"># If we have to turn these into dummy variables, as we tend to do, there are a LOT of different titles, they will be sparse</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;emp_title&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># Under most circumstances, you want to go through and replace nulls intelligently, but to give you an idea on how to efficiently clean</span>
<span class="c1"># this dataset, let&#39;s try replacing all continious values with the mean, and all categorical values with the mode</span>

<span class="n">number_set</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">_get_numeric_data</span><span class="p">()</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span><span class="n">j</span> <span class="ow">in</span> <span class="n">dataset</span><span class="o">.</span><span class="n">iteritems</span><span class="p">():</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Now handeling&quot;</span><span class="p">,</span> <span class="n">i</span><span class="p">)</span>
    <span class="c1"># Let&#39;s break this down (it was also used in titanic)</span>
    <span class="c1"># For each column in the dataset, take the subset of that column made up of null entries for that column</span>
    <span class="c1"># Then, take that subset&#39;s indices and transform it into a list</span>
    <span class="n">NaN_index</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="n">dataset</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="n">dataset</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()]</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>
    <span class="c1"># Skip the target variable obviously</span>
    <span class="k">if</span> <span class="p">(</span><span class="n">i</span> <span class="o">==</span> <span class="s1">&#39;GOOD_STANDING&#39;</span><span class="p">):</span>
        <span class="k">continue</span>
    <span class="k">if</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">number_set</span><span class="p">:</span>
        <span class="c1"># If we are dealing with numerial values, take the median</span>
        <span class="n">med</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">NaN_index</span><span class="p">:</span>
            <span class="c1">#print(&quot;did I get here&quot;)</span>
            <span class="n">dataset</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">x</span><span class="p">]</span> <span class="o">=</span> <span class="n">med</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="c1"># Otherwise, just take the most frequent categorical value </span>
        <span class="n">mode</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">idxmax</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">NaN_index</span><span class="p">:</span>
           <span class="c1"># print(&quot;what about here&quot;)</span>
            <span class="n">dataset</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">x</span><span class="p">]</span> <span class="o">=</span> <span class="n">mode</span>
            
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># We&#39;re going to drop a couple categorical values that have way to many possible values</span>
<span class="c1"># There are certainly ways you can utilize this, expecially the dates, but for the time being we will remove them</span>
<span class="c1"># If there are too many possible values, get_dummies creates too many new columns</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;earliest_cr_line&#39;</span><span class="p">,</span> <span class="s1">&#39;last_credit_pull_d&#39;</span><span class="p">,</span> <span class="s1">&#39;last_pymnt_d&#39;</span><span class="p">,</span> <span class="s1">&#39;addr_state&#39;</span><span class="p">,</span> <span class="s1">&#39;title&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="n">categorical_features</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">set</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span> <span class="o">-</span> <span class="nb">set</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">_get_numeric_data</span><span class="p">()</span><span class="o">.</span><span class="n">columns</span><span class="p">))</span>

<span class="nb">print</span><span class="p">(</span><span class="n">categorical_features</span><span class="p">)</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">dataset</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">categorical_features</span><span class="p">)</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
<span class="n">dataset</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># Separate train and test</span>
<span class="n">train</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[:</span><span class="n">train_len</span><span class="p">]</span>
<span class="n">test</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[</span><span class="n">train_len</span><span class="p">:]</span>
<span class="c1"># Drop the good standing from test (which should all be empty)</span>
<span class="n">test</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">labels</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">],</span><span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># Make sure they are ints</span>
<span class="n">train</span><span class="p">[</span><span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>

<span class="n">Y_train</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">]</span>

<span class="n">X_train</span> <span class="o">=</span> <span class="n">train</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">labels</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">],</span><span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="c1"># Let&#39;s jus tuse a basic random forest</span>
<span class="n">RF</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">()</span>
<span class="n">RF</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">)</span>
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
        <div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="n">test_standing</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">RF</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test</span><span class="p">),</span> <span class="n">name</span><span class="o">=</span><span class="s2">&quot;GOOD_STANDING&quot;</span><span class="p">)</span>

<span class="n">results</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">ID_test</span><span class="p">,</span><span class="n">test_standing</span><span class="p">],</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="n">results</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s2">&quot;GradePrediction.csv&quot;</span><span class="p">,</span><span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="c1">### END SOLUTION</span>
</pre></div>

    </div>
</div>
</div>

</div>
  
 


{% endraw %}