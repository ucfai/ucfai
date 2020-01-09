<img src="https://ucfai.org/core/fa19/2019-10-23-rnns/rnns/banner.png">

<div class="col-12">
    <span class="btn btn-success btn-block">
        Meeting in-person? Have you signed in?
    </span>
</div>

<div class="col-12">
    <h1> Writer's Block? RNNs can help! </h1>
    <hr>
</div>

<div style="line-height: 2em;">
    <p>by: 
        <strong> Brandon</strong>
        (<a href="https://github.com/brandons209">@brandons209</a>)
    
        <strong> John Muchovej</strong>
        (<a href="https://github.com/ionlights">@ionlights</a>)
     on 2019-10-23</p>
</div>



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># This is a bit of code to make things work on Kaggle</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">from</span> <span class="nn">pathlib</span> <span class="kn">import</span> <span class="n">Path</span>

<span class="k">if</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">exists</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-rnns&quot;</span><span class="p">):</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;/kaggle/input/ucfai-core-fa19-rnns&quot;</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="n">DATA_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="s2">&quot;data/&quot;</span><span class="p">)</span>
</pre></div>



# Generate new Simpson scripts with LSTM RNN
## Link to slides [here](https://docs.google.com/presentation/d/1ztu3_4xsuWH1FAsGqnGJkP_TBKS3Q27FPUMFkraMB34/edit?usp=sharing)
In this project, we will be using an LSTM with the help of an Embedding layer to train our network on an episode from the Simpsons, specifically the episode "Moe's Tavern". This is taken from [this](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) dataset on kaggle. This model can be applied to any text. We could use more episodes from the Simpsons, a book, articles, wikipedia, etc. It will learn the semantic word associations and being able to generate text in relation to what it is trained on.

First, lets import all of our libraries we need.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># general imports</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">import</span> <span class="nn">pickle</span>
<span class="kn">import</span> <span class="nn">glob</span>

<span class="c1"># torch imports</span>
<span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">torch.nn</span> <span class="k">as</span> <span class="nn">nn</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="k">as</span> <span class="nn">F</span>
<span class="kn">import</span> <span class="nn">torch.optim</span> <span class="k">as</span> <span class="nn">optim</span>
<span class="kn">import</span> <span class="nn">torch.backends.cudnn</span> <span class="k">as</span> <span class="nn">cudnn</span>
<span class="kn">from</span> <span class="nn">torch.utils.data</span> <span class="kn">import</span> <span class="n">TensorDataset</span><span class="p">,</span> <span class="n">DataLoader</span>

<span class="c1"># tensorboardX</span>
<span class="kn">from</span> <span class="nn">tensorboardX</span> <span class="kn">import</span> <span class="n">SummaryWriter</span>
</pre></div>



#### The cell below contains a bunch of helper functions for us to use today, dealing with string manipulation and printing out epoch results. Feel free to take a look after the workshop! 



<div class=" highlight hl-ipython3"><pre><span></span><span class="sd">&quot;&quot;&quot;</span>
<span class="sd">Loads text from specified path, exits program if the file is not found.</span>
<span class="sd">&quot;&quot;&quot;</span>
<span class="k">def</span> <span class="nf">load_script</span><span class="p">(</span><span class="n">path</span><span class="p">):</span>

    <span class="k">if</span> <span class="ow">not</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">isfile</span><span class="p">(</span><span class="n">path</span><span class="p">):</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Error! </span><span class="si">{}</span><span class="s2"> was not found.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">path</span><span class="p">))</span>
        <span class="n">sys</span><span class="o">.</span><span class="n">exit</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>

    <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">path</span><span class="p">,</span> <span class="s1">&#39;r&#39;</span><span class="p">)</span> <span class="k">as</span> <span class="n">file</span><span class="p">:</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">file</span><span class="o">.</span><span class="n">read</span><span class="p">()</span>
    <span class="k">return</span> <span class="n">text</span>
 
<span class="c1"># saves dictionary to file for use later</span>
<span class="k">def</span> <span class="nf">save_dict</span><span class="p">(</span><span class="nb">dict</span><span class="p">,</span> <span class="n">filename</span><span class="p">):</span>
    <span class="nb">dir</span> <span class="o">=</span> <span class="s1">&#39;data/dictionaries/&#39;</span> <span class="o">+</span> <span class="n">filename</span>
    <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="nb">dir</span><span class="p">,</span> <span class="s1">&#39;wb&#39;</span><span class="p">)</span> <span class="k">as</span> <span class="n">file</span><span class="p">:</span>
        <span class="n">pickle</span><span class="o">.</span><span class="n">dump</span><span class="p">(</span><span class="nb">dict</span><span class="p">,</span> <span class="n">file</span><span class="p">)</span>

<span class="c1"># loads dictionary from file</span>
<span class="k">def</span> <span class="nf">load_dict</span><span class="p">(</span><span class="n">filename</span><span class="p">):</span>
    <span class="nb">dir</span> <span class="o">=</span> <span class="s1">&#39;data/dictionaries/&#39;</span> <span class="o">+</span> <span class="n">filename</span>
    <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="nb">dir</span><span class="p">,</span> <span class="s1">&#39;rb&#39;</span><span class="p">)</span> <span class="k">as</span> <span class="n">file</span><span class="p">:</span>
         <span class="nb">dict</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">file</span><span class="p">)</span>
    <span class="k">return</span> <span class="nb">dict</span>

<span class="c1">#dictionaries for tokenizing puncuation and converting it back</span>
<span class="n">punctuation_to_tokens</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;!&#39;</span><span class="p">:</span><span class="s1">&#39; ||exclaimation_mark|| &#39;</span><span class="p">,</span> <span class="s1">&#39;,&#39;</span><span class="p">:</span><span class="s1">&#39; ||comma|| &#39;</span><span class="p">,</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">:</span><span class="s1">&#39; ||quotation_mark|| &#39;</span><span class="p">,</span>
                          <span class="s1">&#39;;&#39;</span><span class="p">:</span><span class="s1">&#39; ||semicolon|| &#39;</span><span class="p">,</span> <span class="s1">&#39;.&#39;</span><span class="p">:</span><span class="s1">&#39; ||period|| &#39;</span><span class="p">,</span> <span class="s1">&#39;?&#39;</span><span class="p">:</span><span class="s1">&#39; ||question_mark|| &#39;</span><span class="p">,</span> <span class="s1">&#39;(&#39;</span><span class="p">:</span><span class="s1">&#39; ||left_parentheses|| &#39;</span><span class="p">,</span>
                          <span class="s1">&#39;)&#39;</span><span class="p">:</span><span class="s1">&#39; ||right_parentheses|| &#39;</span><span class="p">,</span> <span class="s1">&#39;--&#39;</span><span class="p">:</span><span class="s1">&#39; ||dash|| &#39;</span><span class="p">,</span> <span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="p">:</span><span class="s1">&#39; ||return|| &#39;</span><span class="p">,</span> <span class="s1">&#39;:&#39;</span><span class="p">:</span><span class="s1">&#39; ||colon|| &#39;</span><span class="p">}</span>

<span class="n">tokens_to_punctuation</span> <span class="o">=</span> <span class="p">{</span><span class="n">token</span><span class="o">.</span><span class="n">strip</span><span class="p">():</span> <span class="n">punc</span> <span class="k">for</span> <span class="n">punc</span><span class="p">,</span> <span class="n">token</span> <span class="ow">in</span> <span class="n">punctuation_to_tokens</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span>

<span class="c1">#for all of the puncuation in replace_list, convert it to tokens</span>
<span class="k">def</span> <span class="nf">tokenize_punctuation</span><span class="p">(</span><span class="n">text</span><span class="p">):</span>
    <span class="n">replace_list</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;.&#39;</span><span class="p">,</span> <span class="s1">&#39;,&#39;</span><span class="p">,</span> <span class="s1">&#39;!&#39;</span><span class="p">,</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">,</span> <span class="s1">&#39;;&#39;</span><span class="p">,</span> <span class="s1">&#39;?&#39;</span><span class="p">,</span> <span class="s1">&#39;(&#39;</span><span class="p">,</span> <span class="s1">&#39;)&#39;</span><span class="p">,</span> <span class="s1">&#39;--&#39;</span><span class="p">,</span> <span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="p">,</span> <span class="s1">&#39;:&#39;</span><span class="p">]</span>
    <span class="k">for</span> <span class="n">char</span> <span class="ow">in</span> <span class="n">replace_list</span><span class="p">:</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">text</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="n">char</span><span class="p">,</span> <span class="n">punctuation_to_tokens</span><span class="p">[</span><span class="n">char</span><span class="p">])</span>
    <span class="k">return</span> <span class="n">text</span>

<span class="c1">#convert tokens back to puncuation</span>
<span class="k">def</span> <span class="nf">untokenize_punctuation</span><span class="p">(</span><span class="n">text</span><span class="p">):</span>
    <span class="n">replace_list</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;||period||&#39;</span><span class="p">,</span> <span class="s1">&#39;||comma||&#39;</span><span class="p">,</span> <span class="s1">&#39;||exclaimation_mark||&#39;</span><span class="p">,</span> <span class="s1">&#39;||quotation_mark||&#39;</span><span class="p">,</span>
                    <span class="s1">&#39;||semicolon||&#39;</span><span class="p">,</span> <span class="s1">&#39;||question_mark||&#39;</span><span class="p">,</span> <span class="s1">&#39;||left_parentheses||&#39;</span><span class="p">,</span> <span class="s1">&#39;||right_parentheses||&#39;</span><span class="p">,</span>
                    <span class="s1">&#39;||dash||&#39;</span><span class="p">,</span> <span class="s1">&#39;||return||&#39;</span><span class="p">,</span> <span class="s1">&#39;||colon||&#39;</span><span class="p">]</span>
    <span class="k">for</span> <span class="n">char</span> <span class="ow">in</span> <span class="n">replace_list</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">char</span> <span class="o">==</span> <span class="s1">&#39;||left_parentheses||&#39;</span><span class="p">:</span><span class="c1">#added this since left parentheses had an extra space</span>
            <span class="n">text</span> <span class="o">=</span> <span class="n">text</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span> <span class="o">+</span> <span class="n">char</span> <span class="o">+</span> <span class="s1">&#39; &#39;</span><span class="p">,</span> <span class="n">tokens_to_punctuation</span><span class="p">[</span><span class="n">char</span><span class="p">])</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">text</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span> <span class="o">+</span> <span class="n">char</span><span class="p">,</span> <span class="n">tokens_to_punctuation</span><span class="p">[</span><span class="n">char</span><span class="p">])</span>
    <span class="k">return</span> <span class="n">text</span>

<span class="sd">&quot;&quot;&quot;</span>
<span class="sd">Takes text already converted to ints and a sequence length and returns the text split into seq_length sequences and generates targets for those sequences</span>
<span class="sd">&quot;&quot;&quot;</span>
<span class="k">def</span> <span class="nf">gen_sequences</span><span class="p">(</span><span class="n">int_text</span><span class="p">,</span> <span class="n">seq_length</span><span class="p">):</span>
    <span class="n">seq_text</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">targets</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">int_text</span><span class="p">)</span> <span class="o">-</span> <span class="n">seq_length</span><span class="p">,</span> <span class="mi">1</span><span class="p">):</span>
        <span class="n">seq_in</span> <span class="o">=</span> <span class="n">int_text</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span> <span class="o">+</span> <span class="n">seq_length</span><span class="p">]</span>
        <span class="n">seq_out</span> <span class="o">=</span> <span class="n">int_text</span><span class="p">[</span><span class="n">i</span> <span class="o">+</span> <span class="n">seq_length</span><span class="p">]</span>
        <span class="n">seq_text</span><span class="o">.</span><span class="n">append</span><span class="p">([</span><span class="n">word</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">seq_in</span><span class="p">])</span>
        <span class="n">targets</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">seq_out</span><span class="p">)</span><span class="c1">#target is next word after the sequence</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">seq_text</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int32</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">targets</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">tabulate</span> <span class="kn">import</span> <span class="n">tabulate</span>

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
</pre></div>



## Dataset statistics
Before starting our project, we should take a look at the data we are dealing with. We are loading in a single episode from the Simpsons, but you can load in any other text from a `.txt` file. There is also an included Trump's Tweets dataset and a loop if you want to add multiple text files in at once.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">script_text</span> <span class="o">=</span> <span class="n">load_script</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">DATA_DIR</span> <span class="o">/</span> <span class="s1">&#39;moes_tavern_lines.txt&#39;</span><span class="p">))</span>
<span class="c1">#script_text = load_script(str(DATA_DIR / &#39;harry-potter.txt&#39;))</span>
<span class="c1"># if you want to load in your own data, add it a directory called data (as many text files as you want)</span>
<span class="c1"># and uncomment this here: (remember that these stats wont be accurate unless you use the simpsons dataset)</span>
<span class="c1"># spript_text = &quot;&quot;</span>
<span class="c1">#for script in sort(glob.glob(str(DATA_DIR))):</span>
<span class="c1">#    script_text += load_script(script)</span>

<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;----------Dataset Stats-----------&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Approximate number of unique words: </span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">({</span><span class="n">word</span><span class="p">:</span> <span class="kc">None</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">script_text</span><span class="o">.</span><span class="n">split</span><span class="p">()})))</span>
<span class="n">scenes</span> <span class="o">=</span> <span class="n">script_text</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;</span><span class="se">\n\n</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Number of scenes: </span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">scenes</span><span class="p">)))</span>
<span class="n">sentence_count_scene</span> <span class="o">=</span> <span class="p">[</span><span class="n">scene</span><span class="o">.</span><span class="n">count</span><span class="p">(</span><span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="p">)</span> <span class="k">for</span> <span class="n">scene</span> <span class="ow">in</span> <span class="n">scenes</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Average number of sentences in each scene: </span><span class="si">{:.0f}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">average</span><span class="p">(</span><span class="n">sentence_count_scene</span><span class="p">)))</span>

<span class="n">sentences</span> <span class="o">=</span> <span class="p">[</span><span class="n">sentence</span> <span class="k">for</span> <span class="n">scene</span> <span class="ow">in</span> <span class="n">scenes</span> <span class="k">for</span> <span class="n">sentence</span> <span class="ow">in</span> <span class="n">scene</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="p">)]</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Number of lines: </span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">sentences</span><span class="p">)))</span>
<span class="n">word_count_sentence</span> <span class="o">=</span> <span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">sentence</span><span class="o">.</span><span class="n">split</span><span class="p">())</span> <span class="k">for</span> <span class="n">sentence</span> <span class="ow">in</span> <span class="n">sentences</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Average number of words in each line: </span><span class="si">{:.0f}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">average</span><span class="p">(</span><span class="n">word_count_sentence</span><span class="p">)))</span>
</pre></div>



## Tokenize Text
In order to prepare our data for our network, we need to tokenize the words. That is, we will be converting every unique word and punctuation into an integer. Before we do that, we need to make the punctuation easier to convert to a number. For example, we will be taking any new lines and converting them to the word "||return||". This makes the text easier to tokenize and pass into our model. The functions that do this are in the helper functions code block above.

A note on tokenizing: 0 is a reserved integer, that is its not used to represent any words. So our integers for our words will start at 1. This is needed as when we use the model to generate new text, it needs a starting point, known as a seed. If this seed is smaller than our sequence length, then the function pad_sequences will pad that seed with 0's in order to represent "nothing". This help reduces noise in the network. Think of it as whitespace, it doesn't change the meaning to the input phrase.

This is the list of punctuation and special characters that are converted, notice that spaces are put before and after to make splitting the text easier:
- '!' : ' ||exclaimation_mark|| '
- ',' : ' ||comma|| '
- '"' : ' ||quotation_mark|| '
- ';' : ' ||semicolon|| '
- '.' : ' ||period|| '
- '?' : ' ||question_mark|| '
- '(' : ' ||left_parentheses|| '
- ')' : ' ||right_parentheses|| '
- '--' : ' ||dash|| '
- '\n' : ' ||return|| '
- ':' : ' ||colon|| '

We also convert all of the text to lowercase as this reduces the vocabulary list and trains the network faster.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">script_text</span> <span class="o">=</span> <span class="n">tokenize_punctuation</span><span class="p">(</span><span class="n">script_text</span><span class="p">)</span> <span class="c1"># helper function to convert non-word characters</span>
<span class="n">script_text</span> <span class="o">=</span> <span class="n">script_text</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>

<span class="n">script_text</span> <span class="o">=</span> <span class="n">script_text</span><span class="o">.</span><span class="n">split</span><span class="p">()</span> <span class="c1"># splits the text based on spaces into a list</span>
</pre></div>



## Creating Conversion Dictionaries and Input Data
Now that the tokens have been generated, we will create some dictionaries to convert our tokenized integers back to words, and words to integers. We will also generate our inputs and targets to pass into our model. 

To do this, we need to specify the sequence length, which is the amount of words we pass into the model at one time. I choose 12 for the average sentence length seen in Dataset Stats, but feel free to change this. A sequence length of 1 is just one word, so we could get better output depending on our sequence length. We use the helper function gen_sequences to do this for us.

The dataset and dataloader is defined using the specified batch_size.

The targets are simply just the next word in our text. So if we have a sentence: "Hi, how are you?" and we input "Hi, how are you" our target for this sentence will be "?".



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sequence_length</span> <span class="o">=</span> <span class="mi">12</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">64</span>

<span class="n">int_to_word</span> <span class="o">=</span> <span class="p">{</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">:</span> <span class="n">word</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">word</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">set</span><span class="p">(</span><span class="n">script_text</span><span class="p">))}</span>
<span class="n">word_to_int</span> <span class="o">=</span> <span class="p">{</span><span class="n">word</span><span class="p">:</span> <span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">int_to_word</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span> <span class="c1"># flip word_to_int dict to get int to word</span>
<span class="n">int_script_text</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">word_to_int</span><span class="p">[</span><span class="n">word</span><span class="p">]</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">script_text</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span> <span class="c1"># convert text to integers</span>
<span class="n">int_script_text</span><span class="p">,</span> <span class="n">targets</span> <span class="o">=</span> <span class="n">gen_sequences</span><span class="p">(</span><span class="n">int_script_text</span><span class="p">,</span> <span class="n">sequence_length</span><span class="p">)</span> <span class="c1"># transform int_script_text to sequences of sequence_length and generate targets</span>

<span class="n">vocab_size</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">word_to_int</span><span class="p">)</span> <span class="o">+</span> <span class="mi">1</span> <span class="c1"># add one since indexes are 1 to length</span>
<span class="c1"># convert to tensors and define dataset</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">TensorDataset</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">from_numpy</span><span class="p">(</span><span class="n">int_script_text</span><span class="p">),</span> <span class="n">torch</span><span class="o">.</span><span class="n">from_numpy</span><span class="p">(</span><span class="n">targets</span><span class="p">))</span>
<span class="c1"># define dataloader for the dataset</span>
<span class="n">dataloader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">dataset</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">num_workers</span><span class="o">=</span><span class="mi">4</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of vocabulary: </span><span class="si">{}</span><span class="s2">, Dataloader size: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">vocab_size</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="p">)))</span>
</pre></div>



## Building the Model
Here is the fun part, building our model. We will use LSTM cells and an Embedding layer, with a fully connected Linear layer at the end for the prediction. Documentation for LSTM cells can be found [here](https://pytorch.org/docs/stable/nn.html#lstm) and for embedding [here](https://pytorch.org/docs/stable/nn.html#embedding).

An LSTM network can be defined simply as:    
```
nn.LSTM(input_size, hidden_size, num_layers, dropout, batch_first=True)
```
- dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
           Introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
- hidden_size: Number of LSTM cells in each layer
- batch_first: Whether the first dimensions is batch_size or sequence_length. Leave this to true for our model, as batch_size is first.

Import to note is that the output of the LSTM network here has the shape of (seq_len, batch_size, hidden_size), so in our forward pass you need to use the `transpose` tensor method to swap the batch_size and seq_len axes before input into the Linear layer. Also, for input into the Linear layer from the LSTM layer requires the last step of the output. Remember, the LSTM network returns the output for **every state**, so make sure to get the last one.

An embedding layer can be defined as:
```
Embedding(input_dim, embed_size)
```
Our input dimension will be the length of our vocabulary, the size can be whatever you want to set it at, my case I used 300.
Our model will predict the next word based in the input sequence. We could also predict the next two words, or predict entire sentences. For now though we will just stick with one word.



<div class=" highlight hl-ipython3"><pre><span></span><span class="k">class</span> <span class="nc">LSTM_Model</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">vocab_size</span><span class="p">,</span> <span class="n">embed_size</span><span class="p">,</span> <span class="n">lstm_size</span><span class="o">=</span><span class="mi">400</span><span class="p">,</span> <span class="n">num_layers</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">dropout</span><span class="o">=</span><span class="mf">0.3</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">LSTM_Model</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">embedding</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Embedding</span><span class="p">(</span><span class="n">vocab_size</span><span class="p">,</span> <span class="n">embed_size</span><span class="p">)</span>
        <span class="c1"># batch_size is first</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">hidden_dim</span> <span class="o">=</span> <span class="n">lstm_size</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">vocab_size</span> <span class="o">=</span> <span class="n">vocab_size</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">num_layers</span> <span class="o">=</span> <span class="n">num_layers</span>
        
        
        <span class="bp">self</span><span class="o">.</span><span class="n">LSTM</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">LSTM</span><span class="p">(</span><span class="n">input_size</span><span class="o">=</span><span class="n">embed_size</span><span class="p">,</span> <span class="n">hidden_size</span><span class="o">=</span><span class="n">lstm_size</span><span class="p">,</span> <span class="n">num_layers</span><span class="o">=</span><span class="n">num_layers</span><span class="p">,</span> <span class="n">dropout</span><span class="o">=</span><span class="n">dropout</span><span class="p">,</span> <span class="n">batch_first</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">classifier</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">lstm_size</span><span class="p">,</span> <span class="n">vocab_size</span><span class="p">)</span>
        
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">prev_hidden</span><span class="p">):</span>
        <span class="n">batch_size</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
        <span class="n">out</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">embedding</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">out</span><span class="p">,</span> <span class="n">hidden</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">LSTM</span><span class="p">(</span><span class="n">out</span><span class="p">,</span> <span class="n">prev_hidden</span><span class="p">)</span>
        <span class="c1"># the output from the LSTM needs to be flattened for the classifier, so reshape output to: (batch_size * seq_len, hidden_dim)</span>
        <span class="n">out</span> <span class="o">=</span> <span class="n">out</span><span class="o">.</span><span class="n">contiguous</span><span class="p">()</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">hidden_dim</span><span class="p">)</span>
        
        <span class="n">out</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">classifier</span><span class="p">(</span><span class="n">out</span><span class="p">)</span>
        
        <span class="c1"># reshape to split apart the batch_size * seq_len dimension</span>
        <span class="n">out</span> <span class="o">=</span> <span class="n">out</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="n">batch_size</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">vocab_size</span><span class="p">)</span>
        
        <span class="c1"># only need the output of the layer, so remove the middle seq_len dimension</span>
        <span class="n">out</span> <span class="o">=</span> <span class="n">out</span><span class="p">[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">]</span>
        
        <span class="k">return</span> <span class="n">out</span><span class="p">,</span> <span class="n">hidden</span>
    
    <span class="k">def</span> <span class="nf">init_hidden</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
        <span class="n">weight</span> <span class="o">=</span> <span class="nb">next</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">parameters</span><span class="p">())</span><span class="o">.</span><span class="n">data</span>
    
        <span class="n">hidden</span> <span class="o">=</span> <span class="p">(</span><span class="n">weight</span><span class="o">.</span><span class="n">new</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">num_layers</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">hidden_dim</span><span class="p">)</span><span class="o">.</span><span class="n">zero_</span><span class="p">(),</span>
                      <span class="n">weight</span><span class="o">.</span><span class="n">new</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">num_layers</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">hidden_dim</span><span class="p">)</span><span class="o">.</span><span class="n">zero_</span><span class="p">())</span>
 
        
        <span class="k">return</span> <span class="n">hidden</span>
</pre></div>



## Hyperparameters and Compiling the Model
The Adam optimizer is very effective and has built in dynamic reduction of the learning rate, so let's use that. We will also set the learning rate, epochs, and batch size.

We use the CrossEntropyLoss, which requires raw logits as input, since softmax is built into the loss function.

Dropout should be a bit high as we are training on a small amount of data, so our model is prone to overfit quickly.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### BEGIN SOLUTION</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">LSTM_Model</span><span class="p">(</span><span class="n">vocab_size</span><span class="p">,</span> <span class="mi">300</span><span class="p">,</span> <span class="n">lstm_size</span><span class="o">=</span><span class="mi">400</span><span class="p">,</span> <span class="n">num_layers</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">dropout</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>
<span class="c1">### END SOLUTION</span>
<span class="n">device</span> <span class="o">=</span> <span class="s1">&#39;cuda&#39;</span> <span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span> <span class="k">else</span> <span class="s1">&#39;cpu&#39;</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Using device: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">device</span><span class="p">))</span>
<span class="n">model</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

<span class="n">learn_rate</span> <span class="o">=</span> <span class="mf">0.001</span>

<span class="c1"># write out the optimizer and criterion here, using CrossEntropyLoss and the Adam optimizer</span>
<span class="c1">### BEGIN SOLUTION</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="n">learn_rate</span><span class="p">)</span>
<span class="n">criterion</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">CrossEntropyLoss</span><span class="p">()</span>
<span class="c1">### END SOLUTION</span>

<span class="c1"># torch summary has a bug where it won&#39;t work with embedding layers</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>

<span class="k">if</span> <span class="n">device</span> <span class="o">==</span> <span class="s1">&#39;cuda&#39;</span><span class="p">:</span>
    <span class="c1"># helps with runtime of model, use if you have a constant batch size</span>
    <span class="n">cudnn</span><span class="o">.</span><span class="n">benchmark</span> <span class="o">=</span> <span class="kc">True</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># load weights if continuing training</span>
<span class="n">load</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s2">&quot;best.weights.pt&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">load</span><span class="p">[</span><span class="s2">&quot;net&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Loaded model from epoch </span><span class="si">{}</span><span class="s2">.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">load</span><span class="p">[</span><span class="s2">&quot;epoch&quot;</span><span class="p">]))</span>
</pre></div>



## Training
Now it is time to train the model. We will use tensorboardX to graph our loss and we will save the checkpoint everytime our training loss decreases. We do not use validation data because we want the model to be closely related to how our text is constructed.

The Tensorboard is commented out right now, since we are running on kaggle. If you run locally check you can uncomment the tensorbardX lines and view the tensorboard by:
- Installing tensorboard `pip install tensorboard`
- Running tensorboard --logdir=tensorboard_logs in a terminal
- Going to the link it gives you.

Note that the model overfits easily since we don't have much data, so train for a small number of epochs.



<div class=" highlight hl-ipython3"><pre><span></span><span class="n">epochs</span> <span class="o">=</span> <span class="mi">7</span>

<span class="c1"># view tensorboard with command: tensorboard --logdir=tensorboard_logs</span>
<span class="c1"># os.makedirs(&quot;tensorboard_logs&quot;, exist_ok=True)</span>
<span class="c1"># os.makedirs(&quot;checkpoints&quot;, exist_ok=True)</span>

<span class="c1"># ten_board = SummaryWriter(&#39;tensorboard_logs/run_{}&#39;.format(start_time))</span>
<span class="n">weight_save_path</span> <span class="o">=</span> <span class="s1">&#39;best.weights.pt&#39;</span>
<span class="n">print_step</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="p">)</span> <span class="o">//</span> <span class="mi">20</span>
<span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
<span class="n">best_loss</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">start_time</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>

<span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">epochs</span><span class="p">):</span>
    <span class="n">train_loss</span> <span class="o">=</span> <span class="mi">0</span>
    
    <span class="c1"># get inital hidden state</span>
    <span class="n">hidden</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">init_hidden</span><span class="p">(</span><span class="n">batch_size</span><span class="p">)</span>
    <span class="n">hidden</span> <span class="o">=</span> <span class="p">(</span><span class="n">hidden</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">),</span> <span class="n">hidden</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">))</span>
    
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">dataloader</span><span class="p">):</span>
        <span class="c1"># make sure you iterate over completely full batches, only</span>
        <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">&lt;</span> <span class="n">batch_size</span><span class="p">:</span>
            <span class="k">break</span>
            
        <span class="n">inputs</span><span class="p">,</span> <span class="n">targets</span> <span class="o">=</span> <span class="n">data</span>
        <span class="n">inputs</span><span class="p">,</span> <span class="n">targets</span> <span class="o">=</span> <span class="n">inputs</span><span class="o">.</span><span class="n">type</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">LongTensor</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">),</span> <span class="n">targets</span><span class="o">.</span><span class="n">type</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">LongTensor</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
        
        <span class="n">hidden</span> <span class="o">=</span> <span class="nb">tuple</span><span class="p">([</span><span class="n">each</span><span class="o">.</span><span class="n">data</span> <span class="k">for</span> <span class="n">each</span> <span class="ow">in</span> <span class="n">hidden</span><span class="p">])</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
        <span class="n">outputs</span><span class="p">,</span> <span class="n">hidden</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">inputs</span><span class="p">,</span> <span class="n">hidden</span><span class="p">)</span>
        
        <span class="n">loss</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="n">targets</span><span class="p">)</span>
        <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
        <span class="n">train_loss</span> <span class="o">+=</span> <span class="n">loss</span><span class="o">.</span><span class="n">item</span><span class="p">()</span>
        
        
        <span class="k">if</span> <span class="n">i</span> <span class="o">%</span> <span class="n">print_step</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">print_iter</span><span class="p">(</span><span class="n">curr_epoch</span><span class="o">=</span><span class="n">e</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="n">epochs</span><span class="p">,</span> <span class="n">batch_i</span><span class="o">=</span><span class="n">i</span><span class="p">,</span> <span class="n">num_batches</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="p">),</span> <span class="n">loss</span><span class="o">=</span><span class="n">train_loss</span><span class="o">/</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">))</span>
    
    <span class="c1"># print iteration takes the tensorboardX writer and adds the metrics we have to it</span>
    <span class="c1"># print_iter(curr_epoch=e, epochs=epochs, writer=writer, loss=train_loss/len(train_dataloader))</span>
    <span class="n">print_iter</span><span class="p">(</span><span class="n">curr_epoch</span><span class="o">=</span><span class="n">e</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="n">epochs</span><span class="p">,</span> <span class="n">loss</span><span class="o">=</span><span class="n">train_loss</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="p">))</span>
    
    <span class="k">if</span> <span class="n">e</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="n">best_loss</span> <span class="o">=</span> <span class="n">train_loss</span>
    <span class="k">elif</span> <span class="n">train_loss</span> <span class="o">&lt;</span> <span class="n">best_loss</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;</span><span class="se">\n</span><span class="s1">Saving Checkpoint..</span><span class="se">\n</span><span class="s1">&#39;</span><span class="p">)</span>
        <span class="n">state</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;net&#39;</span><span class="p">:</span> <span class="n">model</span><span class="o">.</span><span class="n">state_dict</span><span class="p">(),</span> <span class="s1">&#39;loss&#39;</span><span class="p">:</span> <span class="n">train_loss</span><span class="p">,</span> <span class="s1">&#39;epoch&#39;</span><span class="p">:</span> <span class="n">e</span><span class="p">,</span> <span class="s1">&#39;sequence_length&#39;</span><span class="p">:</span> <span class="n">sequence_length</span><span class="p">,</span> <span class="s1">&#39;batch_size&#39;</span><span class="p">:</span> <span class="n">batch_size</span><span class="p">,</span> <span class="s1">&#39;int_to_word&#39;</span><span class="p">:</span> <span class="n">int_to_word</span><span class="p">,</span> <span class="s1">&#39;word_to_int&#39;</span><span class="p">:</span> <span class="n">word_to_int</span><span class="p">}</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">weight_save_path</span><span class="p">)</span>
        <span class="n">best_loss</span> <span class="o">=</span> <span class="n">train_loss</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Model took </span><span class="si">{:.2f}</span><span class="s2"> minutes to train.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">((</span><span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">start_time</span><span class="p">)</span> <span class="o">/</span> <span class="mi">60</span><span class="p">))</span>
</pre></div>



## Testing the Model
Testing the model simply requires that we convert the output integer back into a word and build our generated text, starting from a seed we define. However, we might get better results by instead of doing an argmax to find the highest probability of what the next word should be, we can take a sample of the top possible words and choose one from there. 

This is done by taking a "temperature" which defines how many predictions we will consider as the next possible word. A lower temperature means the word picked will be closer to the word with the highest probability. Then using a random selection to choose a word. Try it with using both. Setting a temperature of 0 will just use argmax on the entire prediction.



<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#load model if returning to this notebook for testing, model that I trained:</span>
<span class="n">load</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">weight_save_path</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">load</span><span class="p">[</span><span class="s2">&quot;net&quot;</span><span class="p">])</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>

<span class="k">def</span> <span class="nf">sample</span><span class="p">(</span><span class="n">prediction</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="mi">0</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">temp</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span>
    <span class="n">prediction</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;float64&#39;</span><span class="p">)</span>
    <span class="n">prediction</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span> <span class="o">/</span> <span class="n">temp</span>
    <span class="n">expo_prediction</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span>
    <span class="n">prediction</span> <span class="o">=</span> <span class="n">expo_prediction</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">expo_prediction</span><span class="p">)</span>
    <span class="n">probabilities</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">multinomial</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">prediction</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">probabilities</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">pad_sequences</span><span class="p">(</span><span class="n">sequence</span><span class="p">,</span> <span class="n">maxlen</span><span class="p">,</span> <span class="n">value</span><span class="o">=</span><span class="mi">0</span><span class="p">):</span>
    <span class="k">while</span> <span class="nb">len</span><span class="p">(</span><span class="n">sequence</span><span class="p">)</span> <span class="o">&lt;</span> <span class="n">maxlen</span><span class="p">:</span>
        <span class="n">sequence</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="n">sequence</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">value</span><span class="p">)</span>
    <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">sequence</span><span class="p">)</span> <span class="o">&gt;</span> <span class="n">maxlen</span><span class="p">:</span>
        <span class="n">sequence</span> <span class="o">=</span> <span class="n">sequence</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">sequence</span><span class="p">)</span> <span class="o">-</span> <span class="n">maxlen</span><span class="p">:]</span>
    <span class="k">return</span> <span class="n">sequence</span>

<span class="c1">#generate new script</span>
<span class="k">def</span> <span class="nf">generate_text</span><span class="p">(</span><span class="n">seed_text</span><span class="p">,</span> <span class="n">num_words</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="mi">0</span><span class="p">):</span>
    <span class="n">input_text</span><span class="o">=</span> <span class="n">seed_text</span>
    <span class="k">for</span> <span class="n">_</span>  <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">num_words</span><span class="p">):</span>
        <span class="c1">#tokenize text to ints</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">tokenize_punctuation</span><span class="p">(</span><span class="n">input_text</span><span class="p">)</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">int_text</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">int_text</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">word_to_int</span><span class="p">[</span><span class="n">word</span><span class="p">]</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">int_text</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
        <span class="c1">#pad text if it is too short, pads with zeros at beginning of text, so shouldnt have too much noise added</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">pad_sequences</span><span class="p">(</span><span class="n">int_text</span><span class="p">,</span> <span class="n">maxlen</span><span class="o">=</span><span class="n">sequence_length</span><span class="p">)</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">expand_dims</span><span class="p">(</span><span class="n">int_text</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
        <span class="c1"># init hiddens state</span>
        <span class="n">hidden</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">init_hidden</span><span class="p">(</span><span class="n">int_text</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
        <span class="n">hidden</span> <span class="o">=</span> <span class="p">(</span><span class="n">hidden</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">),</span> <span class="n">hidden</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">))</span>
        <span class="c1">#predict next word:</span>
        <span class="n">prediction</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">from_numpy</span><span class="p">(</span><span class="n">int_text</span><span class="p">)</span><span class="o">.</span><span class="n">type</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">LongTensor</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">),</span> <span class="n">hidden</span><span class="p">)</span>
        <span class="n">prediction</span> <span class="o">=</span> <span class="n">prediction</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="s2">&quot;cpu&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">detach</span><span class="p">()</span>
        <span class="n">prediction</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">prediction</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">data</span>
        <span class="n">prediction</span> <span class="o">=</span> <span class="n">prediction</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>
        <span class="n">output_word</span> <span class="o">=</span> <span class="n">int_to_word</span><span class="p">[</span><span class="n">sample</span><span class="p">(</span><span class="n">prediction</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="n">temp</span><span class="p">)]</span>
        <span class="c1">#append to the result</span>
        <span class="n">input_text</span> <span class="o">+=</span> <span class="s1">&#39; &#39;</span> <span class="o">+</span> <span class="n">output_word</span>
    <span class="c1">#convert tokenized punctuation and other characters back</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">untokenize_punctuation</span><span class="p">(</span><span class="n">input_text</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">result</span>
</pre></div>





<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#input amount of words to generate, and the seed text, good options are &#39;Homer_Simpson:&#39;, &#39;Bart_Simpson:&#39;, &#39;Moe_Szyslak:&#39;, or other character&#39;s names.:</span>
<span class="n">seed</span> <span class="o">=</span> <span class="s1">&#39;Homer_Simpson:&#39;</span>
<span class="n">num_words</span> <span class="o">=</span> <span class="mi">200</span>
<span class="n">temp</span> <span class="o">=</span> <span class="mf">0.5</span>

<span class="c1"># print amount of characters specified.</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Starting seed is: </span><span class="si">{}</span><span class="se">\n\n</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">seed</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">seed</span><span class="p">,</span> <span class="n">num_words</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="n">temp</span><span class="p">))</span>
</pre></div>



## Closing Thoughts
Remember that this model can be applied to any type of text, even code! So go and try different texts, like the (not) included Harry Potter book. (for time purposes, I would not use the whole book, as training would take a long time.)

Try different hyperparameters and model sizes, as you can get some better results.
