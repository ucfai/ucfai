---
title: "Who Needs Show Writers Nowadays?"
categories: ["sp19"]
authors: ['brandons209', 'ionlights']
description: >-
  "This lecture is all about Recurrent Neural Networks. These are networks with with added memory, which means they can learn from sequential data such as speech, text, videos, and more. Different types of RNNs and strategies for building  them will also be covered. The project will be building a LSTM-RNN to generate new original scripts for the TV series “The Simpsons”. Come and find out if our networks can become better writers for the show!"
---

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Generate-new-Simpson-scripts-with-LSTM-RNN">Generate new Simpson scripts with LSTM RNN<a class="anchor-link" href="#Generate-new-Simpson-scripts-with-LSTM-RNN">&#182;</a></h1><h2 id="Link-to-slides-here">Link to slides <a href="https://docs.google.com/presentation/d/1kzI_rRH1sZLUxWWxYpVLzXu1MKLR9ljZ2By6zYHjj5w/edit?usp=sharing">here</a><a class="anchor-link" href="#Link-to-slides-here">&#182;</a></h2><p>In this project, we will be using an LSTM with the help of an Embedding layer to train our network on an episode from the Simpsons, specifically the episode "Moe's Tavern". This is taken from <a href="https://www.kaggle.com/wcukierski/the-simpsons-by-the-data">this</a> dataset on kaggle. This model can be applied to any text, so experiment afterwards! We could use more episodes from the Simpsons, a book, articles, wikipedia, etc. It will learn the word embeddings and being able to generate text in relation to what it is trained on.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>First, lets import all of our libraries we need. We utilize Keras' Tokenizer method for tokenizing our inputs, and pad_sequences for generating our sequences. Our embedding layer has a fixed input size, so instead of passing our entire script at once we supply a sequence of characters with a length we can choose. Documentation can be found for <a href="https://keras.io/preprocessing/text/">tokenizer</a> and <a href="https://keras.io/preprocessing/sequence/">pad_sequences</a>.</p>
<p>I have also created a helper function to handle some loading and saving of dictionaries we will make, and tokenizing punctuation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#general imports</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">import</span> <span class="nn">os</span>

<span class="c1">#pre-processing</span>
<span class="kn">from</span> <span class="nn">keras.preprocessing.text</span> <span class="k">import</span> <span class="n">Tokenizer</span>
<span class="kn">from</span> <span class="nn">keras.preprocessing.sequence</span> <span class="k">import</span> <span class="n">pad_sequences</span>

<span class="c1">#model</span>
<span class="kn">from</span> <span class="nn">keras.models</span> <span class="k">import</span> <span class="n">Sequential</span>
<span class="kn">from</span> <span class="nn">keras.layers</span> <span class="k">import</span> <span class="n">Dropout</span><span class="p">,</span> <span class="n">Dense</span>
<span class="kn">from</span> <span class="nn">keras.layers</span> <span class="k">import</span> <span class="n">Embedding</span><span class="p">,</span> <span class="n">LSTM</span>
<span class="kn">from</span> <span class="nn">keras.models</span> <span class="k">import</span> <span class="n">load_model</span>

<span class="c1">#training</span>
<span class="kn">from</span> <span class="nn">keras.callbacks</span> <span class="k">import</span> <span class="n">ModelCheckpoint</span><span class="p">,</span> <span class="n">TensorBoard</span>
<span class="kn">from</span> <span class="nn">keras</span> <span class="k">import</span> <span class="n">optimizers</span> <span class="k">as</span> <span class="n">opt</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Grab-important-files-and-create-folders">Grab important files and create folders<a class="anchor-link" href="#Grab-important-files-and-create-folders">&#182;</a></h2><p>Here we download the dataset and save it to the folder "data" that we create. There is also a python script file with helper functions I have written to help with processing the data.</p>
<p>To do this, we download a script called gdown that allows downloading files from Google Drive from the command line.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">!</span>wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl
<span class="o">!</span>chmod +x gdown.pl
<span class="o">!</span>mkdir data
<span class="o">!</span>mkdir data/dictionaries
<span class="o">!</span>mkdir saved_model_data
<span class="o">!</span>mkdir tensorboard_logs

<span class="o">!</span>./gdown.pl https://drive.google.com/file/d/1dgOnAVDDTDAg59SYcuu-aBzva7gk5YFk/view helper.py
<span class="o">!</span>./gdown.pl https://drive.google.com/file/d/1F8Jd_0fhlT5kCwd0m5kUDS2KyiVkOSXy/view data/moes_tavern_lines.txt

<span class="kn">import</span> <span class="nn">helper</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Dataset-statistics">Dataset statistics<a class="anchor-link" href="#Dataset-statistics">&#182;</a></h2><p>Before starting our project, we should take a look at the data we are dealing with. Pay attention to the average number of words per line.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">script_text</span> <span class="o">=</span> <span class="n">helper</span><span class="o">.</span><span class="n">load_script</span><span class="p">(</span><span class="s1">&#39;data/moes_tavern_lines.txt&#39;</span><span class="p">)</span>

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

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Tokenize-Text">Tokenize Text<a class="anchor-link" href="#Tokenize-Text">&#182;</a></h2><p>In order to prepare our data for our network, we need to tokenize the words. That is, we will be converting every unique word and punctuation into an integer. Before we do that, we need to make the punctuation more easier to convert to a number. For example, we will be taking any new lines and converting them to the word "||return||". This makes the text easier to tokenize and pass into our model. The functions that do this are in the helper.py file.</p>
<p>A note on Keras' tokenizer function. 0 is a reserved integer, that is not used to represent any words. So our integers for our words will start at 1. This is needed as when we use the model to generate new text, it needs a starting point, known as a seed. If this seed is smaller than our sequence length, then the function pad_sequences will pad that seed with 0's in order to represent "nothing". This help reduces noise in the network.</p>
<p>This is the list of punctuation and special characters that are converted, notice that spaces are put before and after to make splitting the text easier:</p>
<ul>
<li>'!' : ' ||exclaimation_mark|| '</li>
<li>',' : ' ||comma|| '</li>
<li>'"' : ' ||quotation_mark|| '</li>
<li>';' : ' ||semicolon|| '</li>
<li>'.' : ' ||period|| '</li>
<li>'?' : ' ||question_mark|| '</li>
<li>'(' : ' ||left_parentheses|| '</li>
<li>')' : ' ||right_parentheses|| '</li>
<li>'--' : ' ||dash|| '</li>
<li>'\n' : ' ||return|| '</li>
<li>':' : ' ||colon|| '</li>
</ul>
<p>We also convert all of the text to lowercase as this reduces the vocabulary list and trains the network faster.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">tokens</span> <span class="o">=</span> <span class="n">Tokenizer</span><span class="p">(</span><span class="n">filters</span><span class="o">=</span><span class="s1">&#39;&#39;</span><span class="p">,</span> <span class="n">lower</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">char_level</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="c1">#keras tokenizer function, char_level is for setting the tokenizer to treat every character as a token, instead of words</span>
<span class="n">script_text</span> <span class="o">=</span> <span class="n">helper</span><span class="o">.</span><span class="n">tokenize_punctuation</span><span class="p">(</span><span class="n">script_text</span><span class="p">)</span> <span class="c1">#helper function to convert non-word characters</span>
<span class="n">script_text</span> <span class="o">=</span> <span class="n">script_text</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>

<span class="n">script_text</span> <span class="o">=</span> <span class="n">script_text</span><span class="o">.</span><span class="n">split</span><span class="p">()</span><span class="c1">#splits the text based on spaces into a list</span>

<span class="n">tokens</span><span class="o">.</span><span class="n">fit_on_texts</span><span class="p">(</span><span class="n">script_text</span><span class="p">)</span><span class="c1">#this will apply the tokenizer to the text.</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Creating-Conversion-Dictionaries-and-Input-Data">Creating Conversion Dictionaries and Input Data<a class="anchor-link" href="#Creating-Conversion-Dictionaries-and-Input-Data">&#182;</a></h2><p>Now that the tokens have been generated, we will create some dictionaries to convert our tokenized integers back to words, and words to integers. We will also generate our inputs and targets to pass into our model.</p>
<p>To do this, we need to specify the sequence length, which is the amount of words we pass into the model at one time. I choose 12 as it is the average length of a line, therefore generally the input to the model will be an entire line. Feel free to experiment with this parameter. A sequence length of 1 is just one word, so we could get better output depending on our sequence length. We use the helper function gen_sequences to do this for us. Then we can save these for testing.</p>
<p>The targets are simply just the next word in our text. So if we have a sentence: "Hi, how are you?" and we input "Hi, how are you" our target for this sentence will be "?".</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sequence_length</span> <span class="o">=</span> <span class="mi">12</span>

<span class="n">word_to_int</span> <span class="o">=</span> <span class="n">tokens</span><span class="o">.</span><span class="n">word_index</span> <span class="c1">#grab word : int dict</span>
<span class="n">int_to_word</span> <span class="o">=</span> <span class="p">{</span><span class="nb">int</span><span class="p">:</span> <span class="n">word</span> <span class="k">for</span> <span class="n">word</span><span class="p">,</span> <span class="nb">int</span> <span class="ow">in</span> <span class="n">word_to_int</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span> <span class="c1">#flip word_to_int dict to get int to word</span>
<span class="n">int_script_text</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">word_to_int</span><span class="p">[</span><span class="n">word</span><span class="p">]</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">script_text</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span> <span class="c1">#convert text to integers</span>
<span class="n">int_script_text</span><span class="p">,</span> <span class="n">targets</span> <span class="o">=</span> <span class="n">helper</span><span class="o">.</span><span class="n">gen_sequences</span><span class="p">(</span><span class="n">int_script_text</span><span class="p">,</span> <span class="n">sequence_length</span><span class="p">)</span> <span class="c1">#transform int_script_text to sequences of sequence_length and generate targets</span>
<span class="n">vocab_length</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">word_to_int</span><span class="p">)</span> <span class="o">+</span> <span class="mi">1</span> <span class="c1">#vocab_length for embedding needs to 1 one to length of the dictionary.</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of vocabulary: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">vocab_length</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#save dictionaries for use with testing model, also need to save sequence length since it needs to be the same when testing</span>
<span class="n">helper</span><span class="o">.</span><span class="n">save_dict</span><span class="p">(</span><span class="n">word_to_int</span><span class="p">,</span> <span class="s1">&#39;word_to_int.pkl&#39;</span><span class="p">)</span>
<span class="n">helper</span><span class="o">.</span><span class="n">save_dict</span><span class="p">(</span><span class="n">int_to_word</span><span class="p">,</span> <span class="s1">&#39;int_to_word.pkl&#39;</span><span class="p">)</span>
<span class="n">helper</span><span class="o">.</span><span class="n">save_dict</span><span class="p">(</span><span class="n">sequence_length</span><span class="p">,</span> <span class="s1">&#39;sequence_length.pkl&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Building-the-Model">Building the Model<a class="anchor-link" href="#Building-the-Model">&#182;</a></h2><p>Here is the fun part, building our model. We will use LSTM cells and an Embedding layer, with a fully connected Dense layer at the end for the prediction. Documentation for LSTM cells can be found <a href="https://keras.io/layers/recurrent/">here</a> and for embedding <a href="https://keras.io/layers/embeddings/">here</a>.</p>
<p>An LSTM layer can be defined simply as:</p>

<pre><code>LSTM(num_cells, dropout=drop, recurrent_dropout=drop, return_sequences=True)</code></pre>
<p>From the docs:</p>
<ul>
<li>dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</li>
<li>recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</li>
<li>return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.</li>
</ul>
<p>For LSTM layers up until the last LSTM layer, return_sequences is set to True to tell the layer to output the full sequence with its predictions, instead of just the predictions, which allows the next LSTM layer to learn from the input text, and what the LSTM layer added to it before hand. The last layer will leave this unset since we want it to return the last output in the sequence as that will be our final output for the Dense layer, using softmax activation.</p>
<p>An embedding layer can be defined as:</p>

<pre><code>Embedding(input_dim, size, input_length=)</code></pre>
<p>Our input dimension will be the length of our vocabulary, the size can be whatever you want to set it at, my case I used 300, and the input_length is our sequence_length we defined earlier.</p>
<p>Our model will predict the next word based in the input sequence. We could also predict the next two words, or predict entire sentences. For now though we will just stick with one word.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">Sequential</span><span class="p">()</span>
<span class="c1">### Put Model Below: ###</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Hyperparameters-and-Compiling-the-Model">Hyperparameters and Compiling the Model<a class="anchor-link" href="#Hyperparameters-and-Compiling-the-Model">&#182;</a></h2><p>The Adam optimizer is very effective and has built in dynamic reduction of the learning rate, so let's use that. We will also set the learning rate, epochs, and batch size.</p>
<p>You may assume our loss function will be categorical_crossentropy. In our case, this will not work, as that loss function requires our labels/targets to be one-hot encoded. Keras provides another loss function, called  sparse_categorical_crossentropy. This applies categorical_crossentropy, but uses labels that are not one-hot encoded. Since our labels will just be numbers from 1 to vocab_length, this works well for us.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">learn_rate</span> <span class="o">=</span> <span class="mf">0.001</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">opt</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">lr</span><span class="o">=</span><span class="n">learn_rate</span><span class="p">)</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">32</span>

<span class="n">model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">optimizer</span><span class="o">=</span><span class="n">optimizer</span><span class="p">,</span> <span class="n">loss</span><span class="o">=</span><span class="s1">&#39;sparse_categorical_crossentropy&#39;</span><span class="p">,</span> <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;accuracy&#39;</span><span class="p">])</span>
<span class="n">model</span><span class="o">.</span><span class="n">summary</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Training">Training<a class="anchor-link" href="#Training">&#182;</a></h2><p>Now it is time to train the model. We will use the ModelCheckpoint and tensorboard callbacks for saving the best weights and allowing us to view graphs of loss and accuracy of our model as it is training. Since we are not using validation data, our monitor for our ModelCheckpoint callback will be loss. We do not use validation data because we want the model to be closely related to how our text is constructed.</p>
<p>The model is then saved after training.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#load weights if continuing training</span>
<span class="n">model</span><span class="o">.</span><span class="n">load_weights</span><span class="p">(</span><span class="s2">&quot;saved_model_data/model.best.weights.hdf5&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">start_time</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">strftime</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">%a</span><span class="s2">_%b_</span><span class="si">%d</span><span class="s2">_%Y_%H:%M&quot;</span><span class="p">,</span> <span class="n">time</span><span class="o">.</span><span class="n">localtime</span><span class="p">())</span>
<span class="c1">#view tensorboard with command: tensorboard --logdir=tensorboard_logs</span>
<span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="s2">&quot;tensorboard_logs&quot;</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">ten_board</span> <span class="o">=</span> <span class="n">TensorBoard</span><span class="p">(</span><span class="n">log_dir</span><span class="o">=</span><span class="s1">&#39;tensorboard_logs/</span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">start_time</span><span class="p">),</span> <span class="n">write_images</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">weight_save_path</span> <span class="o">=</span> <span class="s1">&#39;saved_model_data/</span><span class="si">{}</span><span class="s1">.best.weights.hdf5&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">start_time</span><span class="p">)</span>
<span class="n">checkpointer</span> <span class="o">=</span> <span class="n">ModelCheckpoint</span><span class="p">(</span><span class="n">weight_save_path</span><span class="p">,</span> <span class="n">monitor</span><span class="o">=</span><span class="s1">&#39;loss&#39;</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">save_best_only</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">save_weights_only</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Tensorboard logs for this run are in: </span><span class="si">{}</span><span class="s2">, weights will be saved in </span><span class="si">{}</span><span class="se">\n</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s1">&#39;tensorboard_logs/</span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">start_time</span><span class="p">),</span> <span class="n">weight_save_path</span><span class="p">))</span>

<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">int_script_text</span><span class="p">,</span> <span class="n">targets</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="n">epochs</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">callbacks</span><span class="o">=</span><span class="p">[</span><span class="n">checkpointer</span><span class="p">,</span> <span class="n">ten_board</span><span class="p">])</span>
<span class="n">model</span><span class="o">.</span><span class="n">load_weights</span><span class="p">(</span><span class="n">weight_save_path</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="s1">&#39;saved_model_data/model.</span><span class="si">{}</span><span class="s1">.h5&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">start_time</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Testing-the-Model">Testing the Model<a class="anchor-link" href="#Testing-the-Model">&#182;</a></h2><p>Testing the model simply requires that we convert the output integer back into a word and build our generated text, based on the argmax to find the highest probability of what the next word should be, starting from a seed we define. However, we might get better results by taking a sample of the top possible words and choose one at random from there.</p>
<p>This is done by taking a "temperature" which defines how many predictions we will consider as the next possible word. A lower temperature means the word picked will be closer to the word with the highest probability. Then using a random selection to choose a word. Try it with using both. Setting a temperature of 0 will just use argmax on the entire prediction.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#download weights, model, and dictionaries if using my model:</span>
<span class="o">!</span>./gdown.pl https://drive.google.com/file/d/1UPUkyo5D2Q-WK1d1NHUEA4lU3AC1fFAM/view data/dictionaries/int_to_word.pkl
<span class="o">!</span>./gdown.pl https://drive.google.com/file/d/1UPUkyo5D2Q-WK1d1NHUEA4lU3AC1fFAM/view data/dictionaries/sequence_length.pkl
<span class="o">!</span>./gdown.pl https://drive.google.com/file/d/1fCwa1KnaAMJTriM4hmIpUAbwKF3rkKU1/view data/dictionaries/word_to_int.pkl

<span class="o">!</span>./gdown.pl https://drive.google.com/file/d/1v5XzYZ3X3xKlJUl-EyRolq-FbwFwvw3n/view saved_model_data/model.best.weights.hdf5
<span class="o">!</span>./gdown.pl https://drive.google.com/file/d/1IJnQA4vKPAesZlF6Sc0hDC_DXa7WKdcY/view saved_model_data/model.Tue_Jul_24_2018_01<span class="se">\:</span><span class="m">22</span>.h5
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#load model if returning to this notebook for testing, model that I trained:</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">load_model</span><span class="p">(</span><span class="s1">&#39;saved_model_data/model.Tue_Jul_24_2018_01:22.h5&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">sample</span><span class="p">(</span><span class="n">prediction</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="mi">0</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">temp</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span>
    <span class="n">prediction</span> <span class="o">=</span> <span class="n">prediction</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">prediction</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;float64&#39;</span><span class="p">)</span>
    <span class="n">prediction</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span> <span class="o">/</span> <span class="n">temp</span>
    <span class="n">expo_prediction</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span>
    <span class="n">prediction</span> <span class="o">=</span> <span class="n">expo_prediction</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">expo_prediction</span><span class="p">)</span>
    <span class="n">probabilities</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">multinomial</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">prediction</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">probabilities</span><span class="p">)</span>

<span class="c1">#generate new script</span>
<span class="k">def</span> <span class="nf">generate_text</span><span class="p">(</span><span class="n">seed_text</span><span class="p">,</span> <span class="n">num_words</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="mi">0</span><span class="p">):</span>
    <span class="n">input_text</span><span class="o">=</span> <span class="n">seed_text</span>
    <span class="k">for</span> <span class="n">_</span>  <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">num_words</span><span class="p">):</span>
        <span class="c1">#tokenize text to ints</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">helper</span><span class="o">.</span><span class="n">tokenize_punctuation</span><span class="p">(</span><span class="n">input_text</span><span class="p">)</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">int_text</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">int_text</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">word_to_int</span><span class="p">[</span><span class="n">word</span><span class="p">]</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">int_text</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
        <span class="c1">#pad text if it is too short, pads with zeros at beginning of text, so shouldnt have too much noise added</span>
        <span class="n">int_text</span> <span class="o">=</span> <span class="n">pad_sequences</span><span class="p">([</span><span class="n">int_text</span><span class="p">],</span> <span class="n">maxlen</span><span class="o">=</span><span class="n">sequence_length</span><span class="p">)</span>
        <span class="c1">#predict next word:</span>
        <span class="n">prediction</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">int_text</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
        <span class="n">output_word</span> <span class="o">=</span> <span class="n">int_to_word</span><span class="p">[</span><span class="n">sample</span><span class="p">(</span><span class="n">prediction</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="n">temp</span><span class="p">)]</span>
        <span class="c1">#append to the result</span>
        <span class="n">input_text</span> <span class="o">+=</span> <span class="s1">&#39; &#39;</span> <span class="o">+</span> <span class="n">output_word</span>
    <span class="c1">#convert tokenized punctuation and other characters back</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">helper</span><span class="o">.</span><span class="n">untokenize_punctuation</span><span class="p">(</span><span class="n">input_text</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">result</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[0]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#input amount of words to generate, and the seed text, good options are &#39;Homer_Simpson:&#39;, &#39;Bart_Simpson:&#39;, &#39;Moe_Szyslak:&#39;, or other character&#39;s names.:</span>
<span class="n">seed</span> <span class="o">=</span> <span class="s1">&#39;Homer_Simpson:&#39;</span>
<span class="n">num_words</span> <span class="o">=</span> <span class="mi">200</span>
<span class="n">temp</span> <span class="o">=</span> <span class="mf">0.2</span>

<span class="c1">#print amount of characters specified.</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Starting seed is: </span><span class="si">{}</span><span class="se">\n\n</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">seed</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">seed</span><span class="p">,</span> <span class="n">num_words</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="n">temp</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Closing-Thoughts">Closing Thoughts<a class="anchor-link" href="#Closing-Thoughts">&#182;</a></h2><p>Remember that this model can be applied to any type of text, even code! So go and try different texts, like Harry Potter and the Goblet of Fire (for time purposes, I would not use the whole book, as training would take a long time, try a subset of the book.)</p>
<p>Try different hyperparameters and model sizes, as you can get some better results. I was able to get better results by adding a third LSTM layer, but training time and model size increased significantly.</p>

</div>
</div>
</div>
 

