---
title: "A Walk Through the Random Forest"
categories: ["sp19"]
authors: ['ionlights']
description: >-
  "Neural Nets are not the end all be all of Machine Learning. In this lecture,  we will see how a decision tree works, and see how powerful a collection of  them can be. From there, we will see how to utilize Random Forests to do digit  recognition."
---

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Overview">Overview<a class="anchor-link" href="#Overview">&#182;</a></h2><p>Before getting going on more complex examples, we're going to take a look at a very simple example using the Iris Dataset.</p>
<p>After that is done, we're going to move onto using a hybrid model made out of an Autoencoder and a Random Forest to classify hand drawn digits from the MNIST dataset.</p>
<p>The final example deals with credit card fraud, and how to identify if fraud is taking place based a dataset of over 280,000 entries.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Importing the important stuff</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">matthews_corrcoef</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Iris-Data-Set">Iris Data Set<a class="anchor-link" href="#Iris-Data-Set">&#182;</a></h2><p>This is a classic dataset of flowers. The goal is to have the model classify the types of flowers based on 4 factors. Those factors are sepal length, sepal width, petal length, and petal width, which are all measured in cm. The dataset is very old in comparison to many of the datasets we use, coming from a <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x">1936 paper about taxonomy</a>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Getting-the-Data-Set">Getting the Data Set<a class="anchor-link" href="#Getting-the-Data-Set">&#182;</a></h3><p>Sklearn has the dataset built into the the library, so getting the data will be easy. Once we do that, we'll do a test-train split.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">load_iris</span>
<span class="n">iris</span> <span class="o">=</span> <span class="n">load_iris</span><span class="p">()</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">,</span> <span class="n">Y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Making-the-Model">Making the Model<a class="anchor-link" href="#Making-the-Model">&#182;</a></h3><p>Making and Random Forests model is very easy, taking just two lines of code! Training times can take a second, but in this example is small so the training time is minimal.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">trees</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">150</span><span class="p">)</span>
<span class="n">trees</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="We-need-to-Figure-out-how-well-this-model-does">We need to Figure out how well this model does<a class="anchor-link" href="#We-need-to-Figure-out-how-well-this-model-does">&#182;</a></h3><p>There are a few ways we are going to test for accuracy using a Confusion Matrix and Mathews Correlation Coefficient .</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Confusion-Matrix">Confusion Matrix<a class="anchor-link" href="#Confusion-Matrix">&#182;</a></h4><p>A Confusion Matrix shows us where the model is messing up. Below is an example from dataschool.io. The benefit of a confusion matrix is that is it a very easy was to visualise the performance of the model.</p>
<p><img src="https://www.dataschool.io/content/images/2015/01/confusion_matrix_simple2.png" alt="alt text"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">predictions</span> <span class="o">=</span> <span class="n">trees</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">Y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Matthews-correlation-coefficient">Matthews correlation coefficient<a class="anchor-link" href="#Matthews-correlation-coefficient">&#182;</a></h4><p>This is used to find the quality of binary classification. It is based on the values found in the Confusion Matrix and tries to take those values and boil it down to one number. It is generally considered one of the better measures of qaulity for classification. MCC does not realy on class size, so in cases were we have very different class sizes we can get a realiable measure of how well it did.</p>
<hr>
<p>Matthews correlation coefficient ranges from -1 to 1. -1 represents total disagreement between the prediction and the observation, while 1 represnets prefect prediction. In other worlds, the closer to 1 we get the better the model is considered.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">matthews_corrcoef</span><span class="p">(</span><span class="n">Y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>That wasn't too bad wasn't it? Random Forests is a very easy model to implement and it's low training times means that the model can be used without the overheads associated with neural networks. It is also a very flexible model, but from some types of data it will require a little more wizardry.</p>
<h2 id="Image-Classification-on-MNIST">Image Classification on MNIST<a class="anchor-link" href="#Image-Classification-on-MNIST">&#182;</a></h2><p>Random forests by itself does not work well on image data. An image in a computer's world is just a array with values representing intensity and colour. By itself, those values do not lend themselves nicely to decesion trees, but if there were values that represent as features then it could be better for a Random Forest to train on it.</p>
<p>To get those features out of an image, a dimension reduction technique should be applied to extract those features out of an image. To do this, we are going to be using an Autoencorder to find the important features from the data so that our Random Forests model can run better.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Making-the-Autoencoder">Making the Autoencoder<a class="anchor-link" href="#Making-the-Autoencoder">&#182;</a></h3><p>Making a Autoencoder is just like making any other type of Neural Network in Keras. This example is very simple, which was done to save on training time more then anything. More complex Autoencoders can be used in this case and would probably give better results then this simple one that we are using today.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">keras.layers</span> <span class="k">import</span> <span class="n">Input</span><span class="p">,</span> <span class="n">Dense</span>
<span class="kn">from</span> <span class="nn">keras.models</span> <span class="k">import</span> <span class="n">Model</span>

<span class="c1"># size of our original image</span>
<span class="n">img</span> <span class="o">=</span> <span class="n">Input</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">784</span><span class="p">,))</span>

<span class="c1"># how small the compressed image is going to be</span>
<span class="n">compression</span> <span class="o">=</span> <span class="mi">42</span>

<span class="c1"># making the autoencoder</span>
<span class="n">encoded</span> <span class="o">=</span> <span class="n">Dense</span><span class="p">(</span><span class="n">compression</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">&#39;relu&#39;</span><span class="p">)(</span><span class="n">img</span><span class="p">)</span>
<span class="n">decoded</span> <span class="o">=</span> <span class="n">Dense</span><span class="p">(</span><span class="mi">784</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">&#39;sigmoid&#39;</span><span class="p">)(</span><span class="n">encoded</span><span class="p">)</span>
<span class="n">autoencoder</span> <span class="o">=</span> <span class="n">Model</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="n">decoded</span><span class="p">)</span>
<span class="n">encoder</span> <span class="o">=</span> <span class="n">Model</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="n">encoded</span><span class="p">)</span>

<span class="c1"># create a placeholder for an encodeder_input</span>
<span class="n">encoded_input</span> <span class="o">=</span> <span class="n">Input</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">compression</span><span class="p">,))</span>

<span class="c1"># retrieve the last layer of the autoencoder model</span>
<span class="n">decoder_layer</span> <span class="o">=</span> <span class="n">autoencoder</span><span class="o">.</span><span class="n">layers</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>

<span class="c1"># create the decoder model</span>
<span class="n">decoder</span> <span class="o">=</span> <span class="n">Model</span><span class="p">(</span><span class="n">encoded_input</span><span class="p">,</span> <span class="n">decoder_layer</span><span class="p">(</span><span class="n">encoded_input</span><span class="p">))</span>

<span class="c1"># compile the model</span>
<span class="n">autoencoder</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">optimizer</span><span class="o">=</span><span class="s1">&#39;adadelta&#39;</span><span class="p">,</span> <span class="n">loss</span><span class="o">=</span><span class="s1">&#39;binary_crossentropy&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Getting-the-MNIST-data">Getting the MNIST data<a class="anchor-link" href="#Getting-the-MNIST-data">&#182;</a></h3><p>Keras has the dataset already, so importing it will be a breaze. Like with the last example, we're going to be doing a test-train split.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">keras.datasets</span> <span class="k">import</span> <span class="n">mnist</span>
<span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">),</span> <span class="p">(</span><span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span> <span class="o">=</span> <span class="n">mnist</span><span class="o">.</span><span class="n">load_data</span><span class="p">()</span>

<span class="c1"># converting to floats </span>
<span class="n">x_train</span> <span class="o">=</span> <span class="n">x_train</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;float32&#39;</span><span class="p">)</span> <span class="o">/</span> <span class="mf">255.</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">x_test</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;float32&#39;</span><span class="p">)</span> <span class="o">/</span> <span class="mf">255.</span>
<span class="n">x_train</span> <span class="o">=</span> <span class="n">x_train</span><span class="o">.</span><span class="n">reshape</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">x_train</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">prod</span><span class="p">(</span><span class="n">x_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">:])))</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">x_test</span><span class="o">.</span><span class="n">reshape</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">x_test</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">prod</span><span class="p">(</span><span class="n">x_test</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">:])))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Training-the-Autoencoder">Training the Autoencoder<a class="anchor-link" href="#Training-the-Autoencoder">&#182;</a></h3><p>Training is going to take a moment, but due to the model being realitively simple it shouldn't take <em>too long</em>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">autoencoder</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">x_train</span><span class="p">,</span>
                <span class="n">epochs</span><span class="o">=</span><span class="mi">45</span><span class="p">,</span>
                <span class="n">batch_size</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>
                <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                <span class="n">validation_data</span><span class="o">=</span><span class="p">(</span><span class="n">x_test</span><span class="p">,</span> <span class="n">x_test</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Autoencoder-results">Autoencoder results<a class="anchor-link" href="#Autoencoder-results">&#182;</a></h3><p>To see if the Autoencoder did a good job in compressing the images, we're going to be comparing the original images to ones that have been passed through the Autoencoder.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># run on the testing set</span>
<span class="n">encoded_imgs</span> <span class="o">=</span> <span class="n">encoder</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="n">decoded_imgs</span> <span class="o">=</span> <span class="n">decoder</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">encoded_imgs</span><span class="p">)</span>

<span class="c1"># how many digits we will display</span>
<span class="n">n</span> <span class="o">=</span> <span class="mi">7</span>  

<span class="c1"># this prints the images below</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">14</span><span class="p">,</span> <span class="mi">4</span><span class="p">))</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
    <span class="c1"># display original</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplot</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">n</span><span class="p">,</span> <span class="n">i</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">x_test</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span> <span class="mi">28</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">gray</span><span class="p">()</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">get_xaxis</span><span class="p">()</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">get_yaxis</span><span class="p">()</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>

    <span class="c1"># display reconstruction</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplot</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">n</span><span class="p">,</span> <span class="n">i</span> <span class="o">+</span> <span class="mi">1</span> <span class="o">+</span> <span class="n">n</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">decoded_imgs</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span> <span class="mi">28</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">gray</span><span class="p">()</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">get_xaxis</span><span class="p">()</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">get_yaxis</span><span class="p">()</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Random-Forest-Training">Random Forest Training<a class="anchor-link" href="#Random-Forest-Training">&#182;</a></h3><p>We're going to train this model just like we did with the Iris dataset, but this time we are going to have to encode the training data before we train the model. Training might take a moment to complete since the dataset is quite large in comparison to the Iris data set we used before.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Lets encode our training data for the Random Forest Model</span>
<span class="n">encoded_x_train</span> <span class="o">=</span> <span class="n">encoder</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_train</span><span class="p">)</span>

<span class="c1"># Test the model</span>
<span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>
<span class="n">trees</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">75</span><span class="p">)</span>
<span class="n">trees</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">encoded_x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

<span class="c1"># get the time it took to train the model</span>
<span class="n">end</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">end</span><span class="o">-</span><span class="n">start</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Something-While-We-Wait">Something While We Wait<a class="anchor-link" href="#Something-While-We-Wait">&#182;</a></h4><p>Since the model is going to take a moment to train, lets look at some XKCD comics to pass the time.</p>
<ul>
<li><p><a href="http://www.xkcd.com/138/">Pointers</a></p>
</li>
<li><p><a href="https://xkcd.com/149/">Sandwich</a></p>
</li>
<li><p><a href="https://xkcd.com/246/">Labyrinth Puzzle</a></p>
</li>
<li><p><a href="https://xkcd.com/285/">Wikipedian Protester</a></p>
</li>
<li><p><a href="https://xkcd.com/552/">Correlation</a></p>
</li>
<li><p><a href="https://www.xkcd.com/303/">Compiling</a></p>
</li>
<li><p><a href="https://xkcd.com/859/">(</a></p>
</li>
<li><p><a href="https://xkcd.com/716/">Time Machine</a></p>
</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Testing-the-Model(s)">Testing the Model(s)<a class="anchor-link" href="#Testing-the-Model(s)">&#182;</a></h3><p>We're going to test this model just like we did with the Iris dataset, but this time we are going to have to encode the training data before we train the model.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">images_pred</span> <span class="o">=</span> <span class="n">trees</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">encoded_imgs</span><span class="p">)</span> 
<span class="nb">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">images_pred</span><span class="p">,</span> <span class="n">y_test</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">matthews_corrcoef</span><span class="p">(</span><span class="n">images_pred</span><span class="p">,</span> <span class="n">y_test</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Credit-Card-Fraud-Dataset">Credit Card Fraud Dataset<a class="anchor-link" href="#Credit-Card-Fraud-Dataset">&#182;</a></h2><p>As always, we are going to need a dataset to work on!
Credit Card Fraud Detection is a serious issue, and as such is something that data sciencists have looked into. This dataset is from a kaggle competition with over 2,000 Kernals based on it. Let's see how well Random Forests can do with this dataset!</p>
<p>Lets read in the data and use <em>.info()</em> to find out some meta-data</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">!</span>git clone <span class="s2">&quot;https://github.com/JarvisEQ/RandomData.git&quot;</span>
<span class="o">!</span>unzip RandomData/creditcardfraud.zip
<span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;creditcard.csv&quot;</span><span class="p">)</span>
<span class="n">data</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>What's going on with this V stuff?
Credit Card information is a bit sensitive, and as such raw information had to be obscured in some way to protect that information.</p>
<p>In this case, the data provider used a method know as PCA transformation to hide those features that would be considered sensitive. PCA is a very usefull technique for dimension reduction, a topic that we will be covering in a later lecture. For now, know that this technique allows us to take data and transform it in a way that maintains the patterns in that data.</p>
<p>Next, lets get get some basic statistical info from this data.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">data</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Some-important-points-about-this-data">Some important points about this data<a class="anchor-link" href="#Some-important-points-about-this-data">&#182;</a></h3><p>For most of the features, there is not a lot we can gather since it's been obscured with PCA. However there are three features that have been left out of the for us to see.</p>
<h4 id="1.-Time">1. Time<a class="anchor-link" href="#1.-Time">&#182;</a></h4><p>Time is the amount of time from first transaction in seconds. The max is 172792, so the data was collected for around 48 hours.</p>
<h4 id="2.-Amount">2. Amount<a class="anchor-link" href="#2.-Amount">&#182;</a></h4><p>Amount is the amount that the transaction was for. The denomination of the transactions was not given, so we're going to be calling them "Simoleons" as a place holder. Some interesting points about this feature is the STD, or standard diviation, which is 250§.That's quite large, but makes sense when the min and max are 0§ and 25,691§ respectively. There is a lot of variance in this feature, which is to be expected. The 75% amount is only in 77§, which means that the whole set skews to the lower end.</p>
<h4 id="3.-Class">3. Class<a class="anchor-link" href="#3.-Class">&#182;</a></h4><p>This tells us if the transaction was fraud or not. 0 is for no fraud, 1 if it is fraud. If you look at the mean, it is .001727, which means that only .1727% of the 284,807 cases are fraud. There are only 492 fraud examples in this data set, so we're looking for a needle in a haystack in this case.</p>
<p>Now that we have that out of the way, let's start making this model! We first need to split up our data into Test and Train sets.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">labels</span><span class="o">=</span><span class="s1">&#39;Class&#39;</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">loc</span><span class="p">[:,</span><span class="s1">&#39;Class&#39;</span><span class="p">]</span>

<span class="c1">#don&#39;t need that original data</span>
<span class="k">del</span> <span class="n">data</span>

<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">,</span> <span class="n">Y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">)</span>

<span class="c1">#don&#39;t need our X and Y anymore</span>
<span class="k">del</span> <span class="n">X</span><span class="p">,</span> <span class="n">Y</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Some-Points-about-Training">Some Points about Training<a class="anchor-link" href="#Some-Points-about-Training">&#182;</a></h3><p>With Neural Networks, we saw that the training times could easily reach the the hours mark and sometimes even days. With Random Forest, training time are much lower so we can finish training the model in realitively sort order, even when our dataset contains 284,807 entries. This done without the need of GPU acceleration, which Random Forests cannot take advantage of.</p>
<p>The area is left blank, but there's examples on how to make a Random Forest model earlier in the notebook that can be used as an example if you need it.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>
<span class="c1">#TODO, make the model</span>



<span class="n">end</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span>

<span class="c1"># this is going to tell you the training Time</span>
<span class="nb">print</span><span class="p">(</span><span class="n">end</span><span class="o">-</span><span class="n">start</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Testing-the-model">Testing the model<a class="anchor-link" href="#Testing-the-model">&#182;</a></h3><p>Whenever the model is done training, use the testing set find the Mathews Correlation Coefficient. We have examples of it from eariler in the notebook, so give them a look if you need an example.</p>
<p>We're going to be collecting the data of your guys results and graphing out the relationship between Number of Trees, Training Time, and Quality using MCC as the metric. Make sure you fill out the form below to tell us how your model did.</p>
<h3 id="Tell-us-your-results"><a href="https://docs.google.com/forms/d/e/1FAIpQLSdMh2J2I6X6DFuwIwXeZeagF8ah6ywPFjWjmMiNFDhONXuUDg/viewform?usp=sf_link">Tell us your results</a><a class="anchor-link" href="#Tell-us-your-results">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># TODO, use your model to predict for a the test set</span>
<span class="n">predictions</span> <span class="o">=</span> 
<span class="nb">print</span><span class="p">(</span><span class="n">matthews_corrcoef</span><span class="p">(</span><span class="n">Y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
 

