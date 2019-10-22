---
title: "Cleaning and Manipulating a Dataset with Python"
categories: ["sp19"]
authors: ['danielzgsilva', 'ionlights']
description: >-
  "In the fields of Data Science and Artificial Intelligence, your models and  analyses will only be as good as the data behind them. Unfortunately, you will find that the majority of datasets you encounter will be filled with  missing, malformed, or erroneous data. Thankfully, Python provides a number  of handy libraries to help you clean and manipulate your data into a usable state. In today's lecture, we will leverage these Python libraries to turn a messy dataset into a gold mine of value!"
---

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Today's-lecture-will-cover-how-to-load,-clean,-and-manipulate-a-dataset-using-Python">Today's lecture will cover how to load, clean, and manipulate a dataset using Python<a class="anchor-link" href="#Today's-lecture-will-cover-how-to-load,-clean,-and-manipulate-a-dataset-using-Python">&#182;</a></h2><h3 id="In-order-to-do-this-we'll-be-utilizing-a-Python-library-named-Pandas.">In order to do this we'll be utilizing a Python library named Pandas.<a class="anchor-link" href="#In-order-to-do-this-we'll-be-utilizing-a-Python-library-named-Pandas.">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Pandas-is-an-open-sourced-library-which-provides-high-performance,-easy-to-use-data-structures-and-data-analysis-tools-in-Python.-It-is-arguably-the-most-preferred-and-widely-used-tool-in-the-DS/AI-industry-for-data-munging-and-wrangling.">Pandas is an open sourced library which provides high-performance, easy-to-use data structures and data analysis tools in Python. It is arguably the most preferred and widely used tool in the DS/AI industry for data munging and wrangling.<a class="anchor-link" href="#Pandas-is-an-open-sourced-library-which-provides-high-performance,-easy-to-use-data-structures-and-data-analysis-tools-in-Python.-It-is-arguably-the-most-preferred-and-widely-used-tool-in-the-DS/AI-industry-for-data-munging-and-wrangling.">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>**If you do not yet have Python and Pandas installed on your machine I recommend using a package such as the <a href="https://www.anaconda.com/" target="_blank">Anaconda Distribution</a>. 
This can be installed for Windows, Linux, or Mac and will quickly install Python, Jupyter Notebook, and the most popular Data Science libraries onto your machine.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Importing-Libraries-and-Downloading-Data">Importing Libraries and Downloading Data<a class="anchor-link" href="#Importing-Libraries-and-Downloading-Data">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In order to use any Python library we need to first import the library...<br>Pandas is actually built on top of Numpy, a scientific computing library, and happens to work hand in hand with Pandas. We'll import this library as well.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This code downloads our dataset. To do this we'll utilize a script named gdown, which enables downloading files from Google Drive from the command line.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">!</span>wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl
<span class="o">!</span>chmod +x gdown.pl

<span class="o">!</span>./gdown.pl https://drive.google.com/open?id<span class="o">=</span>1uFRR5wtQTYjkZgfqUCtHfM1jJAT763Gm LA_Parking_Citations.csv
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Loading-in-a-Dataset-with-Python">Loading in a Dataset with Python<a class="anchor-link" href="#Loading-in-a-Dataset-with-Python">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>At this point you might be asking thinking, "Well this is cool and all, but where the heck can I get a dataset in the first place??"</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Fear not! There are a number of online repositories which supply both messy and clean datasets for almost any Data Science project you could imagine. Here are some of my favorites:</p>
<ul>
<li><a href="https://www.kaggle.com/" target="_blank">Kaggle</a>: A popular site within the Data Science community which hosts Machine Learning competitions. It contains a tremendous amount of datasets, all of which you can download.<ul>
<li>As a note, you can open up a kernel under any competition or dataset and the data will already be loaded into the notebook, no need to download to your machine!</li>
</ul>
</li>
<li><a href="https://cloud.google.com/bigquery/public-data/" target="_blank">Google Public Datasets</a></li>
<li><a href="https://aws.amazon.com/start-now/?sc_channel=BA&sc_campaign=elevator&sc_publisher=captivate" target="_blank">Amazon Web Services Public Datasets</a></li>
<li><a href="http://mlr.cs.umass.edu/ml/UC" target="_blank">Irvine Machine Learning Repository</a></li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>When you want to use Pandas to manipulate or analyze data, you’ll usually get your data in one of three different ways:</strong></p>
<ul>
<li>Convert a Pythonlist, dictionary or Numpy array to a Pandas data frame</li>
<li>Open a local file using Pandas, usually a CSV file, but could also be a tab delimited text file (like TSV), Excel, etc</li>
<li>Open a remote file through a URL or read from a database such as SQL</li>
</ul>
<p>In our case we will be loading our data set from a CSV (comma separated values file) which I downloaded from Kaggle:   <a href="https://www.kaggle.com/cityofLA/los-angeles-parking-citations" target="_blank">link to dataset</a></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Pandas-Components">Pandas Components<a class="anchor-link" href="#Pandas-Components">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Pandas has two core components:</p>
<ul>
<li><strong>Series</strong>: This is essentially a numpy.array, but for the most part these will be the columns within our Dataframes</li>
<li><strong>DataFrames</strong>: These are the bread and butter of pandas. They're equivalent to a table or an excel spreadsheet (made up of columns and rows)</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Inpsecting-and-Analyzing-a-Dataframe">Inpsecting and Analyzing a Dataframe<a class="anchor-link" href="#Inpsecting-and-Analyzing-a-Dataframe">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This dataset contains a line item for each ticket issued in the City of Los Angeles. Let's take a look...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>pd.DataFrame.shape quickly tells us the dimensions of a DataFrame</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The next two DataFrame methods can be used to tell us which datatypes our DataFrame consists of, as well as how many NULL values are found in each column</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Notice below that our Meter Id, Marked Time, and VIN columns have a significant number of NULL values. We'll deal with these in a bit...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This method below, .describe(), provides a statistical summary of our numerical columns (ints and floats)</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You can pass the method different datatypes you'd like a summary of...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Dropping-columns-from-a-Dataframe">Dropping columns from a Dataframe<a class="anchor-link" href="#Dropping-columns-from-a-Dataframe">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Lets take a look at the amount of missing data in each column</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We see that columns VIN, Marked Time, and Meter ID all have a high percent of NULL values, so we decide to simply drop these columns. <br> 
Let's say that for this analysis we're also not concerned with the Route or the Agency, nor the Longitude/Latitude so let's drop those as well.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Pass the drop() method a list of the columns we'd like to drop and specify the axis as 1 (for the columns axis). The inplace parameter allows this method to occur <strong>inplace</strong>, or on our current DataFrame.
Think of it as the difference between:</p>
<ul>
<li>x = x  + 1;</li>
<li>x++;         </li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">columns_to_drop</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;VIN&quot;</span><span class="p">,</span> <span class="s2">&quot;Marked Time&quot;</span><span class="p">,</span> <span class="s2">&quot;Meter Id&quot;</span><span class="p">,</span> <span class="s2">&quot;Route&quot;</span><span class="p">,</span> <span class="s2">&quot;Agency&quot;</span><span class="p">,</span> <span class="s2">&quot;Longitude&quot;</span><span class="p">,</span> <span class="s2">&quot;Latitude&quot;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For the sake of demonstration, we can also also drop rows with this method. Specify our axis as 0 for rows, and instead of column names we'll now use indice numbers</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Rows 0, 1, and 2 will be dropped from the DataFrame</span>
<span class="c1"># Also, notice we do not perform this method inplace. This way we are not permanently altering our DataFrame</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Creating-a-unique-index-for-your-DataFrame">Creating a unique index for your DataFrame<a class="anchor-link" href="#Creating-a-unique-index-for-your-DataFrame">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Pandas allows us to slice and access rows of our Dataframe utilizing the unique indice numbers. In many cases, it is helpful to use a unique identifying field from the data as its index , rather than having our rows labeled 0 - 999999. In this case,
Ticket Number would function as an excellent Index for us</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Ensuring Ticket Numbers are in fact, unique</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Indexing-your-Dataframe">Indexing your Dataframe<a class="anchor-link" href="#Indexing-your-Dataframe">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="pd.DataFrame.loc[-]--allows-us-to-do-label-based-indexing">pd.DataFrame.loc[ ]  allows us to do label-based indexing<a class="anchor-link" href="#pd.DataFrame.loc[-]--allows-us-to-do-label-based-indexing">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This means accessing records using their unique label (index), without regard to their position in the DataFrame. In our case the unique label is now the ticket number</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># This will return the record of ticket number 4346620795 (It happens to be the first row in our DataFrame) </span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="pd.DataFrame.iloc[-]-allows-us-to-do-position-based-indexing">pd.DataFrame.iloc[ ] allows us to do position-based indexing<a class="anchor-link" href="#pd.DataFrame.iloc[-]-allows-us-to-do-position-based-indexing">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This means accessing a row based on what row number it is in the DataFrame. To access the first record in the DataFrame (which we also pulled above) do:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This function also allows for Numpy like slicing of our DataFrame. For example, to retrieve the last 2,000 records of the DataFrame we can do:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Dealing-with-NaN-or-Inaccurate-values">Dealing with NaN or Inaccurate values<a class="anchor-link" href="#Dealing-with-NaN-or-Inaccurate-values">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">percent_missing</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">dtypes</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We'll go ahead and fill the numeric columns which contain NULL values with 0, these are Issue Time, Fine Amount, and Plate Expiration Date. We'll then convert these columns to integers after noticing their values are all whole numbers</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">percent_missing</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">*</span> <span class="mi">100</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">df</span><span class="p">)</span>
<span class="n">percent_missing</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">percent_missing</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Okay, now we're getting there. Let's recap: We started by dropping all the columns that we either weren't interested in, or simply had too many missing values to be useful. We then created a unique index for the data, Ticket Number, and filled in missing values in our numeric columns with an arbitrary value. Let's take a look at the dataset again to decide if any further manipulation is necessary...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Cleaning-up-our-Columns">Cleaning up our Columns<a class="anchor-link" href="#Cleaning-up-our-Columns">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The first thing I notice is that Plate Expiration Date is in integer form. We'd like to turn this column into a date-time type with a proper year-month format. Let's look at the unique values...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We have a couple things to deal with here. The first thing to tackle are the outliers... The expiration dates seem to range from year 2000 to 2099, therefore the integers 1 through 12 don't mean much. (0 came from our NULL values). I'm going to treat these outliers as missing dates and simply replace them with 0</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="To-do-this-let's-utilize-Numpy.where">To do this let's utilize Numpy.where<a class="anchor-link" href="#To-do-this-let's-utilize-Numpy.where">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>&emsp;&emsp;&emsp;np.where(condition, then, else)
<br>
<br>
This will loop through each row of the column we pass to it and check whether the condition is true. If True, apply the 'then' value, if not, apply the else value</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Awesome, we've replaced all of those outliers with 0. Now let's take a look at how to convert these integers to a date format</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>&emsp; We'll utilize <strong>pd.to_datetime()</strong> <br><br>
This method will parse through the column we pass to it and convert it to a Pandas <strong>Datetime</strong> format <br></p>
<ul>
<li>Datetime format is commonly used when dealing with Dates as it provides a great deal of functionality and makes these columns much easier to deal with</li>
<li>The parameter <strong>Errors</strong> communicates how to deal with elements that can't be interpreted as a date, <strong>Coerce</strong> says just make these NULL</li>
<li>We also pass the format of the column we'll be parsing, in this case the Plate Exp Date ints are in year-month format: <strong>%Y%m</strong></li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The last column which needs a bit of cleaning is the <strong>Issue Date</strong> column. We'd like to chop off the end of each string in the column since it seems every entry has 'T00:00:00' tacked on. Let's take a look at how we can do this...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Pandas provides a number of nifty and easy to use vectorized string operations in the way of <strong>pd.Series.str</strong>, some examples are:</p>
<ul>
<li>pd.Series.str.split()</li>
<li>pd.Series.str.replace()</li>
<li>pd.Series.str.contains() <br>
<br> And the one which we'll utilize...</li>
<li><strong>pd.Series.str.extract()</strong>
<br> This will allow us to extract the part of each string in the column that matches the Regular Expression we pass to it 
<br> The Regular Expression <strong>\d{4}-\d{2}-\d{2}</strong> will search for the pattern: Any 4 digits - Any 2 digits - Any 2 digits
<br>
<br> <em>If you're not familiar with RegEx don't worry too much as it's not the purpose of today's lecture</em>
<br> &emsp; <em>If you'd like to read more on RegEx visit: <a href="https://regexr.com/">https://regexr.com/</a>.</em></li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Okay, so we've extracted the date portion of <strong>Issue Date</strong>. As you can see below, this column is still an object, or essentially a string. Similar to Plate Expiration Date, we'd like to convert this to Datetime format so we can make use of Pandas' Datetime functionality later down the road.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">dtypes</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Note how this time the format to be parsed is a bit different. In this case, Issue Date is already in date format <strong>%Y-%m-%d</strong>, we just want to convert it to the datetime datetype</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We're lookin good!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="If-we've-got-time-we-can-cover-these-extra-topics">If we've got time we can cover these extra topics<a class="anchor-link" href="#If-we've-got-time-we-can-cover-these-extra-topics">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Calculated-Columns">Calculated Columns<a class="anchor-link" href="#Calculated-Columns">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>A big part of Machine Learning and Data Science is about brainstorming and creating new features to extract more information from the data than what is present at first glance. In this case we might question whether there is a correlation between number of tickets issued and the current season... Lets create a new feature, or column, in this dataset for Season of Issue Date</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creating a new column is as simple as</p>
<ul>
<li>df['New Column Name'] = Equation or Conditional used to set values in new column
<br><br> Here we'll use a fancy calculation against the month of Issue Date to determine the season (from 1 - 4) based off the month
<br> <strong>Series.dt</strong> provides a number of datetime functions if the column is in datetime format</li>
<li>Specifically, Series.dt.month returns the month of the datetime as a float E.G: January = 1.0</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#    Month      Season</span>
<span class="c1"># 12 | 1 | 2 = &#39;Winter&#39; or 1</span>
<span class="c1"># 3 | 4 | 5 = &#39;Spring&#39; or 2</span>
<span class="c1"># 6 | 7 | 8 = &#39;Summer&#39; or 3</span>
<span class="c1"># 9 | 10 | 11 = &#39;Fall&#39; or 4</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Sorting-and-filtering-a-Dataframe">Sorting and filtering a Dataframe<a class="anchor-link" href="#Sorting-and-filtering-a-Dataframe">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Now,-as-an-example,-we-could-filter-on-just-tickets-issued-in-the-Winter,-and-then-let's-sort-the-data-by-Issue-Date">Now, as an example, we could filter on just tickets issued in the Winter, and then let's sort the data by Issue Date<a class="anchor-link" href="#Now,-as-an-example,-we-could-filter-on-just-tickets-issued-in-the-Winter,-and-then-let's-sort-the-data-by-Issue-Date">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Filtering can be done by passing the conditional we want to filter on into the original DataFrame<br><br>Here, the inner conditional results in a boolean array of length 100000. It's true when Season is 1, and False otherwise <br> We pass this boolean array to our original DataFrame and this filters our data on just rows where our inner condition was found to be True</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>And, as expected, this new DataFrame is a subset of our orginal DataFrame</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Sometimes you may need to filter a dataset based on if a column contains a number of different values, and don't want to create a long OR statement... <br> In this case you could filter your DataFrame based off a list, like so:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As you may have guessed, we're going to attempt to filter our data on tickets issued in every season OTHER than winter. <br>
The technique we use is the same, but the conditional will look a bit different. Lets take a look...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Seasons: &#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">df_not_winter</span><span class="p">[</span><span class="s1">&#39;Issue Season&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">())</span> <span class="o">+</span> <span class="s1">&#39;    Shape: &#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">df_not_winter</span><span class="o">.</span><span class="n">shape</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Seasons: &#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">df_winter</span><span class="p">[</span><span class="s1">&#39;Issue Season&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">())</span> <span class="o">+</span> <span class="s1">&#39;    Shape: &#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">df_winter</span><span class="o">.</span><span class="n">shape</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As you can see, this new DataFrame contains all the seasons except Winter, and is the complentary set to the previous DataFrame we made</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="The-last-thing-I-wanted-to-cover-is-how-we-can-Sort-a-DataFrame">The last thing I wanted to cover is how we can Sort a DataFrame<a class="anchor-link" href="#The-last-thing-I-wanted-to-cover-is-how-we-can-Sort-a-DataFrame">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We'll utlize <strong>pd.DataFrame.sort_values()</strong>
<br><br>Which allows us to sort the rows of a dataset by column value. In this case let's sort all summer tickets by their Issue Date, and then by their Issue Time</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For sake of demonstration, we can also rearrange the columns of a DataFrame as well like so:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># get the list of all columns</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[105]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#rearrange the list of columns in the order we&#39;d like </span>
<span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Issue Date&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Issue time&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Issue Season&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Fine amount&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Violation code&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Violation Description&#39;</span><span class="p">,</span>
 <span class="s1">&#39;RP State Plate&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Plate Expiry Date&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Make&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Body Style&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Color&#39;</span><span class="p">,</span>
 <span class="s1">&#39;Location&#39;</span><span class="p">,]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Finally pass the list of new columns to our original DataFrame</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>ALL DONE ! <BR></h2>
<h3 id="Thanks-to-everyone-for-coming-out-tonight,-please-remember-to-sign-out!">Thanks to everyone for coming out tonight, please remember to sign out!<a class="anchor-link" href="#Thanks-to-everyone-for-coming-out-tonight,-please-remember-to-sign-out!">&#182;</a></h3>
</div>
</div>
</div>
 

