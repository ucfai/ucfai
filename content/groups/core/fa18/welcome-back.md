---
title: Welcome back! Featuring Plotting & Supercomputers
linktitle: Welcome back! Featuring Plotting & Supercomputers

date: '2018-08-29T18:00:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 1

menu:
  core_fa18:
    parent: Fall 2018

authors: [ionlights, waldmannly]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: https://kaggle.com/ucfaibot/core-fa18-welcome-back
  colab: ''

papers: {}

location: VAB 107
cover: ''

categories: [fa18]
tags: []
abstract: >-
  Welcome back to SIGAI! We'll be re-introducing SIGAI for newcomers and
  refreshing it for veterans. Following that, we'll cover some basics of
  generating graphs (a very common task for data science and research). If
  you're enticed, we'll also get you setup on the university's
  supercomputer, as all following meetings will stream from there! :smiley:

---

```python
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-welcome-back").exists():
    DATA_DIR /= "ucfai-core-fa18-welcome-back"
elif DATA_DIR.exists():
    # no-op to keep the proper data path for Kaggle
    pass
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-welcome-back/data
    DATA_DIR = Path("data")
```

## Welcome to SIGAI

### What is it?

We're the AI club on campus. Focusing on both industry skills and getting involved in research, we're here to foster the next generation of AI researchers and engineers at UCF.

### Who are we?

- John Muchovej
- Chas Kane
- Evan Waldmann
- Aidan Lakshman
- Richard DiBacco

### Our goals:

- Break away the ivory tower imposed around AI
- Provide lecture-workshops in which you'll leave with skills you can immediately use
- Cultivate an interest in many facets of artificial intelligence and machine learning
- Form, and foster, a competitive data science team which participates in competitions and also works with local businesses and governments.

## The Master Plan (for Fall 2018)

### Unit 0: Basics
- Welcome Back! Featuring Plotting &amp; Supercomputers
- Intro to Data Analysis with Pandas &amp; NumPy
- Let Data Speak Using Regression &amp; Plots

### Unit 1: Neural Networks
- Neural Networks &amp; Inuition, Using PyTorch
- De-convolving Neural Networks
- Programming Dies, as Machines Learn to Code (Recurrent Neural Networks)
- Generative Adversarial Networks // Variational AutoEncoders (TBD)

### Unit 2: Time-Traveling
- Heuristics
- Decision Trees
- Support Vector Machines

### Unit 3: Applied
- 2-4 workshops where we, as a group, tear apart datasets, applying what we've learned and share insights/conclusions drawn

---

## Accessing Newton

Newton is the name of UCF's GPU cluster. It's where we'll conduct all our lectures, and you'll also have access to the cluster throughout the semester; although we strongly request that you contact John prior to running anything on your own through your `sigai.student` account.

### Getting Your Keys

This section is to be spoken, only. Not present anywhere except the lecture room.

### With the Keys: MacOS / Linux First

We'll be assuming you placed your keys in `~/Downloads` (your user's Downloads directory).

1. `mkdir ~/.ssh/`
1. `unzip Downloads/sigai.student<N>.zip`
1. `cp Downloads/sigai.student<N>/sig* ~/.ssh/`
1. `cat Downloads/sigai.student<N>/config >> ~/.ssh/config`
1. `cat ~/.ssh/sigai.student<N>_pass.txt`
Copy this into your clipboard, you'll need it in the next step!
1. Open a Terminal, and try this: `ssh sigai.newton`
1. You'll be prompted for the passphrase you copied to your clipboard earlier. Paste it and hit "Enter"

If all went well, you'll be greeted with this:
```
+===================================================+
| Welcome to the Newton HPC at the UCF Advanced     |
| Research Computing Center.                        |
|---------------------------------------------------|
| Problems? Email:  <redated>                       |
+===================================================+
[sigai.student<N>@<redacted> ~]$
```

### With the Keys: Windows
We'll be assuming you placed your keys in `~/Downloads` (your user's Downloads directory).

1. Download [PuTTy](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) – use [ucfsigai.org/tools/putty](https://ucfsigai.org/tools/putty), also downlo
1. Open `File Explorer` and unzip `sigai.student<N>.zip`
1. Double-click `sigai.student<N>.ppk` &ndash; you'll be prompted for a passphrase, this is in `sigai.student<N>_pass.txt`, copy and paste into the prompt.

### Now that we're all logged in...
1. `git clone https://github.com/ucfai/core`
1. `cd core/fa18`
1. `./launch.sh`
1. Follow the instructions printed out by `./launch.sh` and open `localhost:19972` on your browser.
1. Navigate to `08-29-welcome-back/welcome-back.ipynb` and open it &ndash; you should find yourself in this very notebook!

----

## Get a Taste of Python and Plotting

In order to understand what your data is and how dirty is it (because all data is dirty), you need to understand what that data looks like. Through plotting you can easily see trends and outliers that will help you determine if your data is what you think it is. 

First things first, you have to download your find what package your comfortable with there are loads, so feel free to look around. For the purposes of this meeting we are going to be using plotnine because it is basically the same as ggplot2 from R which is a very powerful library that is widely used in many different fields (there is a ggplot package as well, but that was giving me some issues).

If you take a look at [`fa18.env.yml`][fa18-env], you'll find the Anaconda Environment (very similar to `pip virtualenv`) we'll be using for the semester &ndash; you don't need to worry about this for the semester, though.

Then you have to import the libraries and read in your data. 

[fa18-env]: https://github.com/ucfsigai/data-science/blob/master/fa18/fa18.env.yml


```python
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from plotnine import *
```


```python
df = pd.read_csv(DATA_DIR / 'iris/Iris.csv')
df.head()
```

So now we have our data. We also want our data in this format, which is normal referred to as a data frame. Each row is an entry of data and each column is a type of data that was collected. With this, we can now start graphing. 


```python
ggplot(aes(x='Species', y='SepalWidthCm', color='Species'), data=df) + \
    geom_boxplot(aes(color='Species')) + \
    xlab("Species") + \
    ylab("Width") + \
    ggtitle("Comparing Sepal length and width across different species")
```

You can that we can get some nice looking graphs very easily. But what if we wanted to customized them more? 


```python
(ggplot(df, aes('Species','SepalLengthCm')) + 
    geom_boxplot(aes(color=('Species'))) +
    labs(title='graph title', x='x title', color='legend title', y='y title') +
    scale_x_discrete(labels=['setosa', 'versicolor', 'virginica']) + 
    scale_color_manual(labels=['setosa', 'versicolor', 'virginica'], values=['#000000', '#9ebcda', '#8856a7'])
    + theme_bw())
```

The customization is also pretty easy, but one of the things that most non statistician love about R is that they can add a fitted linear model line with about 25 characters. 


```python
(ggplot(df, aes(x='SepalWidthCm', y='SepalLengthCm', color=('Species'))) +
    geom_point() + stat_smooth(method='lm') +
    labs(title="graph title", x="x title", color="legend title", y="y title") +
    scale_color_manual(labels=['one', 'two', 'three'], values=['#000000', '#9ebcda', '#8856a7']) +
    theme_classic())
```

Sometimes, seperating the data by category can help you understand the trends better. This is called faceting. Note that since ggplot started in R changing the names of individual graphs is somewhat hard; an easy work around for this is to change the string factors in the data since the names depend on the species names from data.


```python
(ggplot(df, aes(x='SepalWidthCm', y='SepalLengthCm', color=('Species'))) + 
    geom_point() + stat_smooth(method='lm') + facet_wrap('~Species') +
    labs(title="graph title", x="x title", color="legend title", y="y title") + 
    scale_color_manual(values=['#000000', '#9ebcda', '#8856a7']) +
    theme_bw())
```

Now lets try a different dataset. 


```python
pkdata = pd.read_csv(DATA_DIR / 'pokemon/Pokemon.csv')
pkdata.rename(lambda x: str(x).replace(" ", ""), axis="columns", inplace=True)
pkdata.head()
```


```python
(ggplot(pkdata, aes('Legendary', 'Attack')) + geom_boxplot() +
    labs(title='Legendary Pokemon have a higher distribution of\n attack values than non-Legendary Pokemon') +
    theme_xkcd())
```


```python
(ggplot(pkdata, aes('Attack', 'Defense', color="Generation")) +
    geom_point() +
    theme_xkcd())
```


```python
(ggplot(pkdata, aes('Attack', 'Defense', color="Generation")) +
    geom_point() + facet_wrap('~Type1') +
    theme_xkcd())
```


```python
(ggplot(pkdata, aes('Legendary',  'Total')) + geom_boxplot() +
    labs(title='Sum of Total Stats Separated by\n Type and Legendary Status') +
    facet_wrap('~Type1') +
    theme_xkcd())
```

Now what if we wanted to save this graph? 


```python
p = (ggplot(pkdata, aes('Legendary', 'Total')) + geom_boxplot() +
     labs(title='Sum of Total Stats Separated by\n Type and Legendary Status') +
     facet_wrap('~Type1') + 
     theme_xkcd())

p.save(filename='pokemanPlot.png', height=5, width=5, units='in', dpi=150)
```

Now you have the basics of graphing. Keep in mind that this library translates almost exactly into code for `ggplot2` in `R`. If you want to refine your skills, the best thing you can do is to find a graph that you think is cool and try to make it exactly. Doing this, you will quickly learn the quarks of `ggplot`. 


You can check out these links for more information on plotnine: 
- https://www.kaggle.com/residentmario/grammar-of-graphics-with-plotnine-optional?scriptVersionId=4327772
- http://pltn.ca/plotnine-superior-python-ggplot/

And you can check out these for some more help with graphing using `ggplot`:
- https://tutorials.iq.harvard.edu/R/Rgraphics/Rgraphics.html
- http://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html
