---
title: "A Walk Through the Random Forest"
linktitle: "A Walk Through the Random Forest"

date: "2019-09-25T17:30:00"
lastmod: "2019-09-25T17:30:00"

draft: false
toc: true
type: docs

weight: 2

menu:
  core_fa19:
    parent: Fall 2019
    weight: 2

authors: ["jarviseq", ]

urls:
  youtube: ""
  slides:  ""
  github:  "https://github.com/ucfai/core/blob/master/fa19/2019-09-25-rf-svm/2019-09-25-rf-svm.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-fa19-rf-svm"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/fa19/2019-09-25-rf-svm/2019-09-25-rf-svm.ipynb"

room: "MSB 359"
cover: "https://upload.wikimedia.org/wikipedia/commons/1/17/Tarvasj%C3%B5gi.jpg"

categories: ["fa19"]
tags: ["random-forests", "svms", "weak-models", "non-nn", ]
description: >-
  In this lecture, we explore powerful yet lightweight models that are often overlooked. We will see the power of combining multiple simple models together and how they can yield amazing results. You won't believe how easy it is to classify with just a line!
---
```python
import os
from pathlib import Path

if os.path.exists("/kaggle/input"):
    DATA_DIR = Path("/kaggle/input")
else:
    raise ValueError("We don't know this machine.")
```

## Overview

Before getting going on more complex examples, 
we're going to take a look at a very simple example using the Iris Dataset. 

The final example deals with credit card fraud, 
and how to identify if fraud is taking place based a dataset of over 280,000 entries. 


```python
# Importing the important stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
```

## Iris Data Set

This is a classic dataset of flowers. The goal is to have the model classify the types of flowers based on 4 factors. 
Those factors are sepal length, sepal width, petal length, and petal width, which are all measured in cm. 
The dataset is very old in comparison to many of the datasets we use, coming from a 
[1936 paper about taxonomy](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x).

### Getting the Data Set

Sklearn has the dataset built into the the library, so getting the data will be easy.
Once we do that, we'll do a test-train split.


```python
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.1)
```

### Making the Model

Making and Random Forests model is very easy, taking just two lines of code! 
Training times can take a second, but in this example is small so the training time is minimal.

```python
trees = RandomForestClassifier(n_estimators=150, n_jobs=-1)
trees.fit(X_train, Y_train)
```

sklearn has a few parameters that we can tweak to tune our model. 
We won't be going into those different parameters in this notebook, 
but if you want to give it a look, 
[here is the documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  

### We need to Figure out how well this model does

There are a few ways we are going to test for accuracy using a Confusion Matrix and Matthews Correlation Coefficient . 

#### Confusion Matrix

A Confusion Matrix shows us where the model is messing up. Below is an example from dataschool.io. The benefit of a confusion matrix is that it is a very easy way to visualise the performance of the model. 

![alt text](https://www.dataschool.io/content/images/2015/01/confusion_matrix_simple2.png)

```python
predictions = trees.predict(X_test)
confusion_matrix(Y_test, predictions)
```

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

```python
print(matthews_corrcoef(Y_test, predictions))
```

### Now, what about SVMs?

We want to see how well SVMs can work on the Iris, so let's see it in action.

First, let's define the models; one for linear, ploy and rbf.

```python
# SVM regularization parameter, we'll keep it simple for now
C = 1.0 

models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='poly', degree=3, C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C))
```

So you know what the parameters mean:
* degree refers to the degree of the polynomial
* gamma refer to the influence of a single training example reaches
> * with low values meaning ‘far’ and high values meaning ‘close’

Once we have the models defined, let's train them!

```python
models = (clf.fit(X_train, Y_train) for clf in models)
```

Now we are see how the confusion matrices look like:

```python
for clf in models:
    predictions = clf.predict(X_test)
    print(confusion_matrix(Y_test, predictions))
```

The confusion matrix is all nice and dandy, 
but let's check out what the Matthews Coefficient has to say about our models.

```python
for clf in models:
    predictions = clf.predict(X_test)
    print(matthews_corrcoef(Y_test, predictions))
```

That wasn't too bad was it? 
Both Random Forests and SVMs are very easy models to implement,
and its low training times means that the model can be used without the overheads associated with neural networks, 
which we will learn more about next week.

## Credit Card Fraud Dataset

As always, we are going to need a dataset to work on!
Credit card fraud detection is a serious issue, and as such is something that data scientists have looked into. This dataset is from a Kaggle competition with over 2,000 Kernals based on it. Let's see how well Random Forests can do with this dataset!

Lets read in the data and use *.info()* to find out some meta-data

```python
data = pd.read_csv(DATA_DIR / "train.csv")
data.info()
```

What's going on with this V stuff?
Credit Card information is a bit sensitive, and as such raw information had to be obscured in some way to protect that information.

In this case, the data provider used a method know as PCA transformation to hide those features that would be considered sensitive. 
PCA is a very useful technique for dimension reduction. 
For now, know that this technique allows us to take data and transform it in a way that maintains the patterns in that data.

If you are interested in learning more about PCA, 
[Consider checking out this article](https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-b836fb9c97e2). 
Unfortunately, there is a lot that we want to cover in Core and not enough time to do all of it. :(

Next, let's get some basic statistical info from this data.

```python
data.describe()
```

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

```python
X = data.drop(columns='Class', axis=1)
Y = data['Class']

# sklearn requires a shape with dimensions (N, 1), 
# so we expand dimensions of x and y to put a 1 in the second dimension
print(f"X shape: {X.shape} Y shape: {Y.shape}")
```

### Some Points about Training

With Random Forest and SVMs, training time is very quick, so we can finish training the model in realitively short order, 
even when our dataset contains 284,807 entries. This is done without the need of GPU acceleration, 
which Random Forests cannot take advantage of.

The area is left blank, 
but there's examples on how to make the models earlier in the notebook that can be used as an example if you need it. 
What model and the parameters you choose are up to you, so have fun!

```python
# Make the magic happen!
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
X_test = pd.read_csv(DATA_DIR / "test.csv")

# to expedite things: pass `n_jobs=-1` so you can run across all available CPUs
### BEGIN SOLUTION
rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
rf.fit(X, Y)
predictions = rf.predict(X_test.values)
### END SOLUTION
```

### Submitting your Solution

To submit your solution to kaggle, you'll need to save you data.

Luckily, we got the code you need to do that below. 

```python
predictions = pd.DataFrame({'Id': Y_test.index, 'Class': predictions})

predictions.to_csv('submission.csv', header=['Id', 'Class'], index=False)
```

## Thank You

We hope that you enjoyed being here today.

Please fill out [this questionaire](https://docs.google.com/forms/d/e/1FAIpQLSemUFE7YNrnKYT-KBUJcsWbmNkBIj_1aT0mtw3LszJLOMAXLA/viewform?usp=sf_link) so we can get some feedback about tonight's meeting.

We hope to see you here again next week for our Core meeting on *Neural Networks*.

### Vevu en Virtu 

