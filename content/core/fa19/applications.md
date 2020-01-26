---
title: "Applications"
linktitle: "Applications"

date: "2019-10-09T17:30:00"
lastmod: "2019-10-09T17:30:00"

draft: false
toc: true
type: docs

weight: 5

menu:
  core_fa19:
    parent: Fall 2019
    weight: 5

authors: ["jarviseq", ]

urls:
  youtube: ""
  slides:  ""
  github:  "https://github.com/ucfai/core/blob/master/fa19/2019-10-09-applications/2019-10-09-applications.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-fa19-applications"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/fa19/2019-10-09-applications/2019-10-09-applications.ipynb"

location: "MSB 359"
cover: "https://www.autodesk.com/products/eagle/blog/wp-content/uploads/2018/04/shutterstock_1011096853.jpg"

categories: ["fa19"]
tags: ["neural-nets", "applications", "random-forest", "svms", ]
abstract: >-
  You know what they are, but "how do?" In this meeting, we let you loose on a dataset to help you apply your newly developed or honed data science skills. Along the way, we go over the importance of visulisations and why it is important to be able to pick apart a dataset.
---
```python
# This is a bit of code to make things work on Kaggle
import os
from pathlib import Path

if os.path.exists("/kaggle/input/ucfai-core-fa19-applications"):
    DATA_DIR = Path("/kaggle/input/ucfai-core-fa19-applications")
else:
    DATA_DIR = Path("./")
```

# Dataset for the day: Suicide Preventation
## [Slides](https://docs.google.com/presentation/d/1fzw2j1BJuP3Z-Y1noB4bcEkjFUak_PxIKjHBC9_vp6E/edit?usp=sharing)

The [dataset](https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016) we will be using today is socio-economic data alongside suicide rates per country from 1985 to 2016. It is your task today to try to predict the suicide rate per 100,000 people in a give country. Building a good model for this can help find areas where there might be a high suicide rate so that prevention measures can be put in place to help people before it happens. 

We cannot natively use a SVM, Logistic Regression, or RF because they predict on categorical data, while today we will be making predictions on continuous data. Check out Regression Trees if you want to see these types of models applied to regression problems.

However, this problem can be changed to a categorical one so that you can use a RF or SVM. To do so, after the data analysis we will get some statistics on mean, min, max etc of suicide rate per 100,000 people column. Then, you can define ranges and assign them to a integer class. For example, assigning 0-5 suicides/100k as "Low", 5-50 as "High" etc. You can make as many ranges as you want, then train on those classes. In this case, we want to focus on producing actual values for this, so we will stick with regression, but this is something really cool you can try on your own! (Hint: use pandas dataframe apply function for this!)

Linear regression can work, although is a bit underwhelming for this task. So instead we will be using a Neural Network!

Let's dive in!

```python
# import all the libraries you need

# torch for NNs
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

# general imports
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Load in data and process
The data contains many different datatypes, such as floats, integers, strings, dates etc. We need to load this in and transform it all properly to something the models can understand. Once this is done, its up to you to build a model to solve this problem!

```python
dataset = pd.read_csv(DATA_DIR / "master.csv")
```

```python
dataset.head()
```

The first thing to notice is all the NaNs for HDI, meaning that there is missing data for those row entries. There is also a possibly useless row country-year as well. Lets see how many entires are missing HDI data first.

```python
print("Total entries: {}, null entries: {}".format(len(dataset["HDI for year"]), dataset["HDI for year"].isnull().sum()))
```

As you can see, most entires are null, so lets remove this column and the country-year column.

```python
dataset = dataset.drop("HDI for year", axis=1).drop("country-year", axis=1)
dataset.head()
```

```python
dataset.describe()
```

```python
dataset.info()
```

Now that looks much better. We need to transform the categories that are objects (like sex, country, age ranges) to number representations. For example, sex will become `Male = 0` and `Female = 1`. The countries and age-ranges will be similiarly encoded to integer values. Then we can describe our data again and see the full stats.

This is done using dictionaries that map's these keys to values and apply that to the dataframe. The gdp_for_year however has commas in the numbers, so we need a function that can strip these and convert them to integers.

```python
country_set = sorted(set(dataset["country"]))
country_map = {country : i for i, country in enumerate(country_set)}

sex_map = {'male': 0, 'female': 1}

age_set = sorted(set(dataset["age"]))
age_map = {age: i for i, age in enumerate(age_set)}

gen_set = sorted(set(dataset["generation"]))
gen_map = {gen: i for i, gen in enumerate(gen_set)}

def gdp_fix(x):
    x = int(x.replace(",", ""))
    return x

dataset = dataset.replace({"country": country_map, "sex": sex_map, "generation": gen_map, "age": age_map})
dataset[" gdp_for_year ($) "] = dataset.apply(lambda row: gdp_fix(row[" gdp_for_year ($) "]), axis=1)
```

```python
dataset.head()
```

```python
dataset.info()
```

```python
dataset.describe()
```

Now that is looking much better! However, as you can see the values vary pretty different, such as the year can be 1985-2016 and suicide rate could be from 0 to about 225. While you can train on this, its better if all of your data is within the same range. To do this, you would need to divide each value in the column, subtract its minimum value, then divide by its max value. This isn't required but sometimes can make your model train a lot faster and converge on a lower loss. For example on changing the range of year:

```python
print((dataset["year"] - 1985) / 31)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-e44eea9aecc7> in <module>
    ----> 1 (dataset["year"] - 1985) / 31
    

    NameError: name 'dataset' is not defined


Here you need to split the data for input into your NN model. 

If you using an NN, you need to use `torch.from_numpy()` to get torch tensors and use that to build a simple dataset class and dataloader. You'll also need to define the device to use GPU if you are using pytorch, check the previous lecture for how that works. The [pytorch documentation](https://pytorch.org/docs/stable/index.html) is also a great resource!

```python
X, Y = dataset.drop("suicides/100k pop", axis=1).values, dataset["suicides/100k pop"].values
```

```python
# Split data here using train_test_split
### BEGIN SOLUTION
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
### END SOLUTION
```

```python
print("X shape: {}, Y shape: {}".format(X.shape, Y.shape))
```

```python
# run this if you are using torch and a NN
class Torch_Dataset(Dataset):
  
  def __init__(self, data, outputs):
        self.data = data
        self.outputs = outputs

  def __len__(self):
        #'Returns the total number of samples in this dataset'
        return len(self.data)

  def __getitem__(self, index):
        #'Returns a row of data and its output'
      
        x = self.data[index]
        y = self.outputs[index]

        return x, y

# use the above class to create pytorch datasets and dataloader below
# REMEMBER: use torch.from_numpy before creating the dataset! Refer to the NN lecture before for examples
```

```python
# Lets get this model!
# for your output, it will be one node, that outputs the predicted value. What would the output activation function be?
### BEGIN SOLUTION


### END SOLUTION
```
