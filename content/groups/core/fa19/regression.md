---
title: Getting Started, Regression
linktitle: Getting Started, Regression

date: '2019-09-18T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 1

menu:
  core_fa19:
    parent: Fall 2019

authors: [jarviseq, ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: https://kaggle.com/ucfaibot/core-fa19-regression
  colab: ''

papers: {}

location: MSB 359
cover: ''

categories: [fa19]
tags: [regression, linear regression, logistic regression, statistics]
abstract: >-
  You always start with the basics, and in this meeting we are doing just
  that! We will be going over what Machine Learning consists of and how we
  can use models to do awesome stuff!

---

```python
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-regression").exists():
    DATA_DIR /= "ucfai-core-fa19-regression"
elif DATA_DIR.exists():
    # no-op to keep the proper data path for Kaggle
    pass
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-regression/data
    DATA_DIR = Path("data")
```

First thing first, we need to import some packages:
* `matplotlib` allows us to graph 
* `numpy` is a powerful Linear Algebra library
* `pandas` allows us to Extract, Load, and Transform (ETL) datasets
* `sklearn` is a great Machine Learning library


```python
# import some important stuff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import datasets, linear_model
```

## Basic Example 

The data for this example is arbitrary (we'll use real data in a bit), but there is a clear linear relationship here.

Let's also graph this to demonstrate that relationship:


```python
# Get some data 
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# Let's plot the data to see what it looks like
plt.scatter(x, y, color = "black") 
plt.show()
```

Now, let's use "least squares estimation" to try minimizing the **squared error** of our function's output and the data we plotted.

$SS_{xy}$ is the cross deviation about $x$, and $SS_{xx}$ is the deviation about $x$.

[It's basically some roundabout algebra methods to optimize a function.](https://www.amherst.edu/system/files/media/1287/SLR_Leastsquares.pdf) 

The concept isn't super complicated but it gets hairy when you do it by hand.


```python
# calculating the coefficients

# number of observations/points 
n = np.size(x) 

# mean of x and y vector 
m_x, m_y = np.mean(x), np.mean(y) 

# calculating cross-deviation and deviation about x 
SS_xy = np.sum(y*x - n*m_y*m_x) 
SS_xx = np.sum(x*x - n*m_x*m_x) 

# calculating regression coefficients 
b_1 = SS_xy / SS_xx 
b_0 = m_y - b_1*m_x

#var to hold the coefficients
b = (b_0, b_1)

#print out the estimated coefficients
print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1])) 
```

But, we don't need to directly program all of the maths everytime we do linear regression.

`sklearn` has built in functions that allows you to quickly do Linear Regression with just a few lines of code.

We're going to use `sklearn` to make a model and then plot it using `matplotlib`.


```python
# Sklearn learn require this shape
x = x.reshape(-1,1)
y = y.reshape(-1,1)

# making the model
regress = linear_model.LinearRegression()
regress.fit(x, y)
```

And now, lets see what the model looks like:


```python
# plotting the actual points as scatter plot 
plt.scatter(x, y, color = "black", 
           marker = "o", s = 30) 

# predicted response vector 
y_pred = b[0] + b[1]*x 

# plotting the regression line 
plt.plot(x, y_pred, color = "blue") 

# putting labels 
plt.xlabel('x') 
plt.ylabel('y') 

# function to show plot 
plt.show()
```

So now we can make predictions with new points based off our data


```python
# here we can try out any data point
print(regress.predict([[6]]))
```

---

## Applied Linear Regression


### The Ames Housing Dataset 
> Ames is a city located in Iowa.
> 
> - This data set consists of all property sales
collected by the Ames City Assessor’s Office between the years
of 2006 and 2010.
> - Originally contained 113 variables and 3970 property sales
pertaining to the sale of stand-alone garages, condos, storage
areas, and of course residential property.
> - Distributed to the public as a means to replace the old Boston
Housing 1970’s data set.  
> - [Link to Original](http://lib.stat.cmu.edu/datasets/boston) 
> - The "cleaned" version of this dataset contains 2930 observations along with 80
predictor variables and two identification variables.

### What was the original purpose of this data set? 

- Why did the City of Ames decide to collect this data? 
- What does the prices of houses effect?

### What's inside? 

- This ”new” data set contains 2930 (n=2930) observations along with 80
predictor variables and two identification variables. 
- You can [read the paper that introduced this data][paper] and an [exhaustive breakdown of the variables][breakdown] if you want.

[paper]: http://jse.amstat.org/v19n3/decock.pdf
[breakdown]: http://jse.amstat.org/v19n3/decock/DataDocumentation.txt

### *Quick Summary*
------
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

### *Question to Answer:*
- What is the linear relationship between sale price on above ground 
  living room area?

To try answering this, let's visually investigate what we're trying to predict.
We'll kick this off with getting some summary statistics.


```python
housing_data =  pd.read_csv(DATA_DIR / "house-prices-advanced-regression-techniques/train.csv", delimiter="\t") 

# Mean Sales price 
mean_price = np.mean(housing_data["SalePrice"])
print("Mean Price : " + str(mean_price))

# Variance of the Sales Price 
var_price = np.var(housing_data["SalePrice"], ddof=1)
print("Variance of Sales Price : " + str(var_price))

# Median of Sales Price 
median_price = np.median(housing_data["SalePrice"])
print("Median Sales Price : " + str(median_price))

# Skew of Sales Price 
skew_price = st.skew(housing_data["SalePrice"])
print("Skew of Sales Price : " + str(skew_price))

housing_data["SalePrice"].describe()
```

Another way we can view our data is with a box and whisker plot.


```python
plt.boxplot(housing_data["SalePrice"])
plt.ylabel("Sales Price")
plt.show()
```

Now let's look at sales price on above ground living room area. 


```python
plt.scatter(housing_data["GrLivArea"], housing_data["SalePrice"])
plt.ylabel("Sales Price")
plt.show()
```

Finally, lets generate our model and see how it predicts Sales Price!!


```python
# we need to reshape the array to make the sklearn gods happy
area_reshape = housing_data["GrLivArea"].values.reshape(-1,1)
price_reshape = housing_data["SalePrice"].values.reshape(-1,1)

# Generate the Model
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(area_reshape, price_reshape)
price_prediction = model.predict(area_reshape)

# plotting the actual points as scatter plot 
plt.scatter(area_reshape, price_reshape) 

# plotting the regression line 
plt.plot(area_reshape, price_prediction, color = "red") 

# putting labels 
plt.xlabel('Above Ground Living Area') 
plt.ylabel('Sales Price') 

# function to show plot 
plt.show()
```

--------------------------------------------

## **Applied Logistic Regression**

--------------------------------------------


```python
# we're going to need a different model, so let's import it
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
```

For Logistic Regression, we're going to be using a real dataset.

You can find this data from UCI's Machine Learning Repository, or on Kaggle alongside this notebook.

* [UCI: Adult Dataset (AKA: Census Income)][uci]
* [Kaggle: Adult Census Income][kaggle]

[uci]: https://archive.ics.uci.edu/ml/datasets/Adult
[kaggle]: https://www.kaggle.com/uciml/adult-census-income

On Kaggle, we've packaged up the data for you, so now we can play around with it.

But before that, we need to read in the data &ndash; `pandas` has the functions we need to do this


```python
# read_csv allow us to easily import a whole dataset
data = pd.read_csv(
    DATA_DIR / "adult-census-income/adult.csv",
    names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
)

# this tells us whats in it 
print(data.info())
```


```python
# data.head() gives us some the the first 5 rows of the data
data.head()
```

The code below will show us some information about the *continunous* parameters that our dataset contains. 

* `age` is Age 
* `fnlwgt` is final weight, or the number of people that are represented in this group relative to the overall population of this dataset. 
* `education-num` is a numerical way of representing Education level
* `capital-gain` is the money made investments
* `capital-loss` is the loss from investments
* `hours-per-week` is the number of hours worked during a week


```python
# this is the function that give us some quick info about continous data in the dataset
data.describe()
```

**Now here is the Qustion:**
* Which one of these parameters are best in figuring out if someone is going to be making more then 50k a year?
* Make sure you choose a continunous parameter, as categorical stuff isn't going to work 


```python
# put the name of the parameter you want to test
### BEGIN SOLUTION
test = "capital-gain"
### END SOLUTION
```


```python
# but before we make our model, we need to modify our data a bit

# little baby helper function
def incomeFixer(x):
    if x == " <=50K":
        return 0
    else:
        return 1

# change the income data into 0's and 1's
data["income"] = data.apply(lambda row: incomeFixer(row['income']), axis=1)

# get the data we are going to make the model with 
x = np.array(data[test])
y = np.array(data["income"])

# again, lets make the scikitlearn gods happy
x = x.reshape(-1,1)

# Making the test-train split
splits = train_test_split(x ,y ,test_size=0.25, random_state=42)
x_train, x_test, y_train, y_test = splits
```


```python
# now make data model!
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x_train, y_train)
```


```python
# now need to test the model's performance
print(logreg.score(x_test, y_test))
```

## Thank You

We hope that you enjoyed being here today.

Please fill out [this questionaire](https://docs.google.com/forms/d/e/1FAIpQLSe8kucGh3_2Dcm7BFPv89qy-F4_BZKF-_Jm0nie37Ty6FuL9g/viewform?usp=sf_link) so we can get some feedback about tonight's meeting.

We hope to see you here again next week for our core meeting on *Random Forests and Support Vector Machines*.

### Live in Virtue
