---
title: "Introduction to Statistics, Featuring Datascience"
linktitle: "Introduction to Statistics, Featuring Datascience"

date: "2020-02-06T18:00:00"
lastmod: "2020-02-06T18:00:00"

draft: false
toc: true
type: docs

weight: 3

menu:
  supplementary_sp20:
    parent: Spring 2020
    weight: 3

authors: ["calvinyong", "jordanstarkey95", ]

urls:
  youtube: ""
  slides:  ""
  github:  "https://github.com/ucfai/supplementary/blob/master/sp20/02-06-stats-intro/02-06-stats-intro.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/supplementary-sp20-stats-intro"
  colab:   "https://colab.research.google.com/github/ucfai/supplementary/blob/master/sp20/02-06-stats-intro/02-06-stats-intro.ipynb"

location: "ENG2 203"
cover: "https://www.upgrad.com/blog/wp-content/uploads/2018/02/FI2_Statistics-for-Data-Science-uai-1440x438.png"

categories: ["sp20"]
tags: ["statistics", "descriptive statistics", "datascience", "data cleaning", "data prep", "graphing", "correlations", ]
abstract: >-
  Data Science is grounded in statistics, and to make sense of your data before predicting with any model requires some statistical analysis and lots of charts. A significant portion of your time and resources are spent on data preparation and clearning, not model building. How do we clean data? How do we make meaningful observations? How can you create graphs that represent data trends? Answers to these questions are highly sought after in all computationally-driven fields, so learning the basics here is a great start to your future in AI, Machine Learning, and Data Science!
---
## Purpose

The goal of this workshop is to provide the essential statistical knowledge required for data science.

To demonstrate these essentials, we'll look at a 

This workshop assumes you have reviewed the supplementary [Python3 workshop](https://ucfai.org/supplementary/sp20/math-primer-python-bootcamp) and core [Linear Regression workshop](https://ucfai.org/core/sp20/linear-regression).

## Introduction

Lets look at how statistical methods are used in an applied machine learning project:

* Problem Framing: Requires the use of exploratory data analysis and data mining.
* Data Understanding: Requires the use of summary statistics and data visualization.
* Data Cleaning: Requires the use of outlier detection, imputation and more.
* Data Selection: Requires the use of data sampling and feature selection methods.
* Data Preparation: Requires the use of data transforms, scaling, encoding and much more.
* Model Evaluation: Requires experimental design and resampling methods.
* Model Configuration: Requires the use of statistical hypothesis tests and estimation statistics.
* Model Selection: Requires the use of statistical hypothesis tests and estimation statistics.
* Model Presentation: Requires the use of estimation statistics such as confidence intervals.
* Model Predictions: Requires the use of estimation statistics such as prediction intervals

[Source: https://machinelearningmastery.com/statistics_for_machine_learning/]

## Descriptive and Inferential Statistics

**Descriptive statistics** identify patterns in the data, but they don't allow for making hypotheses about the data.

Within descriptive statistics, there are three measures used to describe the data: *central tendency* and *deviation*. 

* Central tendency tells you about the centers of the data. Useful measures include the mean, median, and mode.
* Variability tells you about the spread of the data. Useful measures include variance and standard deviation.
* Correlation or joint variability tells you about the relation between a pair of variables in a dataset. Useful measures include covariance and the correlation coefficient.

**Inferential statistics** allow us to make hypotheses (or inferences) about a sample that can be applied to the population. 

In statistics, the **population** is a set of all elements or items that you’re interested in. Populations are often vast, which makes them inappropriate for collecting and analyzing data. That’s why statisticians usually try to make some conclusions about a population by choosing and examining a representative subset of that population.

This subset of a population is called a **sample**. Ideally, the sample should preserve the essential statistical features of the population to a satisfactory extent. That way, you’ll be able to use the sample to glean conclusions about the population.




```
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
```

```
## Load the Boston dataset into a variable called boston
boston = load_boston()
```

```
## Separate the features from the target
x = boston.data
y = boston.target
```

To view the dataset in a standard tabular format with the all the feature names, you will convert this into a pandas dataframe.

```
## Take the columns separately in a variable
columns = boston.feature_names

## Create the Pandas dataframe from the sklearn dataset
boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
```

## Descriptive Statistics

This portion serves as a very basic primer on Descriptive statistics and will explain concepts which are fundamental to understanding Inferential Statistics, its tools and techniques. We will be using Boston House Price dataset: 

https://www.kaggle.com/c/boston-housing

Here is the Dataset description: 

* crim
 * per capita crime rate by town.

* zn
 * proportion of residential land zoned for lots over 25,000 sq.ft.

* indus
 * proportion of non-retail business acres per town.

* chas
 * Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

* nox
 * nitrogen oxides concentration (parts per 10 million).

* rm
 * average number of rooms per dwelling.

* age
 * proportion of owner-occupied units built prior to 1940.

* dis
 * weighted mean of distances to five Boston employment centres.

* rad
 * index of accessibility to radial highways.

* tax
 * full-value property-tax rate per \$10,000.

* ptratio
 * pupil-teacher ratio by town.

* black
 * 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

* lstat
 * lower status of the population (percent).

* medv
 * median value of owner-occupied homes in \$1000s.



### Summary Statistics

To begin learning about the sample, we uses pandas' `describe` method, as seen below. The column headers in bold text represent the variables we will be exploring. Each row header represents a descriptive statistic about the corresponding column.

```
boston_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
    </tr>
  </tbody>
</table>
</div>



`describe` isnt particularly enlightening on the distributions of our data 
but can help use figure out how to approach our visualization techniques. Before we explore essential graphs for exploring our data, lets use a few more important pandas methods to aid in our exploratory data analysis task.

```
print ("Rows     : " , boston_df.shape[0])
print ("Columns  : " , boston_df.shape[1])
print ("\nFeatures : \n" , boston_df.columns.tolist())
print ("\nMissing values :  ", boston_df.isnull().sum().values.sum())
print ("\nUnique values :  \n",boston_df.nunique())
print('\n')
print(boston_df.head())
```

    Rows     :  506
    Columns  :  13
    
    Features : 
     ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    
    Missing values :   0
    
    Unique values :  
     CRIM       504
    ZN          26
    INDUS       76
    CHAS         2
    NOX         81
    RM         446
    AGE        356
    DIS        412
    RAD          9
    TAX         66
    PTRATIO     46
    B          357
    LSTAT      455
    dtype: int64
    
    
          CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT
    0  0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98
    1  0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14
    2  0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03
    3  0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94
    4  0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33
    
    [5 rows x 13 columns]


We first show the shape of our dataset. We have 506 rows for our 13 features (columns). This is a relatively nice dataset in that there arent many missing values. A future supplementary lecture in preprocessing will cover techniques in dealing with missing values. 
We can see that there is a feature (CHAS) which has 2 unique values. This could indicate that it is a catgeorical variables. There are three types of statistical data we may be dealing with: 

* Numerical (Quantitative) data have meaning as a measurement, such as a person’s height, weight, IQ, or blood pressure; or they’re a count, such as the number of stock shares a person owns or how many teeth a dog has. Numerical data can be further broken into two types: discrete and continuous.

 * Discrete data represent items that can be counted; they take on possible values that can be listed out. The list of possible values may be fixed (also called finite); or it may go from 0, 1, 2, on to infinity (making it countably infinite). For example, the number of heads in 100 coin flips takes on values from 0 through 100 (finite case), but the number of flips needed to get 100 heads takes on values from 100 (the fastest scenario) on up to infinity (if you never get to that 100th heads).  

 * Continuous data represent measurements; their possible values cannot be counted and can only be described using intervals on the real number line. For example, the exact amount of gas purchased at the pump for cars with 20-gallon tanks would be continuous data from 0 gallons to 20 gallons, represented by the interval [0, 20], inclusive. Continuous data can be thought of as being uncountably infinite. 

* Categorical (Qualitative) data represent characteristics such as a person’s gender, marital status, hometown, or the types of movies they like. Categorical data can take on numerical values (such as “1” indicating married and “2” indicating unmarried), but those numbers don’t have mathematical meaning. The process of giving these mathematical meaning for our model to understand is variable encoding. This will be covered in the preprocessing supplementary lecture.

* Ordinal data mixes numerical and categorical data. The data fall into categories, but the numbers placed on the categories have meaning. For example, rating a restaurant on a scale from 0 (lowest) to 4 (highest) stars gives ordinal data. Ordinal data are often treated as categorical, where the groups are ordered when graphs and charts are made. However, unlike categorical data, the numbers do have mathematical meaning. For example, if you survey 100 people and ask them to rate a restaurant on a scale from 0 to 4, taking the average of the 100 responses will have meaning. This would not be the case with categorical data.

### Central Tendencies

The central tendencies are values which represent the central or 'typical' value of the given distribution. The three most popular central tendency estimates are the mean, median and mode. Typically, in most cases, we resort to using mean (for normal distributions) and median (for skewed distributions) to report central tendency values.

A good rule of thumb is to use mean when outliers don't affect its value and median when it does (Bill Gates joke, anyone?).

Calculating the mean and median are extremely trivial with Pandas. In the following cell, we have calculated the mean and median of the average number of rooms per dwelling.  As we can see below, the mean and the median are almost equal.

```
rooms = boston_df['RM']
rooms.mean(), rooms.median()
```




    (6.284634387351787, 6.2085)



If the mean, median and the mode of a set of numbers are equal, it means, the distribution is symmetric. The more skewed is the distribution, greater is the difference between the median and mean, and we should lay greater emphasis on using the median as opposed to the mean

### Measures of Spread

Apart from the central or typical value of the data, we are also interested in knowing how much the data spreads. That is, how far from the mean do values tend to go. Statistics equips us with two measures to quantitatively represent the spread: the variance and the standard deviation. They are dependent quantities, with the standard deviation being defined as the square root of variance.

```
rooms.std(), rooms.var()
```




    (0.7026171434153237, 0.4936708502211095)



The mean and the standard deviation are often the best quantities to summarize the data for distributions with symmetrical histograms without too many outliers. As we can see from the histogram below, this indeed is the case for RM feature. Therefore, the mean and the standard deviation measures are sufficient information and other tendencies such as the median does not add too much of extra information.

```
sns.distplot(rooms)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f40c1baee80>




<img src="data:image/png;base64,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
NxHIMBkE1AAAAABJRU5ErkJggg==
">

This is an example of a normal (Gaussian) distribution. It is ideal that our continuous variables folllow this distribution because of the central limit theorem. See [here](https://towardsdatascience.com/why-data-scientists-love-gaussian-6e7a7b726859) for an explanation on why the Gaussian is ideal for machine learning models.

```
stats.normaltest(rooms)
```




    NormaltestResult(statistic=37.89574305099423, pvalue=5.90260814347777e-09)



`normaltest` returns a 2-tuple of the chi-squared statistic, and the associated p-value. Given the null hypothesis that x came from a normal distribution, the p-value represents the probability that a chi-squared statistic that large (or larger) would be seen. If the p-val is very small, it means it is unlikely that the data came from a normal distribution.

Here is an example of a skewed dsitribution and how to fix it in order to fit a normal distribution.


```
age = boston_df['AGE']
print(age.std(), age.mean())
sns.distplot(age)
```

    28.072042765119424 67.38433734939758





    <matplotlib.axes._subplots.AxesSubplot at 0x7f40bb53c940>




<img src="data:image/png;base64,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
AAAAAElFTkSuQmCC
">

There are many ways to transform skewed data in order to fit a normal distribution. This will transform the data into a normal distribution. Moreover, you can also try Box-Cox transformation which calculates the best power transformation of the data that reduces skewness although a simpler approach which can work in most cases would be applying the natural logarithm. More details about Box-Cox transformation can be found here and here

```
log_age = np.log(age)
print(log_age.std(), log_age.mean())
sns.distplot(log_age)
```

    0.6225646449674217 4.0690897314054





    <matplotlib.axes._subplots.AxesSubplot at 0x7f40bb3def98>




<img src="data:image/png;base64,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
">

Although there is a long left tail, the log transformation reduces the deviation of the data. Can we measure normalcy? Yes! Rather than read from a Histogram, we can perform the Normal Test. This comes in the Scipy package and that lets us calculate the probability that the distrbution is normal, by chance.

### Univariate Analysis

It is a common practice to start with univariate outlier analysis where you consider just one feature at a time. Often, a simple box-plot of a particular feature can give you good starting point. You will make a box-plot using `seaborn` and you will use the `DIS` feature.

```
sns.boxplot(x=boston_df['DIS'])
plt.show()
```


<img src="data:image/png;base64,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
">

 A box-and-whisker plot is helpful for visualizing the distribution of the data from the mean. Understanding the distribution allows us to understand how far spread out her data is from the mean. Check out [how to read and use a Box-and-Whisker plot](https://flowingdata.com/2008/02/15/how-to-read-and-use-a-box-and-whisker-plot/).


The above plot shows three points between 10 to 12, these are **outliers** as they're are not included in the box of other observations. Here you analyzed univariate outlier, i.e., you used DIS feature only to check for the outliers.

An outlier is considered an observation that appears to deviate from other observations in the sample. We can spot outliers in plots like this or scatterplots.

Many machine learning algorithms are sensitive to the range and distribution of attribute values in the input data. Outliers in input data can skew and mislead the training process of machine learning algorithms resulting in longer training times and less accurate models.

A more robust way of statistically identifying outliers is by using the Z-Score.

The Z-score is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured. [*Source definition*](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/z-score/).

The idea behind Z-score is to describe any data point regarding their relationship with the Standard Deviation and Mean for the group of data points. Z-score is about finding the distribution of data where the mean is 0, and the standard deviation is 1, i.e., normal distribution.

```
z = np.abs(stats.zscore(boston_df))
print(z)
```

    [[0.41978194 0.28482986 1.2879095  ... 1.45900038 0.44105193 1.0755623 ]
     [0.41733926 0.48772236 0.59338101 ... 0.30309415 0.44105193 0.49243937]
     [0.41734159 0.48772236 0.59338101 ... 0.30309415 0.39642699 1.2087274 ]
     ...
     [0.41344658 0.48772236 0.11573841 ... 1.17646583 0.44105193 0.98304761]
     [0.40776407 0.48772236 0.11573841 ... 1.17646583 0.4032249  0.86530163]
     [0.41500016 0.48772236 0.11573841 ... 1.17646583 0.44105193 0.66905833]]


```
threshold = 3
## The first array contains the list of row numbers and the second array contains their respective column numbers.
print(np.where(z > 3))
```

    (array([ 55,  56,  57, 102, 141, 142, 152, 154, 155, 160, 162, 163, 199,
           200, 201, 202, 203, 204, 208, 209, 210, 211, 212, 216, 218, 219,
           220, 221, 222, 225, 234, 236, 256, 257, 262, 269, 273, 274, 276,
           277, 282, 283, 283, 284, 347, 351, 352, 353, 353, 354, 355, 356,
           357, 358, 363, 364, 364, 365, 367, 369, 370, 372, 373, 374, 374,
           380, 398, 404, 405, 406, 410, 410, 411, 412, 412, 414, 414, 415,
           416, 418, 418, 419, 423, 424, 425, 426, 427, 427, 429, 431, 436,
           437, 438, 445, 450, 454, 455, 456, 457, 466]), array([ 1,  1,  1, 11, 12,  3,  3,  3,  3,  3,  3,  3,  1,  1,  1,  1,  1,
            1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  5,  3,  3,  1,  5,
            5,  3,  3,  3,  3,  3,  3,  1,  3,  1,  1,  7,  7,  1,  7,  7,  7,
            3,  3,  3,  3,  3,  5,  5,  5,  3,  3,  3, 12,  5, 12,  0,  0,  0,
            0,  5,  0, 11, 11, 11, 12,  0, 12, 11, 11,  0, 11, 11, 11, 11, 11,
           11,  0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]))


You could use Z-Score and set its threshold to detect potential outliers in the data. With this, we can remove the outliers from our dataframe. For example:

```
print(boston_df.shape)
boston_df = boston_df[(np.abs(stats.zscore(boston_df)) < 3).all(axis=1)]
print(boston_df.shape)
```

    (506, 13)
    (415, 13)


For each column, first it computes the Z-score of each value in the column, relative to the column mean and standard deviation.
Then is takes the absolute of Z-score because the direction does not matter, only if it is below the threshold.
all(axis=1) ensures that for each row, all column satisfy the constraint.
Finally, result of this condition is used to index the dataframe.

## References 

* https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781784390150/2

* https://www.learndatasci.com/tutorials/data-science-statistics-using-python/

* https://www.datacamp.com/community/tutorials/demystifying-crucial-statistics-python


