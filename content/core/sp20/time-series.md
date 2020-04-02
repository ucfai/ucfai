---
title: "Time Series Analysis"
linktitle: "Time Series Analysis"

date: "2020-03-04T17:30:00"
lastmod: "2020-04-02T19:11:26.411430836"

draft: false
toc: true
type: docs

weight: 7

menu:
  core_sp20:
    parent: "Spring 2020"
    weight: "7"

authors: ["nspeer12", ]

urls:
  youtube: ""
  slides:  "https://docs.google.com/presentation/d/16Gp1QBEB9faVjxICC4Cpi8dZ-TTE1FZjr0gBjZ8Y_cU"
  github:  ""
  kaggle:  "https://kaggle.com/ucfaibot/core-sp20-time-series"
  colab:   ""

location: "HPA1 112"
cover: ""

categories: ["sp20"]
tags: ["Time Series", "Temporal Predictions", ]
abstract: "How can we infer on the past to predict the future? In this meeting we are going to be learning about time series data and its unique qualities. After we sharpen up our data science skills, we will be putting them to good use by analyzing and predicting the spread of the Coronavirus!"
---

```python
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-time-series").exists():
    DATA_DIR /= "ucfai-core-sp20-time-series"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-time-series/data
    DATA_DIR = Path("data")
```

# Time Series Data
### Looking to the past to infer  the future

Time series data is a special kind of data that we will be exploring in this notebook. Time series data has a few important properties such as:
- it's sequential
- time is the independent variable
- observed at discrete intervals, such as 5 minutes or 1 hour

Let's dive into some examples of time series data and learn how we can observe, understand, and predict this unique kind of data.


```python
# import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet

import time
import datetime
import os
import sys

# matplotlib configuration
%matplotlib inline
plt.style.use('ggplot')
```

# What is Time Series Data
Time series data consists of data points recorded at discrete intervals over time. This kind of data occurs everywhere in our lives, a few examples include:
- Stock market data
- Ocean tides
- Commute times
- Weather and outdoor temperatures
- Business sales or revenue
- Disney World line wait times    
And so many more examples. Time series data is so common, it becomes increasingly important to learn how to analyze it. In this notebook, we'll be taking on the challenge of using the past to predict the future.

# Dow Jones Industrial Average
Abbreviated as DJI and commonly refered to as the "Dow", this financial index monitors the prices of 30 large US companies such as Apple, Home Depot, IBM and Disney. The Dow is a good indicator of how the US stock markets are performing overall. The Dow dates back to 1885 and was founded by a Wall Street Journal reporter, Charles Dow, and his statitician, Edward Jones. We're going to take a look at the Dow and use it as an example to learn some important properties of time series data.

## Columns
- Date
    Our data dates back to 1985. It includes every day the stock market open, which is normal week days. This excludes weekends and holidays.
- Open
    The price our data opens 9:30 AM New York Time.
- High
    The highest price the Dow reaches during the day.
- Low
    The lowest price of the Dow during each day
- Close
    The price of the Dow when the market closes at 4:00 New York Time.
- Adj Close
    The closing price adjusted for dividends and stock splits.
- Volume
    The number of transactions on the market during each day.



```python
# load data for Dow Jones Index
df = pd.read_csv(DATA_DIR / 'DJI.csv', delimiter=',')
df.head()
```


```python
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df['Open'], color='green', label='Dow Jones Opening Price')
ax.set_ylabel('Price')
ax.set_xlabel('Days')
ax.legend()
plt.show()

```


```python
# Plot the last 100 days of Dow opening prices with date labels
fig, axs = plt.subplots(2, figsize=(20,10))

# To plot dates is a somewhat expensive operation, so we're only going to look at the past m days
m = 100
dow = df[:m]

# plot the Dow with dates on the first axes
axs[0].plot(dow['Date'], dow['Open'], color='green', label='Dow Jones Opening Price')
axs[0].set_ylabel('Price')
axs[0].set_xlabel('Days')

# plot volume using a bar graph on the second axes
axs[1].bar(dow['Date'], dow['Volume'], color='blue', alpha=0.5, label='Dow Jones Volume')

# show every nth x tick label
i = 5
plt.xticks([x for x in dow['Date']][::i])


# rotate date labels
fig.autofmt_xdate()
axs[0].format_xdata = mdates.DateFormatter('%Y-%m-%d')
axs[1].format_xdata = mdates.DateFormatter('%Y-%m-%d')

# show legend
axs[0].legend()
axs[1].legend()
plt.show()

```

# Moving Averages
The first analysis tool that we'll be looking at is called a moving average. We're going to be looking at a subset of our data across a given range, and then be taking the average of all of the observations within that range. To do this, we'll loop through our entire dataset, then loop through each sliding window and calculate the average of that window.


```python
# calculate 5 day moving average
MA5 = []

# start at the fifth index and loop through all data
for i in range(5, len(df)):
    # sum previous 5 data points and average them
    sum = 0
    for j in range(i-5, i):
        sum += df['Adj Close'][j]
    # add the average to the list
    MA5.append(sum / 5)
    

# drop rows 0 - 5
df = df[:-5]

# append the moving average to the data frame
df['MA5'] = MA5

```


```python
# Let's take a look at the 5 day moving average we just calculated

fig, ax = plt.subplots(figsize=(20,10))

# zoom in a bit
plt.axis([2300, 2500, 3600, 4000])
ax.plot(df['Open'], color='blue', label='Dow Jones Open')
ax.plot(df['MA5'], color="green", label='5 Day MA')
ax.legend()
plt.show()
```


```python
# define a list of averages to calculate
averages = [10, 15, 50, 100, 200, 500]

# expand our data to several different moving averages
for avg in averages:
    # easier way to calculate moving averages than using for loop method
    df['MA' + str(avg)] = df['Adj Close'].rolling(window=avg).mean()
```


```python
# view the new moving averages in our data frame
df.head()
```


```python
# drop rows with null values, which is up to our largest moving average
df = df.dropna()
df.head()
```


```python
fig, ax = plt.subplots(figsize=(20,10))

# adjust the view
plt.axis([5000, 8000, 5000, 20000])

ax.plot(df['Adj Close'], c='green', label='Dow Adj Close')

for avg in averages:
    name = 'MA' + str(avg)
    ax.plot(df[name], label=name)

ax.legend()
plt.show()
```

# Exponential Moving Averages
Exponential moving averages work similarly to simple moving averages, except they add greater weight to more recent values. This is a good way to see what our trends look like with some "memory" of the past, while still being reactive to recent trends and sudden movements.


```python
# EWM: Exponential Weighted Functions

df['EMA5'] = df['Open'].ewm(span=5, adjust=False).mean()

fig, ax = plt.subplots(figsize=(20,10))
plt.axis([6100, 6200, 7500, 10000])

ax.plot(df['Open'], c='green', label='Dow Open')
ax.plot(df['EMA5'], c='blue', label='5 Day EMA')
ax.legend()
plt.show()
```


```python
# compare EMA to SMA
fig, ax = plt.subplots(figsize=(20,10))
plt.axis([6100, 6200, 7500, 10000])
ax.plot(df['Open'], c='green', label='Dow Open')
ax.plot(df['EMA5'], c='blue', label='5 Day EMA')
ax.plot(df['MA5'], c='orange', label='5 Day SMA')
ax.legend()
plt.show()
```

# Decomposing Data
For any given time series data set, there are a lot of factors that go into each value. For the stock market, some of it is seemingly random, yet many top traders are able to extract useful information in order to better their odds. Decomposing data is the method of extracting each working piece of time series data.

## Seasonal Decomposition

### Additive formula: y(t) = S(t) + T(t) + R(t)
#### y = seaonality + trend + remainder

### Multiplicative formula: y(t) = S(t) * T(t) * R(t)
#### y = seaonality * trend * remainder

### Seasonality
Imagine data being subjected to cyclical forces, like tides of the ocean or seasons of the year

#### Examples
- In most locations, the temperature is going to cooler in winter in warmer in summer
- TSA wait times are going to be longer during holiday seasons.
- Historically stocks perform the worst in september.

### Trend
What sort of momentum does the data have?

#### Examples
- Moore's Law
- Increasing global population
- Global poverty decreasing

### Remainder
Randomness, noise

#### Examples
- microsecond flucuations in the price of the stock
- seemingly random warm day in winter

## Air Travel Passenger Dataset
This data set shows the number of monthly airline travelers from 1949 to 1960. Maybe you are one of the many people that have experienced the long wait times and crowded planes during the holidays. The data shows this huge uptick that occurs during each year, seemingly like clockwork. We're going to be applying seasonal decomposition to this dataset and learn how we can find each of the working pieces of this time series data.


```python
df = pd.read_csv(DATA_DIR / 'AirPassengers.csv', delimiter=',')
df.head()
```


```python
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(df['Month'], df['Passengers'], color='blue', label='Passenger Volume')

# fancy way to show every nth tick label
n = 12
plt.xticks([x for x in df['Month']][::n])
    
fig.autofmt_xdate()
ax.format_xdata = mdates.DateFormatter('%Y')
ax.legend()
plt.show()
```

## Detrending Data
Many factors influence the measurements we observe in time series data. For instance, in the 20th century, air travel was an emergent industry that saw a steady increase. We can estimate this general trend of growth and then subtract it from each point to greater understand the other trends that have influence. Once we detrend our data, we can isolate and observe the seasonality in our data.

### Setting a Base Line
This step is going to be highly dependent on the data set being detrended. In most cases that show a generally increasing or decreasing trend, there will be a sort of "Goldilocks" value that will work well. Try to establish some sort of baseline that is around the low end of the data and covers most of the are underneath the data. In this example, we'll be using a 48 month moving average that creates a nice baseline to our data.

#### Why 48?
We're using a 48 month moving average because it is a multiple of 12. Since we are looking at seasonality, we want to be left with perfect 12 month periods.





```python
# Use the 48 Month moving average as the trend line
# note: we're using 48 here for a couple reasons
 
df['MA48'] = df['Passengers'].rolling(window=48).mean()

# plot the the trend line
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df['Month'], df['Passengers'], color='blue', label='Passenger Data')
ax.plot(df['MA48'], color='green', label='48 Month MA')

n = 6
plt.xticks([x for x in df['Month']][::n])
fig.autofmt_xdate()
ax.legend()
plt.show()
```

## Let's use some algebra
### S(t) + R(t) = y(t) - T(t)
By subtracting our trend line from our seasonal data, we'll be left with our seasonal data, plus some randomness.


```python
# subtract the trend line from the passenger data
df['detrend'] = df['Passengers'] - df['MA48']

# plot detrended data
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df['detrend'], color='orange', label='detrended data')
ax.set_xlabel('Month')
ax.set_ylabel('Seasonal Fluctuation')
ax.legend()
plt.show()
```

### Find Seasonality
In the graph above, it's obvious that our data follows a cyclical pattern. In this next bit of code, what we're going to do is overlay each year on top of each other to show the recurrent pattern.


```python
# create a new dataframe to hold our annual detrended data
annual = pd.DataFrame()

# iterate and chop our data into 12 month periods
i = 0
while (i * 12) < (len(df['detrend']) - 12):
    x = i * 12
    annual[i] = df['detrend'].iloc[x:x+12].reset_index(drop=True)
    i += 1

fig, ax = plt.subplots(figsize=(15,5))
fig.suptitle('Annual Seasonal Data')
for i in range(0, 11):
    ax.plot(annual[i], label='year '+str(i))


# plot each month out on a timeline
ax.set_xlabel('Month')
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.xticks(ticks=[x for x in range(12)], labels=months)
ax.set_ylabel('Seasonal Fluctuation')


ax.legend()
plt.show()
```

## Seasonal Mean
To identify what our seasonal fluctuation looks like on average, we'll be taking the average of each year. Doing this will allow us to estimate what our seasonal difference may look like in the future.


```python
# Find the average seasonal data by taking the mean from each year
annual['Seasonal Mean'] = annual.mean(axis=1)
# blue line represents seasonal trend
fig, ax = plt.subplots(figsize=(15,5))
for i in range(0, 11):
    ax.plot(annual[i], label='year '+ str(i), alpha=0.3)


ax.set_title('Detrended Seasonal Data')
ax.plot(annual['Seasonal Mean'], color='blue', label='Seasonal Mean')
ax.legend()
ax.set_ylabel('Seasonal Fluctuation')
ax.set_xlabel('Month')
plt.xticks(ticks=[x for x in range(12)], labels=months)
plt.show()
```

## Multiplicative Formula
In the cell below, we're going to implement a multiplicative formula using the statsmodels seasonal decomposition library. This library will expidite a lot of the previous work we just did.


```python
# implemented with statsmodels
# not as much fun, but a whole lot easier
from statsmodels.tsa.seasonal import seasonal_decompose

# strips time
def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m')

data = pd.read_csv(DATA_DIR / 'AirPassengers.csv', delimiter=',', index_col=['Month'], parse_dates=True, squeeze=True, date_parser=parser)

# try toggling the model between additive and multiplicative!
res = seasonal_decompose(data, model='multiplicative')
pd.plotting.register_matplotlib_converters() # this fixes an error when plotting
res.plot()
plt.show()
```

# ARIMA
ARIMA stands for Autoregressive Integrated Moving Average

### AR: Autoregression
Model that uses the dependent relationship between an observation and some number of lagged observations.

### I: Integrated
Integrated meaning the decomposition of data using differencing, or subtracting previous values from each other, to make the data stationary.

### MA: Moving Average
Average of previous time series observations over a given period.

Each of these components are explicitly specified in the model as a parameter. A standard notation is used of ARIMA(p,d,q) where the parameters are substituted with integer values to quickly indicate the specific ARIMA model being used.

The parameters of the ARIMA model are defined as follows:

#### p: The number of lag observations included in the model, also called the lag order.
#### d: The number of times that the raw observations are differenced, also called the degree of differencing.
#### q: The size of the moving average window, also called the order of moving average.

source, and for more information on ARIMA: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

## Autocorrelation
Autocorrelation is the similarity between observed time series observations. By analyzing autocorrelation, we can try to pick up signals such as seasonality and fluctuation.
By plotting autocorrelation, we can see which time lag window has the highest correlation. When we make the ARIMA model, we want to pick a lag observation window that has a high correlation. We can adjust this window using our "p" parameter


```python
from pandas.plotting import autocorrelation_plot

# plot autocorrelation
autocorrelation_plot(data)
plt.show()
```

## ARIMA With Regression
Let's take a look at an ARIMA model that uses regression on the lag to fit to our training.
https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html


```python
# Isolate values
X = df['Passengers'].values

# manual test-train split
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]

# create a new list for our training data. We'll expand this list once we "see" testing values
history = [x for x in train]

# list to store just our predictions
predictions = []
```


```python
# Make predictions at each point in our testing data
for t in range(len(test)):
    # since our training data is dynamically growing, we make and train a new model each iteration
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    
    # make a prediction
    output = model_fit.forecast()
    
    # forecast method returns a tuple, take the first value
    yhat = output[0]
    
    # add prediction to our list
    predictions.append(yhat)
    
    # add our testing value back into our history
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(test, color='blue', label='test data')
ax.plot(predictions, color='red', label='predictions')
ax.legend()
plt.show()
```

## Prophet Model

### Seasonal Decomposition + Holidays



```python
model = Prophet()
print(df)
# create a copy of the data frame and rename columns so they work nicely with fbprophet library
df = df.rename(columns={'Month':'ds', 'Passengers': 'y'})

# fit the model to our data frame
model.fit(df)

# make a new dataframe for the forecast
future = model.make_future_dataframe(periods=12, freq='M')

# make a forecast
forecast = model.predict(future)

# plot forecast
fig = model.plot(forecast)
```

# Corona Virus Dataset
The 2019 Novel Coronavirus is a global pandemic that is spreading quickly throughout the world. You are tasked with analyzing and forcasting the spread of this virus. Use your data science and CS skills to help fight the outbreak!


```python
# Take a look at the data set below, it's totally not in the right format that we want
df = pd.read_csv(DATA_DIR / 'covid_19_data.csv', delimiter=',')
print(df.head())
```


```python
# TODO: sum up confirmed cases and plot a graph over time

# create a dictionary (hash map) to store the total observations on a given date
keys = [date for date in df['ObservationDate'].unique()]
casesMap = dict.fromkeys(keys, 0)

# iterate through each row and add up the number of confirmed cases to a hash map
# TODO: Sum up the total confirmed cases in the hash map
for index, row in df.iterrows():
    ### BEGIN SOLUTION
    date = row['ObservationDate']
    casesMap[date] += row['Confirmed']
    ### END SOLUTION

```


```python
# We have each of the totals in a dict object, now we have to put them back into time series data

# create a new dataframe
df = pd.DataFrame()

# add in a row for each of our dates
df['Date'] = keys
print(df.head())

# add in a new row for confirmed cases at each date
df['Confirmed'] = [casesMap[x] for x in keys]
df = df.dropna()

print('\n Confirmed Cases Dataframe')
print(df.head())
```


```python
# TODO: Plot the number of confirmed cases over time

### BEGIN SOLUTION
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df['Date'], df['Confirmed'], label='Confirmed Cases', color='red')
fig.autofmt_xdate()
ax.fmt_xdata = mdates.DateFormatter('%m/%d/%Y')
ax.legend();
plt.show()
### END SOLUTION
```


```python
# TODO: Calculate the 5 Day Moving Average and plot it

### BEGIN SOLUTION

# calculate moving average
df['MA5'] = df['Confirmed'].rolling(window=5).mean()

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df['Date'], df['Confirmed'])
ax.plot(df['Date'], df['MA5'])
fig.autofmt_xdate()
ax.fmt_xdata = mdates.DateFormatter('%m/%d/%Y')
plt.show()
### END SOLUTION
```


```python
# TODO: Apply linear regression to the graph
model = Prophet()
    
# create a copy of the data frame and rename columns so they work nicely with fbprophet
df = df.rename(columns={'Date':'ds', 'Confirmed': 'y'})

model.fit(df)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# changing format of the dates
forecast['ds'] = [x.strftime('%m/%d/%Y') for x in forecast['ds']]
```


```python
fig, ax = plt.subplots(figsize=(20,10))

# plot lower and upper bound of forecast and compare to real data
dates = [d for d in forecast['ds']]
ax.plot(df['ds'], df['y'], color='green', label='Real Confirmed Cases')
ax.plot(dates, forecast['yhat_lower'], color='blue', label='yhat lower')
ax.plot(dates, forecast['yhat_upper'], color='orange', label='yhat upper')

# fancy way to show every nth tick label
n = 7
plt.xticks([x for x in forecast['ds']][::n])
    
fig.autofmt_xdate()
ax.format_xdata = mdates.DateFormatter('%m-%d-%Y')
ax.legend()
plt.show()
```


```python
# TODO: Use the ARIMA model to forecast the spread of Coronavirus

### BEGIN SOLUTION
# Make predictions at each point in our testing data
# Isolate values
X = df['y'].values

# manual test-train split
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]

# create a new list for our training data. We'll expand this list once we "see" testing values
history = [x for x in train]

# list to store just our predictions
predictions = []

for t in range(len(test)):
    # since our training data is dynamically growing, we make and train a new model each iteration
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    
    # make a prediction
    output = model_fit.forecast()
    
    # forecast method returns a tuple, take the first value
    yhat = output[0]
    
    # add prediction to our list
    predictions.append(yhat)
    
    # add our testing value back into our history
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(test, color='blue', label='test data')
ax.plot(predictions, color='red', label='predictions')
ax.legend()
plt.show()

### END SOLUTION
```


```python
# TODO: Try to forecast 30 days into the future

### BEGIN SOLUTION
forecast = model_fit.forecast(steps=30)[0]
fig, ax = plt.subplots()
ax.plot(forecast, label='ARIMA Coronavirus Forecast')
ax.set_xlabel('Days into the Future')
ax.set_ylabel('Estimated Confirmed Cases')
plt.show()
### END SOLUTION
```

# Challenge: Show the increase of confirmed cases by location over time


```python

```
