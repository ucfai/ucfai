---
title: Time Series Analysis
linktitle: Time Series Analysis

date: '2020-03-04T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 7

menu:
  core_sp20:
    parent: Spring 2020

authors: [nspeer12]

urls:
  youtube: ''
  slides: ''
  github: https://github.com/ucfai/core/blob/master/sp20/03-04-time-series/time-series.ipynb
  kaggle: https://kaggle.com/ucfaibot/core-sp20-time-series
  colab: https://colab.research.google.com/github/ucfai/core/blob/master/sp20/03-04-time-series/time-series.ipynb

papers: {}

location: HPA1 112
cover: ''

categories: [sp20]
tags: [Time Series, Temporal Predictions, Coronavirus]
abstract: >-
  How can we infer on the past to predict the future? In this meeting
  we are going to be learning about time series data and its unique
  qualities. After we sharpen up our data science skills, we will be
  putting them to good use by analyzing and predicting the spread of the
  Coronavirus!

---

```
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
