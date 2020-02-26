---
title: "Time Series Analysis"
linktitle: "Time Series Analysis"

date: "2020-03-04T17:30:00"
lastmod: "2020-03-04T17:30:00"

draft: false
toc: true
type: docs

weight: 7

menu:
  core_sp20:
    parent: Spring 2020
    weight: 7

authors: ["nspeer12", ]

urls:
  youtube: ""
  slides:  "https://docs.google.com/presentation/d/16Gp1QBEB9faVjxICC4Cpi8dZ-TTE1FZjr0gBjZ8Y_cU"
  github:  "https://github.com/ucfai/core/blob/master/sp20/03-04-time-series/time-series.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-sp20-time-series"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/sp20/03-04-time-series/time-series.ipynb"

location: ""
cover: "https://s31888.pcdn.co/wp-content/uploads/2018/07/index.jpg"

categories: ["sp20"]
tags: []
abstract: >-
  How can we infer on the past to predict the future? In this meeting, we're going to be uncovering time series data and learning how to solve problems like predicting the stock market or the weather.
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
