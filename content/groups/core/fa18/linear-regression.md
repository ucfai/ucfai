---
title: Let Data Speak Using Regression & Plots
linktitle: Let Data Speak Using Regression & Plots

date: '2020-09-19T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 3

menu:
  core_fa18:
    parent: Fall 2018

authors: [ahl98]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: PSY 105
cover: ''

categories: [fa18]
tags: []
abstract: >-
  Neural Networks are all the rage, nowadays, but simpler models are
  always great places to start! We'll cover how to do Linear/Logistic
  Regression as well as preparing data for such a function to work.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-linear-regression").exists():
    DATA_DIR /= "ucfai-core-fa18-linear-regression"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-linear-regression/data
    DATA_DIR = Path("data")
```
