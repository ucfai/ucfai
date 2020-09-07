---
title: Starting with the Basics, Regression
linktitle: Starting with the Basics, Regression

date: '2020-01-22T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 1

menu:
  core_sp20:
    parent: Spring 2020

authors: [JarvisEQ, bb912]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: HPA1 112
cover: ''

categories: [sp20]
tags: [regression, linear regression, logistic regression]
abstract: >-
  You always start with the basics, and with AI it's no different! We'll be
  getting our feet wet with some simple, but powerful, models and
  demonstrate their power by applying them to real world data.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-regression").exists():
    DATA_DIR /= "ucfai-core-sp20-regression"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-regression/data
    DATA_DIR = Path("data")
```
