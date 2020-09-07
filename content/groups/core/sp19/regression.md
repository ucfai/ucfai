---
title: Starting with the Basics, Regression
linktitle: Starting with the Basics, Regression

date: '2019-02-06T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 1

menu:
  core_sp19:
    parent: Spring 2019

authors: [jarviseq, causallycausal, ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: HEC 119
cover: ''

categories: [sp19]
tags: [regression, linear regression, logistic regression, statistics]
abstract: >-
  You always start with the basics, and with Data Science it's no different!
  We'll be getting our feet wet with some simple, but powerful, models and  demonstrate
  their power by applying them to real world data.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp19-regression").exists():
    DATA_DIR /= "ucfai-core-sp19-regression"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp19-regression/data
    DATA_DIR = Path("data")
```
