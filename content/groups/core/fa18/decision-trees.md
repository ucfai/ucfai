---
title: Decision Trees
linktitle: Decision Trees

date: '2020-10-31T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 9

menu:
  core_fa18:
    parent: Fall 2018

authors: [chaskane]

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
  Sometimes the algorithms we use to predict the future can be difficult to interpret
  and trust. A Decision Tree is a learning algorithm that does a half decent job at
  prediction, but, more importantly, is very easy to understand and interpret. No
  black boxes here... until we start talking about Random Forests.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-decision-trees").exists():
    DATA_DIR /= "ucfai-core-fa18-decision-trees"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-decision-trees/data
    DATA_DIR = Path("data")
```
