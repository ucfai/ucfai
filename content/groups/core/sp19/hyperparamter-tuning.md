---
title: What Makes Deep Learning More of an Art Than a Science?
linktitle: What Makes Deep Learning More of an Art Than a Science?

date: '2019-03-06T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 5

menu:
  core_sp19:
    parent: Spring 2019

authors: [ionlights]

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
tags: [neural networks, convolutional networks, recurrent networks, hyperparamter
    tuning, gridsearch]
abstract: >-
  Some of the hardest aspects of Machine Learning are the details. Almost every
  algorithm we use is sensitive to "hyperparameters" which affect the initialization,
  optimization speed, and even the possibility of becoming accurate. We'll cover
  the
  general heuristics you can use to figure out what hyperparameters to use, how
  to  find the optimal ones, what you can do to make models more resilient, and
  the like.
  This workshop will be pretty "down-in-the-weeds" but will give you a better
  intuition
  about Machine Learning and its shortcomings.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp19-hyperparamter-tuning").exists():
    DATA_DIR /= "ucfai-core-sp19-hyperparamter-tuning"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp19-hyperparamter-tuning/data
    DATA_DIR = Path("data")
```
