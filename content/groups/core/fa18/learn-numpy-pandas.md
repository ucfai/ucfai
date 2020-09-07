---
title: Intro to Data Analysis with Pandas & Numpy
linktitle: Intro to Data Analysis with Pandas & Numpy

date: '2020-09-12T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 2

menu:
  core_fa18:
    parent: Fall 2018

authors: [ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: BA1 122
cover: ''

categories: [fa18]
tags: []
abstract: >-
  Data is arguably more important than the algorithms we'll be learning
  this semester - and that data almost always needs to be curated and
  finagled to really develop an understanding of what the data is trying
  to tell you.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-learn-numpy-pandas").exists():
    DATA_DIR /= "ucfai-core-fa18-learn-numpy-pandas"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-learn-numpy-pandas/data
    DATA_DIR = Path("data")
```
