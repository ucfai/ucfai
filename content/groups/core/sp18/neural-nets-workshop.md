---
title: 'Workshop: Building an Extensible ANN'
linktitle: 'Workshop: Building an Extensible ANN'

date: '2018-02-15T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 3

menu:
  core_sp18:
    parent: Spring 2018

authors: [ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: HEC 103
cover: ''

categories: [sp18]
tags: []
abstract: >-
  We'll take what we learned last week and actually write up a Neural Network to train
  on the MNIST dataset, to recognize hand-written digits with about 92% accuracy.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp18-neural-nets-workshop").exists():
    DATA_DIR /= "ucfai-core-sp18-neural-nets-workshop"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp18-neural-nets-workshop/data
    DATA_DIR = Path("data")
```
