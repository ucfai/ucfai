---
title: 'Lecture: Deconvoluting Convolutional Neural Networks'
linktitle: 'Lecture: Deconvoluting Convolutional Neural Networks'

date: '2018-02-22T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 4

menu:
  core_sp18:
    parent: Spring 2018

authors: [dibaccory]

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
tags: [convolutional networks, deep learning, machine learning, computer vision]
abstract: >-
  We're filling this out!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp18-conv-nets-lecture").exists():
    DATA_DIR /= "ucfai-core-sp18-conv-nets-lecture"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp18-conv-nets-lecture/data
    DATA_DIR = Path("data")
```
