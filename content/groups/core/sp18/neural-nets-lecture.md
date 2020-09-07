---
title: 'Lecture: An Intro to Neural Networks (ANNs)'
linktitle: 'Lecture: An Intro to Neural Networks (ANNs)'

date: '2018-02-08T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 2

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
tags: [neural networks, deep learning, MNIST]
abstract: >-
  Here, we'll dive, head first, into the nitty-gritty of Neural Networks,
  how they work, what Gradient Descent achieves for them, and how Neural
  Networks act on the feedback that Gradient Descent derives.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp18-neural-nets-lecture").exists():
    DATA_DIR /= "ucfai-core-sp18-neural-nets-lecture"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp18-neural-nets-lecture/data
    DATA_DIR = Path("data")
```
