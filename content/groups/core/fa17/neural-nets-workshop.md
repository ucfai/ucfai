---
title: 'Workshop: An Intro to Neural Nets'
linktitle: 'Workshop: An Intro to Neural Nets'

date: '2017-10-04T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 2

menu:
  core_fa17:
    parent: Fall 2017

authors: [ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: VAB 113
cover: ''

categories: [fa17]
tags: [neural networks, deep learning, machine learning, MNIST, TensorFlow, numpy]
abstract: >-
  UPDATE: We've partnered with TechKnights to throw a lecture+workshop combo during
  KnightHacks!

  To finish Unit 0 for the Fall series, we're following up our lecture last week
  with a workshop.

  Here, we'll build a neural network to classify hand-written digits using a popular
  dataset, MNIST, with some help from Google's Tensorflow library.

  ***Everything will be provided in a self-contained environment for you but you
  will need to come prepared with the below requirements before the workshop begins.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa17-neural-nets-workshop").exists():
    DATA_DIR /= "ucfai-core-fa17-neural-nets-workshop"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa17-neural-nets-workshop/data
    DATA_DIR = Path("data")
```
