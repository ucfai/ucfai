---
title: How We Give Our Computers Eyes and Eyes
linktitle: How We Give Our Computers Eyes and Eyes

date: '2019-10-16T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 5

menu:
  core_fa19:
    parent: Fall 2019

authors: [danielzgsilva, brandons209]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: MSB 359
cover: ''

categories: [fa19]
tags: [computer vision, CNNs, convolutional networks, deep learning, neural networks]
abstract: >-
  Ever wonder how Facebook tells you which friends to tag in your photos,
  or how Siri can even understand your request? In this meeting we'll dive
  into convolutional neural networks and give you all the tools to build
  smart systems such as these. Join us in learning how we can grant our  computers
  the gifts of hearing and sight!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-conv-nets").exists():
    DATA_DIR /= "ucfai-core-fa19-conv-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-conv-nets/data
    DATA_DIR = Path("data")
```
