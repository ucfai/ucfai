---
title: How We Can Give Our Computers Eyes and Ears
linktitle: How We Can Give Our Computers Eyes and Ears

date: '2020-02-12T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 4

menu:
  core_sp20:
    parent: Spring 2020

authors: [danielzgsilva]

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
tags: [Convolutional Networks, Image Processing, Feature Extraction]
abstract: >-
  Ever wonder how Facebook tells you which friends to tag in your photos, or
  how Siri can even understand your request? In this meeting we'll dive into
  convolutional neural networks and give you all the tools to build smart
  systems such as these. Join us in learning how we can grant our computers
  the gifts of hearing and sight!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-conv-nets").exists():
    DATA_DIR /= "ucfai-core-sp20-conv-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-conv-nets/data
    DATA_DIR = Path("data")
```
