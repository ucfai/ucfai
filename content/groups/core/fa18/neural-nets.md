---
title: Intro the Neural Nets, featuring PyTorch
linktitle: Intro the Neural Nets, featuring PyTorch

date: '2020-09-26T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 4

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

location: PSY 105
cover: ''

categories: [fa18]
tags: []
abstract: >-
  With some basic ideas in mind about how one might tackle a task, we'll now go and
  explore a Tensor framework (PyTorch) and build a Neural Network which can accurately
  classify handwritten digits, as well as articles of clothing.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-neural-nets").exists():
    DATA_DIR /= "ucfai-core-fa18-neural-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-neural-nets/data
    DATA_DIR = Path("data")
```
