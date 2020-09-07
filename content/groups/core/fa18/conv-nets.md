---
title: Teaching Machines to Make Sense of Images
linktitle: Teaching Machines to Make Sense of Images

date: '2020-10-03T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 5

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
  Convolutional Neural Networks are at the forefront of Deep Learning and they enable
  machines to "see" much more effectively than they used to. So well, in fact, that
  they can tell what's in an image, or even place points onto them.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-conv-nets").exists():
    DATA_DIR /= "ucfai-core-fa18-conv-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-conv-nets/data
    DATA_DIR = Path("data")
```
