---
title: Machines that Write as Well as Shakespeare
linktitle: Machines that Write as Well as Shakespeare

date: '2020-10-10T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 6

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
  Fully Connect Neural Networks and Convolutional Neural Networks are absolutely wonderful,
  but they miss out on one key component of our world, time. This week we'll look
  at Networks which also model time as part of their inputs – because of this, they'll
  be able to write nearly as well as Shakespeare! 😃

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-recurrent-nets").exists():
    DATA_DIR /= "ucfai-core-fa18-recurrent-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-recurrent-nets/data
    DATA_DIR = Path("data")
```
