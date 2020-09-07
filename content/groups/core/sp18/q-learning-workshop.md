---
title: 'Workshop: Overcoming Car Troubles with Q-Learning'
linktitle: 'Workshop: Overcoming Car Troubles with Q-Learning'

date: '2018-04-19T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 11

menu:
  core_sp18:
    parent: Spring 2018

authors: []

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
  We're filling this out!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp18-q-learning-workshop").exists():
    DATA_DIR /= "ucfai-core-sp18-q-learning-workshop"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp18-q-learning-workshop/data
    DATA_DIR = Path("data")
```
