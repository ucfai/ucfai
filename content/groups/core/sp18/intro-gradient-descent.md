---
title: Welcome back to SIGAI, featuring Gradient Descent
linktitle: Welcome back to SIGAI, featuring Gradient Descent

date: '2018-02-01T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 1

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
tags: [gradient descent, optimization, machine learning]
abstract: >-
  Welcome back to SIGAI! We'll be covering some administrative needs – like how we're
  doing lectures/workshops and what we expect of coordinators, since we'll have elections
  in March. Then we'll go over some math to check everyone's background so you're
  uber prepared for next week! Once we've covered that, we'll go over Gradient Descent
  and get a rough idea of how it works – this is integral to almost all our content
  this semester.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp18-intro-gradient-descent").exists():
    DATA_DIR /= "ucfai-core-sp18-intro-gradient-descent"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp18-intro-gradient-descent/data
    DATA_DIR = Path("data")
```
