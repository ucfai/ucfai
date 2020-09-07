---
title: A Walk Through the Random Forest
linktitle: A Walk Through the Random Forest

date: '2019-03-27T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 7

menu:
  core_sp19:
    parent: Spring 2019

authors: [jarviseq, ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: HEC 119
cover: ''

categories: [sp19]
tags: [random forests, boosting, statistics]
abstract: >-
  Neural Nets are not the end all be all of Machine Learning. In this lecture,  we
  will see how a decision tree works, and see how powerful a collection of  them
  can be. From there, we will see how to utilize Random Forests to do digit  recognition.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp19-random-forests").exists():
    DATA_DIR /= "ucfai-core-sp19-random-forests"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp19-random-forests/data
    DATA_DIR = Path("data")
```
