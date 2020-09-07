---
title: A Walk Through the Random Forest
linktitle: A Walk Through the Random Forest

date: '2020-01-29T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 2

menu:
  core_sp20:
    parent: Spring 2020

authors: [JarvisEQ, nspeer12]

urls:
  youtube: ''
  slides: ''
  github: https://github.com/ucfai/core/blob/master/sp20/01-29-rf-svm/rf-svm.ipynb
  kaggle: https://kaggle.com/ucfaibot/core-sp20-rf-svm
  colab: https://colab.research.google.com/github/ucfai/core/blob/master/sp20/01-29-rf-svm/rf-svm.ipynb

papers: {}

location: HPA1 112
cover: ''

categories: [sp20]
tags: [Random Forests, SVMs, Nearest Neighbors]
abstract: >-
  In this lecture, we explore powerful yet lightweight models that are often
  overlooked. We will see the power of combining multiple simple models
  together and how they can yield amazing results. You won't believe how
  easy it is to classify with just a line!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-rf-svm").exists():
    DATA_DIR /= "ucfai-core-sp20-rf-svm"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-rf-svm/data
    DATA_DIR = Path("data")
```
