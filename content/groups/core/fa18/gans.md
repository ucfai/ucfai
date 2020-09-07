---
title: Who Made this Face?
linktitle: Who Made this Face?

date: '2020-10-17T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 7

menu:
  core_fa18:
    parent: Fall 2018

authors: [thedibaccle]

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
  We're filling this out!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-gans").exists():
    DATA_DIR /= "ucfai-core-fa18-gans"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-gans/data
    DATA_DIR = Path("data")
```
