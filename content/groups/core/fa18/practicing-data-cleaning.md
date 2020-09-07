---
title: Practice Makes Permanent, but Data's Messy
linktitle: Practice Makes Permanent, but Data's Messy

date: '2020-11-14T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 11

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
  We're filling this out!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-practicing-data-cleaning").exists():
    DATA_DIR /= "ucfai-core-fa18-practicing-data-cleaning"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-practicing-data-cleaning/data
    DATA_DIR = Path("data")
```
