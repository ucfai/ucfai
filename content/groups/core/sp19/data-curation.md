---
title: Cleaning and Manipulation a Dataset with Python
linktitle: Cleaning and Manipulation a Dataset with Python

date: '2019-03-20T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 6

menu:
  core_sp19:
    parent: Spring 2019

authors: [danielzgsilva, ionlights]

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
tags: [data science, data curation, data cleaning, data manipulation, data engineering]
abstract: >-
  In the fields of Data Science and Artificial Intelligence, your models and  analyses
  will only be as good as the data behind them. Unfortunately, you
  will find that the majority of datasets you encounter will be filled with  missing,
  malformed, or erroneous data. Thankfully, Python provides a number  of handy
  libraries to help you clean and manipulate your data into a usable
  state. In today's lecture, we will leverage these Python libraries to turn
  a messy dataset into a gold mine of value!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp19-data-curation").exists():
    DATA_DIR /= "ucfai-core-sp19-data-curation"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp19-data-curation/data
    DATA_DIR = Path("data")
```
