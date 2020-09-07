---
title: Machine Learning Applications
linktitle: Machine Learning Applications

date: '2019-10-09T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 4

menu:
  core_fa19:
    parent: Fall 2019

authors: [jarviseq]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: MSB 359
cover: ''

categories: [fa19]
tags: [neural networks, deep learning, applications, random forests, svms]
abstract: >-
  You know what they are, but "how do?" In this meeting, we let you loose on a    dataset
  to help you apply your newly developed or honed data science skills.  Along the
  way, we go over the importance of visulisations and why it is  important to be able
  to pick apart a dataset. 
---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-ml-applications").exists():
    DATA_DIR /= "ucfai-core-fa19-ml-applications"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-ml-applications/data
    DATA_DIR = Path("data")
```
