---
title: Support Vector Machines
linktitle: Support Vector Machines

date: '2020-11-07T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 10

menu:
  core_fa18:
    parent: Fall 2018

authors: [ahl98]

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
  Support Vector Machines are a simple and powerful classification algorithm that
  perform well in nearly every situation. They're commonly used in image recognition,
  face detection, bioinformatics, handwriting recognition, and text categorization.
  The math behind it is pretty cool, as it relies upon embedding data into higher
  dimensional space to create linear divisions between categories. SVMs are a great
  resource to add to your data science toolkit, as they're relatively simple to understand
  and are also one of the best classification algorithms that do not involve neural
  networks.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-svms").exists():
    DATA_DIR /= "ucfai-core-fa18-svms"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-svms/data
    DATA_DIR = Path("data")
```
