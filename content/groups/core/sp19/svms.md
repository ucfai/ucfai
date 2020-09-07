---
title: Support Vector Machines
linktitle: Support Vector Machines

date: '2019-04-03T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 8

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
tags: [support vector machines, traditional approaches]
abstract: >-
  Support Vector Machines were among the most highly used ML algorithms before
  Neural Nets came back into the foreground. Unlike Neural Nets, SVMs can explain
  themselves quite well and allow us to use these ML mdels in fields like medicine,
  finance, and the like – where regulations require that we can inquire about
  our
  models.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp19-svms").exists():
    DATA_DIR /= "ucfai-core-sp19-svms"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp19-svms/data
    DATA_DIR = Path("data")
```
