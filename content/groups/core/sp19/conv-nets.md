---
title: How Computers Can see and Other Ways Machines Can Think
linktitle: How Computers Can see and Other Ways Machines Can Think

date: '2019-02-20T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 3

menu:
  core_sp19:
    parent: Spring 2019

authors: [irene-l-tanner, brandons209, ionlights]

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
tags: [neural networks, convolutional networks, deep learning, computer vision]
abstract: >-
  Ever wonder how Facebook can tell you which friends to tag in your photos
  or how Google automatically makes collages and animations for you? This
  lecture is all about that: We'll teach you the basics of computer vision  using
  convolutional neural networks so you can make your own algorithm to
  automatically analyze your visual data!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp19-conv-nets").exists():
    DATA_DIR /= "ucfai-core-sp19-conv-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp19-conv-nets/data
    DATA_DIR = Path("data")
```
