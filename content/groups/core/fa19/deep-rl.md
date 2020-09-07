---
title: Learning by Doing, This Time with Neural Networks
linktitle: Learning by Doing, This Time with Neural Networks

date: '2019-11-13T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 9

menu:
  core_fa19:
    parent: Fall 2019

authors: [ahkerrigan]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: https://kaggle.com/ucfaibot/core-fa19-deep-rl
  colab: ''

papers: {}

location: MSB 359
cover: ''

categories: [fa19]
tags: [machine learning, deep learning, reinforcement learning, deep reinforcement
    learning, neural networks, policy optimization, DoTA]
abstract: >-
  It's easy enough to navigate a 16x16 maze with tables and some dynamic
  programming, but how exactly do we extend that to play video games with
  millions of pixels as input, or board games like Go with more states than
  particals in the observable universe? The answer, as it often is, is deep
  reinforcement learning.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-deep-rl").exists():
    DATA_DIR /= "ucfai-core-fa19-deep-rl"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-deep-rl/data
    DATA_DIR = Path("data")
```
