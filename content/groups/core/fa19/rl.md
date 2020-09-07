---
title: Training Machines to Learn From Experience
linktitle: Training Machines to Learn From Experience

date: '2019-11-06T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 8

menu:
  core_fa19:
    parent: Fall 2019

authors: [danielzgsilva, jarviseq, ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: https://kaggle.com/ucfaibot/core-fa19-rl
  colab: ''

papers: {}

location: MSB 359
cover: ''

categories: [fa19]
tags: [reinforcement learning, machine learning, planning, value iteration, deep reinforcement
    learning, neural networks, deep learning]
abstract: >-
  We all remember when DeepMind’s AlphaGo beat Lee Sedol, but what actually
  made the program powerful enough to outperform an international champion?
  In this lecture, we’ll dive into the mechanics of reinforcement learning
  and its applications.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-rl").exists():
    DATA_DIR /= "ucfai-core-fa19-rl"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-rl/data
    DATA_DIR = Path("data")
```
