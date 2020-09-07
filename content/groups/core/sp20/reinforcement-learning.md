---
title: Training Machines to Learn From Experience
linktitle: Training Machines to Learn From Experience

date: '2020-04-01T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 9

menu:
  core_sp20:
    parent: Spring 2020

authors: [danielzgsilva, JarvisEQ, ionlights]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers: {}

location: HPA1 112
cover: ''

categories: [sp20]
tags: [Reinforcement Learning, Q-learning, OpenAI Gym]
abstract: >-
  We all remember when DeepMind’s AlphaGo beat Lee Sedol, but what actually
  made the program powerful enough to outperform an international champion?
  In this lecture, we’ll dive into the mechanics of reinforcement learning
  and its applications.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-reinforcement-learning").exists():
    DATA_DIR /= "ucfai-core-sp20-reinforcement-learning"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-reinforcement-learning/data
    DATA_DIR = Path("data")
```
