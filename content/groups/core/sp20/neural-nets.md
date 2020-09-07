---
title: Introduction to Neural Networks
linktitle: Introduction to Neural Networks

date: '2020-02-05T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 3

menu:
  core_sp20:
    parent: Spring 2020

authors: [JarvisEQ, DillonNotDylan]

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
tags: [Neural Networks, Gradient Descent, Backpropagation, Deep Learning]
abstract: >-
  You've heard about them: Beating humans at all types of games, driving cars,
  and recommending your next Netflix series to watch, but what ARE neural networks?
  In this lecture, you'll actually learn step by step how neural networks function
  and learn. Then, you'll deploy one yourself!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-neural-nets").exists():
    DATA_DIR /= "ucfai-core-sp20-neural-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-neural-nets/data
    DATA_DIR = Path("data")
```
