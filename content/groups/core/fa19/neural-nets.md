---
title: Getting Started with Neural Networks
linktitle: Getting Started with Neural Networks

date: '2019-10-02T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 3

menu:
  core_fa19:
    parent: Fall 2019

authors: [jarviseq, ionlights]

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
tags: [neural networks, gradient descent, optimization]
abstract: >-
  You've heard about them: Beating humans at all types of games, driving cars,
  and recommending your next Netflix series to watch, but what ARE neural  networks?
  In this lecture, you'll actually learn step by step how neural
  networks function and learn. Then, you'll deploy one yourself!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-neural-nets").exists():
    DATA_DIR /= "ucfai-core-fa19-neural-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-neural-nets/data
    DATA_DIR = Path("data")
```
