---
title: 'Beyond the Buzzwords: Getting Started with Neural Networks'
linktitle: 'Beyond the Buzzwords: Getting Started with Neural Networks'

date: '2019-02-13T19:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 2

menu:
  core_sp19:
    parent: Spring 2019

authors: [ahkerrigan, ionlights]

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
tags: [neural networks, deep learning, machine learning]
abstract: >-
  You've heard about them: Beating humans at all types of games, driving cars,
  and recommending your next Netflix series to watch, but what ARE neural  networks?
  In this lecture, you'll actually learn step by step how neural
  networks function and how they learn. Then, you'll deploy one yourself!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp19-neural-nets").exists():
    DATA_DIR /= "ucfai-core-sp19-neural-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp19-neural-nets/data
    DATA_DIR = Path("data")
```
