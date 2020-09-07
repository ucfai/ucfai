---
title: Solving the Computationally Impossible with Heuristics
linktitle: Solving the Computationally Impossible with Heuristics

date: '2020-10-24T16:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 8

menu:
  core_fa18:
    parent: Fall 2018

authors: []

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
  The world is complex, making it difficult for algorithms to come to solutions in
  reasonable amounts of time. To speed them along, we can employ Heuristics to get
  us significantly closer, faster.  Today, we’ll try to approximate the Traveling
  Salesman Problem by using Simulated Annealing and Particle Swarm Optimization –
  two Heuristics which move us towards finding the shortest path we can use to visit
  all the destinations.

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa18-heuristics").exists():
    DATA_DIR /= "ucfai-core-fa18-heuristics"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa18-heuristics/data
    DATA_DIR = Path("data")
```
