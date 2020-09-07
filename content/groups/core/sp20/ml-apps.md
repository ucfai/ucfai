---
title: Machine Learning Applications
linktitle: Machine Learning Applications

date: '2020-02-26T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 6

menu:
  core_sp20:
    parent: Spring 2020

authors: [brandons209, nspeer12]

urls:
  youtube: ''
  slides: ''
  github: https://github.com/ucfai/core/blob/master/sp20/02-26-ml-apps/ml-apps.ipynb
  kaggle: https://kaggle.com/ucfaibot/core-sp20-ml-apps
  colab: https://colab.research.google.com/github/ucfai/core/blob/master/sp20/02-26-ml-apps/ml-apps.ipynb

papers: {}

location: HPA1 112
cover: ''

categories: [sp20]
tags: [Applications, Pokémon, Pokédex, Expolanets, Machine Learning]
abstract: >-
  It's time to put what you have learned into action. Here, we have prepared
  some datasets for you to build a a model to solve. This is different from
  past meetings, as it will be a full workshop. We provide the data sets and
  a notebook that gets you started, but it is up to you to build a model to
  solve the problem. So, what will you be doing? We have two datasets, one
  is using planetary data to predict if a planet is an exoplanet or not, so
  your model can help us find more Earth-like planets that could contain
  life! The second dataset will be used to build a model that mimics a
  pokedex! Well, not fully, but the goal is to predict the name of a pokemon
  and also predict its type (such as electric, fire, etc.) This will be
  extremely fun and give you a chance to apply what you have learned, with
  us here to help!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-ml-apps").exists():
    DATA_DIR /= "ucfai-core-sp20-ml-apps"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-ml-apps/data
    DATA_DIR = Path("data")
```
