---
title: A Look Behind DeepFake ~ GANs
linktitle: A Look Behind DeepFake ~ GANs

date: '2020-03-18T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 8

menu:
  core_sp20:
    parent: Spring 2020

authors: [brandons209, bb912]

urls:
  youtube: ''
  slides: ''
  github: https://github.com/ucfai/core/blob/master/sp20/03-18-gans/gans.ipynb
  kaggle: https://kaggle.com/ucfaibot/core-sp20-gans
  colab: https://colab.research.google.com/github/ucfai/core/blob/master/sp20/03-18-gans/gans.ipynb

papers: {}

location: HPA1 112
cover: ''

categories: [sp20]
tags: [DeepFake, GANs, CycleGANs, generative models]
abstract: >-
  GANs are relativity new in the machine learning world, but they have
  proven to be a very powerful architecture. Recently, they made headlines
  in the DeepFake network, being able to mimic someone else in real time in
  both video and audio. There has also been cycleGAN, which takes one domain
  (horses) and makes it look like something similar (zebras). Come and learn
  the secret behind these type of networks, you will be surprised how
  intuitive it is! The lecture will cover the basics of GANs and different
  types, with the workshop covering how we can generate human faces, cats,
  dogs, and other cute creatures!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-gans").exists():
    DATA_DIR /= "ucfai-core-sp20-gans"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-gans/data
    DATA_DIR = Path("data")
```
