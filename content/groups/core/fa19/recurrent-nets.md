---
title: Writer's Block? RNNs Can Help!
linktitle: Writer's Block? RNNs Can Help!

date: '2019-10-23T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 6

menu:
  core_fa19:
    parent: Fall 2019

authors: [brandons209, ionlights]

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
tags: [neural networks, recurrent networks, deep learning, long short-term memory,
  embeddings]
abstract: >-
  This lecture is all about Recurrent Neural Networks. These are networks
  with memory, which means they can learn from sequential data such as speech,
  text, videos, and more. Different types of RNNs and strategies for building  them
  will also be covered. The project will be building a LSTM-RNN to generate
  new original scripts for the TV series “The Simpsons”. Come and find out if our
  networks can become better writers for the show!

---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-recurrent-nets").exists():
    DATA_DIR /= "ucfai-core-fa19-recurrent-nets"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-recurrent-nets/data
    DATA_DIR = Path("data")
```
