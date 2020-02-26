---
title: "Building AI, the Human Way"
linktitle: "Building AI, the Human Way"

date: "2020-04-08T17:30:00"
lastmod: "2020-04-08T17:30:00"

draft: false
toc: true
type: docs

weight: 11

menu:
  core_sp20:
    parent: Spring 2020
    weight: 11

authors: ["ionlights", ]

urls:
  youtube: ""
  slides:  ""
  github:  ""
  kaggle:  ""
  colab:   ""

location: ""
cover: "https://neuroscape.ucsf.edu/wp-content/uploads/glassbrain-gazzaleylab-neuroscapelab-18-1024x542.jpg"

categories: ["sp20"]
tags: ["machine learning", "common sense reasoning", "computational cognitive science", "cognitive science", "probabilistic programming", "program induction", "intuitive theories", "intuitive physics", "intuitive psychology", ]
abstract: >-
  We’ve learned about linear and statistical models as well as different training paradigms, but we’ve yet to think about how it all began. In Cognitive Computational Neuroscience, we look at AI and ML from the perspective of using them as tools to learn about human cognition, in the hopes of building better AI systems, but more importantly, in the hopes of better understanding ourselves.asdf
---
```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-meeting10").exists():
    DATA_DIR /= "ucfai-core-sp20-meeting10"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-meeting10/data
    DATA_DIR = Path("data")
```
