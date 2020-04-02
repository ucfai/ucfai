---
title: "Building AI, the Human Way"
linktitle: "Building AI, the Human Way"

date: "2020-04-08T17:30:00"
lastmod: "2020-04-02T19:11:29.875346184"

draft: false
toc: true
type: docs

weight: 10

menu:
  core_sp20:
    parent: "Spring 2020"
    weight: "10"

authors: ["ionlights", ]

urls:
  youtube: ""
  slides:  ""
  github:  ""
  kaggle:  ""
  colab:   ""

location: "HPA1 112"
cover: ""

categories: ["sp20"]
tags: ["Machine Learning", "Common Sense AI", "Computational Cognitive Science", "CoCoSci", "Cognitive Science", "Probabilistic Programming", "Program Induction", "Intuitive Theories", "Intuitive Physics", "Intuitive Psychology", ]
abstract: "We've learned about linear and statistical models as well as different training paradigms, but we've yet to think about how it all began. In Cognitive Computational Neuroscience, we look at AI and ML from the perspective of using them as tools to learn about human cognition, in the hopes of building better AI systems, but more importantly, in the hopes of better understanding ourselves."
---

```
from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-sp20-build-ai-the-human-way").exists():
    DATA_DIR /= "ucfai-core-sp20-build-ai-the-human-way"
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-build-ai-the-human-way/data
    DATA_DIR = Path("data")
```
