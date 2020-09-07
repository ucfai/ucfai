---
title: Generative Adversarial Networks
linktitle: Generative Adversarial Networks

date: '2019-04-08T17:30:00'
lastmod: <?UNK?>

draft: false
toc: true

weight: 8

menu:
  discussions_sp19:
    parent: Spring 2019

authors: [ionlights, waldmannly]

urls:
  youtube: ''
  slides: ''
  github: ''
  kaggle: ''
  colab: ''

papers:
  gans: https://github.com/ucfai/discussions/raw/master/sp19/gans.pdf
location: HEC 119
cover: ''

categories: [sp19]
tags: [generative models, generative adversarial networks, deep learning, adversarial
    learning]
abstract: >-
  Abstract: We propose a new framework for estimating generative models via
  an adversarial process, in which we simultaneously train two models: a
  generative model G that captures the data distribution, and a
  discriminative model D that estimates the probability that a sample came
  from the training data rather than G. The training procedure for G is to
  maximize the probability of D making a mistake. This framework corresponds
  to a minimax two-player game. In the space of arbitrary functions G and D,
  a unique solution exists, with G recovering the training data distribution
  and D equal to 1/2 everywhere. In the case where G and D are defined by
  multilayer perceptrons, the entire system can be trained with
  backpropagation. There is no need for any Markov chains or unrolled
  approximate inference networks during either training or generation of
  samples. Experiments demonstrate the potential of the framework through

---

<!-- TODO Add Meeting Notes/Contents here -->
<!-- NOTE Refer the Documentation if you're unsure how to format/add to this. -->
