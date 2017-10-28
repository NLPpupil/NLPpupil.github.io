---
layout: post
title:  "CS224n Assignment 3"
date:   2017-10-28
categories: 其他
---

这是斯坦福大学[CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/assignment3/index.html)的第三个作业的翻译。见[Assignment 3: Named Entity Recognition and Recurrent Neural Networks](https://github.com/NLPpupil/cs224n_assignment3)。

问题分为三部分。第一部分是窗口的命名体识别，第二部分是RNN的命名体识别，第三部分介绍了GRU。

#### 命名体识别介绍
命名体识别（Name Entity Recoginition,NER)目的是在文本中找出之前定义好的类别，比如人名、地名、组织、时间、数量等等。在这个作业里，对于一个在上下文中给定的词，我们想要预测他是否属于以下四类：人（PER)、组织（ORG)、位置（LOC)、混杂（MISC，比如「日本人」，「美元」，「1000」）。

这是一个五分类问题，加上一个（O）表示不属于任何命名体。由多个词组成的命名体要每个词都标记，比如「中华人民共和国」。连续被标记的词整体当做一个命名体。
