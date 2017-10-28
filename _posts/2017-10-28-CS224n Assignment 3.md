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

#### 1 窗口NER
原始数据形式是$x = x^{(1)},x^{(2)},...,x^{(T)}$ 是一个长度为$T$的输入序列，$y=y^{(1)},y^{(2)},...,y^{(T)}$ 是正确标记。这里$x^{(t)},y^{(t)}$ 是表示一个一个词或标记的独热向量。在窗口NER里，$x^{(t)}$ 变成了 $\widetilde{x}^{(t)}=[x^{(t-w)},...,x^{(t)},...,x^{(t+w)}]$。$y^{(t)}$ 不变。$w$是窗口半径。句首用<**START**>填充，句尾用<**END**>填充。我们用简单的前馈神经网络从 $\widetilde{x}^{(t)}$ 预测 $y^{(t)}$ 。数学表示如下：

$$
\begin{equation}
e^{(t)} = [x^{(t-w)}L,...,x^{(t)}L,...,x^{(t+w)}L] \\
h^{(t)} = ReLU(e^{(t)}W+b_1) \\
\hat y^{(t)} = softmax(h^{(t)}U+b_2)\\
J = CrossEntropy(y^{(t)},\hat y^{(t)} \\
CrossEntropy(y^{(t)},\hat y^{(t)}) = - \sum_i{}{}y_i^{(t)} log(\hat y_i^{(t)})
\end{equation}
$$

这里$L \in \mathbb{R}^{V \times D}$ 是词嵌入，$h^{(t)}$ 的维度是 $H$ ， $\hat y^{(t)}$ 的维度是$C$，$V$是词汇表的大小，$D$是词嵌入的维度，$H$是隐层的大小，$C$是预测种类的个数（在这个问题里是5）。

**问题**：
- 在`make_windowed_data`函数里把一个批量的输入序列转换成一个批量的窗口化（输入-输出）对。用`python q1_window.py test1`测试。
- 在`WindowModel`类里实现相应的函数来实现之前描述的前馈神经网络。用`python q1_window.py test2`测试。
- 用`python q1 window.py train`训练，大概运行2-3分钟，得到至少81%的$F1$。结果储存在`results/window/<timestamp>/`，可以用`python q1 window.py shell -m results/window/<timestamp>/`互动。
