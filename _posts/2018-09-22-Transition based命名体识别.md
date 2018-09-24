---
layout: post
title:  "Transition-Based命名体识别"
date:   2018-09-22
categories: 自然语言处理 深度学习
---

#### 引言
[Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030) 这篇2016年引用达544的论文提出了两个实体识别方法，第一个是广为使用的LSTM+CRF模型，第二个就是transition-based模型。

Transition-based NER用类似transition-based依存句法分析(dependency parsing)的方法来给句子分块(chunk)和打标签。在NER里，分块就是把组成同一个实体的连续词语分在一起，打标签就是为某个词或若干个连续的词分类成某个实体类别。transition的意思是转移，紧密围绕transition的概念有:

1. buffer，存放输入序列。
2. shift，挪走buffer前面的一个元素，放到stack上。
3. stack，存放从buffer挪过来的元素。
4. configuration，刻画当前各部分的状态，最简单的由当前stack的情况和buffer的情况组成。
4. reduce，把stack里的所有元素提取出来作为一个块(chunk)，打上标签并放入另一个存放输出的stack。

shift和reduce都是transition，算法就是在这些转移中进行和结束的，所以不管是依存句法分析还是命名体识别，凡是用到了转移，都叫transition-based methods。下面详细解释基于转移的命名体识别。

#### 基于转移的命名体识别
一开始($t=0$)，输入句子以词序列的形式放入*buffer*，有两个空的栈，输出栈 *output stack*和临时栈 *temp stack*。在接下来的每一时刻，都要做出一个行动*action*，直到*buffer*为空且*temp stack*为空，*action*包括(1)*shift*：将*buffer*的第一个词挪到*temp buffer*；（2）*out*：直接将*buffer*的第一个词挪到*output stack*；(3)*reduce*：将*temp stack*的所有词取出作为一个块，标记上实体类别，放入*output stack*。做出某个行动的依据是当前的*configuration*，*configuration*由当前*buffer*的状态，*temp stack*的状态，*output stack*的状态和过往的所有行动组成。根据*configuration*选择行动是一个分类问题。绝大多数机器学习问题归根结底都是分类问题。

怎么从*configuration*分类行动标签呢？*configuration*首先要通过神经网络转换成一个向量表示，这个向量表示再线性映射到一个三维向量，三维向量中分数大的那个相应行动就是分类结果。

现在问题转化成了怎么获得*configuration*的向量表示。configuration的向量表示是*buffer*，*temp stack*，*output stack*和过往行动四者各自的contents的拼接(concatenation)。*buffer*的content是已经挪走的词的向量经过LSTM后的隐状态，*temp stack*的content是栈上的词向量经过LSTM后的隐状态，*reduce*行动把*temp stack*上的词向量序列经过双向LSTM得到两个隐状态，和类别的向量拼接到一起作为*reuduce*行动放在*output stack*的向量，*output stack*的content是栈的向量经过LSTM后的隐状态。

根据标记数据，计算每个行动的损失，用梯度下降进行参数训练。

