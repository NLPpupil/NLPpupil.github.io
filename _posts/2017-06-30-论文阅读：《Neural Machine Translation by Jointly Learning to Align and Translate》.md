---
layout: post
title:  "论文阅读：《Neural Machine Translation by Jointly Learning to Align and Translate》"
date:   2017-06-30
categories: 论文阅读 深度学习 自然语言处理
---

这是2015年的一篇会议论文，作者是Dmitry Bandana，KyungHyun Cho，Yoshua Bengio。本文最重要的贡献就是提出了attention机制。下面我先总结每段的要点，然后适当添加自己的解读。

>**ABSTRACT**
>神经机器翻译（NMT）里，别人大多用encoder-decoder，我们推测把原句编码到一个固定长度的向量是一个瓶颈，然后我们提出了改进。
>
>
>**1 INTRODUCTION**
>神经机器翻译是新技术，大多数都是encoder-decoder。一个潜在的问题是把信息都压缩在固定长度的向量，无法对应长句子。为了解决这个问题，我们提出了一个扩展，它同时进行对齐和翻译。每次我们的模型生成新的翻译词，它在原句那些最有可能包含有关信息的位置上进行搜索。
>这个方法最重要的特点是，它没有尝试将原句的所有部分编码到固定长度的向量，而是它把原句编码到一序列向量，然后在解码的时候灵活的选用这个序列的子集。
>
>
>**2 BACKGROUND：NEURAL MACHINE TRANSLATION**
>从统计的角度看，翻译相当于寻找译句 $$\textbf{y}$$，使得给定原句  $$\textbf{x}$$ 时条件概率最大，即 $$arg max_{\textbf{y}} p(\textbf{y} | \textbf{x})$$。
>
>
>**2.1 RNN ENCODER-DECODER**
>在Encoder-Decoder框架里，编码器把原句，一个序列的向量 $$x = (x_{1},...,x_{T_{x}})$$，编码到一个向量$$c$$。最普遍的方法是用一个RNN：
>
>$$
\begin{equation}
h_{t} = f(x_{t},h_{t-1})
\end{equation}
>$$
>还有
>
>$$
\begin{equation}
 c = q(\{h_{t},...,h_{T_{x}}\})
\end{equation}
>$$
>
>解码器用来给定上下文向量$$c$$和所有之前预测好的词$$\{y_{1},...,y_{t^{'}-1}\}$$，预测下一个词$$y_{t^{'}}$$
。换句话说，解码器定义了翻译 $$\textbf{y}$$上的概率分布：
>
>$$
\begin{equation}
p(\textbf{y}) = \prod_{t=1}^{T}p(y_{t} \bracevert \{y_{1},...,y_{t-1}\},c)
\end{equation}
>$$
>
>在这里 $$\textbf{y} = (y_{1},...,y_{T_{y}})$$。有了RNN，每个条件概率都表示成：
>
>$$
\begin{equation}
p(y_{t} \bracevert \{y_{1},...,y_{t-1}\},c) = g(y_{t-1},s_{t},c)
\end{equation}
>$$
>
>在这里$g$是一个非线性的，多层的函数，$$s_{t}$$是decoder RNN的隐状态。

解读：到这里编码器和解码器都定义好了，我们的目的就是同时训练编码器和解码器，使得它们能最合适地表征训练数据。换句话说，使得训练好的编码器和解码器最大可能地生成训练数据。这就是极大似然估计的含义。
>**3 LEARNING TO ALIGN AND TRANSLATE**
>
>
>**3.1 DECODER:GENERAL DESCRIPTION**
>在新的模型结构里，我们定义条件概率为：
>
>$$
\begin{equation}
p(y_{i} \bracevert \{y_{1},...,y_{i-1}\},\textbf{x}) = g(y_{i-1},s_{i},c_{i})
\end{equation}
>$$
>
>在这里$$s_{i}$$是RNN $$i$$时刻的隐状态，这样计算：
>
>$$
\begin{equation}
s_{i} = f(s_{i-1},y_{i-1},c_{i})
\end{equation}
>$$
>
>值得一提的是，跟之前的不一样，这里每个词$$y_{i}$$都有单独的上下文$$c_{i}$$。

解读：关键部分来了，在之前的模型里，一句话的每个词都共用一个上下文，但是新的模型是一句话的每个词都有自己的上下文，这就是注意力机制的精髓。上下文是怎么计算的呢？下面讲了。
>上下文$$c_{i}$$取决于编码器产生的注释序列$$(h_{1},...,h_{T_{x}})$$。每个注释$$h_{i}$$都包含了整个原句的信息，伴随着对原句第$$i$$个词周围的关注。下一节会详细解释注释。
>上下文向量$$c_{i}$$是这些注释的加权和：
>
>$$
\begin{equation}
c_{i} = \sum_{j=1}^{T_{x}}\alpha_{ij}h_{j}
\end{equation}
>$$
>每个注释的权重这样计算：
>
>$$
\begin{equation}
\alpha_{ij}= \frac{\exp(e_{ij})}{\sum_{k=1}^{T_{x}}\exp(e_{ik})}
\end{equation}
>$$
>
>在这里
>
>$$
\begin{equation}
e_{ij} = a(s_{i-1},h_{j})
\end{equation}
>$$
>
>是一个对齐模型，这个模型衡量了原句的$$j$$位置和译句的$$i$$位置在多大程度上匹配。对齐模型$$a$$作为一个前馈神经网络，跟编码器和解码器共同进行训练。

解读：到这里就呼应题目了——对齐和翻译是同时学习的。学习对齐模型只是手段，目的是完善注意力机制——计算每个翻译词的上下文。下段继续解释对齐模型和权重。
>我们可以把计算注释的加权和看成计算*期望注释*。把$$\alpha_{ij}$$当做译句词$$y_{i}$$由原句词$$x_{j}$$翻译而来的概率。概率$$\alpha_{ij}$$反映了注释$$h_{j}$$相对于前一个隐状态$$s_{i-1}$$在预测下一个状态$$s_{i}$$和生成$$y_{i}$$过程中的重要性。直观上，这在解码器上实现了注意机制。

解读：简单讲，$$\alpha_{ij}$$越大，翻译第$$i$$个词的时候就更应该着重注意原句第$$j$$个词。
>**3.2 ENCODER:BIDIRECTIONAL RNN FOR ANNOTATING SEQUENCES**
>
>以往的RNN，都是从句子第一个符号读到最后一个符号。然而，我们想要让注释不仅囊括之前的信息，还要包含之后的信息，所以我们采用双向RNN。
>
>一个BiRNN由向前和向后RNN组成。向前RNN $$\overrightarrow{f}$$从左到右读取原句（从$$x_{1}$$到$$x_{T_{x}}$$），然后计算一个序列的*向前隐状态*$$(\overrightarrow{h}_{1},...,\overrightarrow{h}_{T_{x}})$$。向后RNN $$\overleftarrow{f}$$反方向读取原句，然后计算一个序列的*向后隐状态*$$(\overleftarrow{h}_{1},...,\overleftarrow{h}_{T_{x}})$$。
>
>剩下的几章就是实验，结论之类，略掉。
