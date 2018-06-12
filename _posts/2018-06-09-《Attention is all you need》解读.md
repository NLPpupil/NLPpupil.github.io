---
layout: post
title:  "《Attention is all you need》解读"
date:   2018-06-09
categories: 自然语言处理 深度学习
---

## 基础要求

- 传统NMT encoder-decoder + attention结构。
- 概率基础。
- 矩阵乘法。
- Residual Connection
- Layer Normalization



## Attention 定义
你所需要的只是注意力，那么注意力是什么呢？原文有一段精辟的定义，要深刻理解：
>An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

在传统NMT结构里，双向RNN编码器输出一个序列的向量 $\{h_1,h_2,...,h_t\}$，解码器每次解码的时候，query是解码器上一时刻的隐状态，一个序列的key-value是 $\{(h_1,h_1),(h_2,h_2),...,(h_t,h_t)\}$ ，即key和value相同。query跟每个key做一个点乘得到分数，将分数用$softmax$归一化得到权重，将权重跟value做加权和得到当前时刻的context。context作为解码器的RNN的当前时刻的输入进行更新。

## 模型组成
把整个模型当做一个函数，该函数由若干个子函数的复合函数组成。解读流程就是从一开始的子函数说起，直到最后一个子函数的输出。


模型输入：将每个词转成unique整数之后的原句，已经解码了的词序列（将每个词转成unique整数）。
模型输出：对译句所有词的概率分布。

### Encoder部分
#### 子函数1：词向量矩阵和Positional Encoding
输入：同模型输入 
<br>
输出：输入的每个词转成一个$d_{model}$维的向量，贯穿全篇 $p_{model} = 512$

这个子函数分两步，第一步将每个词通过词向量矩阵转成一个$d_{model}$维的向量。这是普通词向量的常规做法。

第二步，为每一个词做一个位置嵌入，这样之后的attention层（子函数）就可以感知每个词的位置。公式是：

$$
PE(pos,2i) = sin(pos/10000^{2i/d_{model}}) \\
PE(pos,2i+1) = cos(pos/10000^{2i/d_{model}})
$$

其中， $pos$表示词的相对位置。比如原句有五个词，那么这五个词的位置嵌入就是 $\{PE(0),PE(1),PE(2),PE(3),PE(4)\}$。$PE(pos,i)$表示$PE(pos)$的第$i$维。让$i=0-255$就可以得到$PE(pos)$每个维度的数值，进而得到$PE(pos)$。

将每个词的词嵌入和位置嵌入做一个向量加法，就得到了最终输出。

#### 子函数2：一系列Scaled Dot-Product Attention
子函数1的输出是两个矩阵，行数为词个数，列数为$d_{model}$。前一个矩阵是原句得到的，后一个是译句得到的。两个参数不同的子函数2分别作用于两个矩阵，这里只说encoder部分的子函数2。

输入：$num\_ words \times d_{model}$ 的矩阵
<br>
输出：$num\_ words \times d_{v}$ 的矩阵

Scaled Dot-Product Attention中scaled的含义是对常规注意力得到的向量乘以一个缩放系数(scaling factor)。它的query是词的向量做一个 $W^Q \in \mathbb{R}^{d_{model} \times d_k}$的线性变换，即将一个$d_{model}$维的行向量线性变换到一个$d_k$维的行向量，论文中$d_k = 64$。它的keys是为每个词的向量做一个$W^K \in \mathbb{R}^{d_{model} \times d_k}$ 的线性变换得到一个$d_k$维的向量，即keys是一个序列的向量，一个$num\_words \times d_k$的矩阵。它的values是为每个词的向量做一个$W^V \in \mathbb{R}^{d_{model} \times d_v}$ 的线性变换得到一个$d_v$维的向量，即values是一个序列的向量，一个$num\_words \times d_v$的矩阵，这里$d_v = d_k$。
我们要为每一个词做一个Scaled Dot-Product Attention得到这个词的attention向量，计算每个词的时候以这个词的向量的线性变换作为query，所有词的向量的线性变换作为keys和values，query和每个key做一个点积得到相应key的权重，将所有values做加权和得到当前词的attentio向量。因为是句子内部进行注意力权重分配，不像传统NMT结构那样需要外部query(decoder的隐状态)，所以叫self attention。

这个一系列对每个词做一个attention的子函数2可以用矩阵乘法一次性解决，不需要像RNN encoder那样每次处理一个时间点，这就是parallelization的含义，将时间复杂度从$O(n)$缩小到$O(1)$。上段已经隐约表明$Q = K = V$，是子函数1得到的原句的矩阵。子函数2的输出就是：

$$
Attention(QW^Q,KW^K,QW^V) = softmax(\frac{(QW^Q)(KW^K)^T}{\sqrt{d_k}})(QW^V)
$$

这就是论文中的式$(1)$，区别是式$(1)$中的$Q$在这里是$QW^Q$。$(QW^Q)(KW^K)$(记为$weights$)是$num\_words \times num\_words$的矩阵，它的第$i$行表示计算第$i$个词的attention向量的时候每个词的向量的分数，然后对$weights$中的每一个数乘以缩放系数，再对每一行做一个$softmax$归一化，即$softmax$的对象是一个矩阵而不是向量，它对矩阵的每一行做归一化。$Attention(QW^Q,KW^K,QW^V)$的最终结果是一个$num\_words \times d_v$的矩阵，它的第$i$行表示第$i$个词的attention向量，在传统NMT语境下是context向量的意思。





#### 子函数3：Multi-Head Attention
Multi-Head Attention重点在Multi-Head上，多头的意思是假设每个词有多个角度的信息，一次attention操作获得一个角度的信息。如果同时做$h$个attention操作，再将这些attention向量连接起来，那么就会得到更加饱满的信息，论文里$h=8$。

也就是子函数2是子函数3的一部分，子函数3同时做$h$个子函数2，这样每个词就有$h$个attention向量，再将每个词的$h$个向量连接起来，做一个$W^O \in \mathbb{R}^{d_{model} \times d_{model}}$的线性映射，就是每个词经过多头注意力后的结果。

因此，子函数3的输出仍是一个$num\_words \times d_{model}$的矩阵。


#### 子函数4： Residual Connection + Layer Normalization
encoder有6个相同的层，每个层又有两个子层。子函数2，3是第一个子层，记为sublayer1。sublayer1的输入是子函数1的输出，记为x，是一个$num\_words \times d_{model}$的矩阵。sublayer1的输出是子函数3的输出，也是$num\_words \times d_{model}$的矩阵。子函数4的输出是：
LayerNorm(x + sublayer1(x))，也是一个$num\_words \times d_{model}$的矩阵。

#### 子函数5：Position-wise Feed-Forward Networks
一个position对应一个词，一个词对应矩阵的一行，子函数5对矩阵的每一行做两个线性映射，中间加一个ReLU。

输入：子函数4的输出
<br>
输出：相同形状的矩阵

记输入矩阵的某行为x，则：

$$
FFN(X)= max(0,xW_1 + b_1)W_2 + b_2
$$

这是encoder每层的第二个子层，再做一个子函数4。子层1和2加在一起构成了encoder的一个层，六个这样的层构成了encoder。

### Decoder部分
decode的第一个多头注意力跟encoder的多头注意力一样操作，第二个多头注意力的区别是$Q$变成第一个的多头注意力输出。第二个注意力的作用跟传统NMT的attention作用一样，比较decoder当前向量跟原句哪个词相近，就更注意那个词。decoder的$FFN$跟encoder一样的操作。

最后对decoder输出的矩阵的最后一行做一个线性变换和softmax，映射到词表的概率分布，就是模型最终的输出。

之前说到模型输入的译句部分是已经解码过的词，实际操作中，训练时的输入其实是整个译句，不过在Scaled Dot-Product Attention的时候给未解码的词加了一个负无穷的mask。翻译时不需要顾及未解码的词（因为不存在），所以不用加mask。




