---
layout: post
title:  "论文阅读：《BLEU: a Method for Automatic Evaluation of Machine Translation》"
date:   2017-08-23
categories: 论文阅读 自然语言处理
---

这是2002年的一篇论文，引用率达六千多。BLEU是一种评估机器翻译系统的方法，其目的是替代人工评估。自此以后，机器翻译相关的论文都用BLEU分数衡量机器翻译系统的优劣。

怎么衡量翻译的好坏呢？显然是跟人工翻译越接近，翻译就越好。那么，我们就需要两个要素：一个是表达“翻译相似性”的数字度量标准，一个是人工参考翻译(reference)。先从一些基准的度量标准讲起。

<h4>The Baseline BLEU Metric</h4>
人类很容易区分好的翻译和坏的翻译。比如，考虑下面两句从中文翻译来的英文：
<h5>Example 1</h5>
Candidate 1: *It is a guide to action which ensures that the military always obeys the commands of the party.* <br />
Candidate 2: *It is to insure the troops forever hearing the activity guidebook that party direct.*

然后有三个参考：<br />
Reference 1: *It is a guide to action that ensures that the military will forever heed Party commands.*<br />
Reference 2: *It is the guiding principle which guarantees the military forces always being under the command of the Party.*<br />
Reference 3: *It is the practical guide for the army always to heed the directions of the party.*

我们可以观察到候选1跟三个参考有很多词和短语重复，但是候选2没有。通过简单的n元(n-gram)匹配，候选1可以比候选2排名高。我们首先看1元匹配。
<h5>修改后的n元查准率(precision)</h5>
要计算查准率，统计候选句子出现在任意参考句子里的词(1元)的个数，然后除以候选句子的词数。但是这样有的译句即使不好，也会得到较高的查准率，比如：
<h5>Example 2</h5>
Candidate: *the the the the the the the the.* <br/>
Reference 1: *The cat is on the mat.*<br/>
Reference 2:*There is a cat on the mat.*<br/>
Modified Unigram Precision = 2/7<br/>

修改后的1元查准率这样计算：首先统计某个词出现在单个参考句子的最大次数（注意次数跟个数的区别）。然后，选某个词在候选句子里的个数和它出现在单个参考句子的最大次数两者较小的那个数作为count，把候选句子所有词的count加在一起，除以候选句子的词数。

修改后的n元查准率的计算方法类似。在Example 1中，候选1的2元查准率是10/17，候选2是1/13。
<h5>结合这些修改后的n元查准率</h5>
怎么结合这些n元查准率呢？查准率的加权平均对数满足了要求。 BLEU用均匀的平均对数，相当于查准率的几何平均数。
<h4>BLEU details</h4>
我们计算测试语料库的几何平均查准率，$$p_{n}$$，然后乘以指数简洁惩罚因子。
