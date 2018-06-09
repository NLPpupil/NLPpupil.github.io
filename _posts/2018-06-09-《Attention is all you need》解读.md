---
layout: post
title:  "《Attention is all you need》解读"
date:   2018-06-09
categories: 自然语言处理 深度学习
---

### 基础要求

- 掌握基本的encoder-decoder + attention结构。
- 
- 

### Attention 定义
你所需要的只是注意力，那么注意力是什么呢？原文有一段话，要深刻理解：
>An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.



