---
layout: post
title:  "Stanford CoreNLP安装与使用（支持中文）"
date:   2018-07-18-
categories: 自然语言处理
---

Stanford CoreNLP很好很强大，但是官网的安装指南并不是一目了然。本文记录了Linux或MacOS上的安装流程，特别是中文包的导入。然后简单介绍使用方法。

#### 安装
进入[Stanford CoreNLP – Natural language software](https://stanfordnlp.github.io/CoreNLP/)，下滑找到**Download**按钮，点击下载。

<img src="https://nlppupil.github.io/images/corenlpdownload.png" alt="BP" style="width:650px;height:160px;">

解压，将文件夹放入任意目录。

同样在这个网站上，找到

<img src="https://nlppupil.github.io/images/corenlpchinese.png" alt="BP" style="width:300px;height:300px;">

点击Chinese栏的**download**下载中文模型jar包，将下载好的中文模型jar包放入解压好的`stanford-corenlp-full-2018-02-27/`文件夹。

打开`~/.bashrc`(MacOS上为`~/.bash_profile`)，添加

```
export CLASSPATH=/Users/pupil/Downloads/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar\
:/Users/pupil/Downloads/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar\
:/Users/pupil/Downloads/stanford-corenlp-full-2018-02-27/stanford-chinese-corenlp-2018-02-27-models.jar
```

其中`/Users/pupil/Downloads`改为`stanford-corenlp-full-2018-02-27`所在的绝对路径，版本号也要改成当前下载对应的版本号。

最后，`source ~/.bashrc`。

在其他任何文件夹下运行
`echo "the quick brown fox jumped over the lazy dog" > input.txt` `java -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat json -file input.txt`
如果通过，表示安装成功。`-Xmx3g `表示为程序最大分配3G内存。

#### 使用
大多数问题都可以通过`java -Xmx16g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize -file input.txt`或`java -Xmx16g edu.stanford.nlp.pipeline.StanfordCoreNLP -props StanfordCoreNLP-chinese.properties -annotators tokenize-file input.txt`解决，前一个是英文，后一个是中文。不同的问题在`-annotators`后面添加标注器的名字就可以，比如中文命名体识别，可以用`java -Xmx16g edu.stanford.nlp.pipeline.StanfordCoreNLP -props StanfordCoreNLP-chinese.properties -annotators tokenize,ssplit,pos,lemma,ner -file input.txt`，注释器有依赖，`ner`前面必须先有`tokenize,ssplit,pos,lemma`。

详细用法查看官方文档，[Using Stanford CoreNLP from the command line](https://stanfordnlp.github.io/CoreNLP/cmdline.html)。