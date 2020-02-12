# python 3.7
# -*- coding: utf-8 -*-
# @Time    : 2020-02-11 14:59
# @Author  : Xueli
# @File    : generate_corpus.py
# @Software: PyCharm

from gensim.models import word2vec
import time





corpus = '/Users/sherry/Downloads/lemmatized_corpus.txt'

f.open(corpus)
start_time = time.time()
# training model
sentences=word2vec.Text8Corpus(corpus)
# set parameters
model=word2vec.Word2Vec(sentences,min_count=5, size=300, window=5, workers=4)
print(time.time() - start_time)
#------------------------------------------------------------------------
# Word2vec有很多可以影响训练速度和质量的参数：
# （1） sg=1是skip-gram算法，对低频词敏感，默认sg=0为CBOW算法，所以此处设置为1。
# （2） min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
# （3） size是输出词向量的维数，即神经网络的隐藏层的单元数。值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，大的size需要更多的训练数据, 但是效果会更好，在本文中设置的size值为300维度。
# （4） window是句子中当前词与目标词之间的最大距离，即为窗口。本文设置窗口移动的大小为5。
# （5） negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
# （6） hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
# （7） 最后一个主要的参数控制训练的并行:worker参数只有在安装了Cython后才有效，由于本文没有安装Cython的, 使用的单核。
#------------------