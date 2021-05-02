# Natural Language Processing (from scratch to GPT-3)
#### Note: This repo is updated regularly as I learn. If you want to learn NLP, just start from the first point and go on till the bottom. Everything is hierarchically arranged (from basic concepts to advanced)
---

## Basic stuff 
#### Cosine Similarity
- [Cosine Similarity vs Euclidean distance](https://cmry.github.io/notes/euclidean-v-cosine)
- [Probability vs Likelihood](https://stats.stackexchange.com/questions/2641/what-is-the-difference-between-likelihood-and-probability)
---

## Text Processing

### Tokenization
- [A comprehensive article on tokenization](https://blog.floydhub.com/tokenization-nlp/)
- [Subword Tokenization](https://www.thoughtvector.io/blog/subword-tokenization/)
- [Byte Pair Encoding](https://leimao.github.io/blog/Byte-Pair-Encoding/)
- [Byte Pair Encoding - Paper](https://arxiv.org/pdf/1508.07909v5.pdf)
- [Comprehensive notebook](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/Basics/Text-Preprocessing/Tokenization.ipynb)

---

### Stemming and Lemmatization
#### Videos
- [YouTube Video - Abhishek Thakur](https://www.youtube.com/watch?v=OQxi-d5C9j8&list=PL98nY_tJQXZk-NeS9jqeH2iY4-IvoYbRC&index=1&ab_channel=AbhishekThakur)
- [Stemming - Krish Naik](https://www.youtube.com/watch?v=1OMmbtVmmbg&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=4&ab_channel=KrishNaik)
- [Lemmatization - Krish Naik](https://www.youtube.com/watch?v=cqcUk6hC5hk&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=5&ab_channel=KrishNaik)

#### Articles
- [Stanford book](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
- [Stemming vs Lemmatization](https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8)

#### Code
- [Practical Implementation](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)
- [Lemmatization different approaches](https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/)

---

### Bag Of Words
#### Videos
- [Krish Naik](https://www.youtube.com/watch?v=iu2-G_5YkEo&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=7&ab_channel=KrishNaik)

#### Articles
- [MachineLearningMastery](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
- [Medium article](https://towardsdatascience.com/a-simple-explanation-of-the-bag-of-words-model-b88fc4f4971)

#### Code
- [Implementation (sklearn + nltk)](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/Basics/Text-Preprocessing/Bag-of-Words.ipynb) 

---

### TF-IDF 
#### Videos
- [Krish Naik](https://www.youtube.com/watch?v=D2V1okCEsiE&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=9)

#### Articles
- [Medium](https://medium.com/analytics-vidhya/tf-idf-term-frequency-technique-easiest-explanation-for-text-classification-in-nlp-with-code-8ca3912e58c3)
  
#### Code
- [Implementation (scratch + sklearn)](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/Basics/Text-Preprocessing/TF-IDF.ipynb)

---

#### Great you are now done with some of the basics, time to implement a basic project

### SpamClassifier Project
- [Video](https://www.youtube.com/watch?v=fA5TSFELkC0&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=11)
- [Code](https://github.com/Gladiator07/Natural-Language-Processing/tree/main/Basics/mini-projects/Spam-Classifier)

--- 
Great, after implementing a basic project it's time to get a bit mathematical

Watch the first lecture of the most sought after course of NLP (CS224N by Stanford)
### CS224N Lecture-1
- [CS224n - Lecture-1 ==> Introduction and Word vectors](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=1&ab_channel=stanfordonline)
- [Slides](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/CS224N-Materials/Word_Vectors/cs224n-2021-lecture01-wordvecs1.pdf)
- [Notes](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/CS224N-Materials/Word_Vectors/cs224n-2019-notes01-wordvecs1.pdf)
#### Suggested Readings
- [Efficient Estimation of Word Representations in Vector space - original word2vec paper](https://arxiv.org/pdf/1301.3781.pdf)
- [Distributed Representations of Words and Phrases and their Compositonality - negative sampling paper](https://arxiv.org/pdf/1301.3781.pdf)

---

### Word2Vec
After watching above lecture and going through the suggested readings (Stanford CS224N), let's understand more about word2vec
- [Jay Alammar Blog - detailed](https://jalammar.github.io/illustrated-word2vec/)
  
#### Code
- [Word2Vec detailed implementation](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/Word2vec/w2v.ipynb)
- [Gensim Word2vec docs](https://radimrehurek.com/gensim/models/word2vec.html)

---

Great, now it's time to do one more projects to solidify the concepts learnt so far

#### Predict Stock Price Movement Based on News Headlines
- [Jupyter Notebook](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/Basics/mini-projects/stock_sentiment_analysis/stock_sentiment_analysis.ipynb)

---

### More about Word Embeddings (CS224N - Lecture-2)

- [Lecture Video](https://www.youtube.com/watch?v=kEMJRjEdNzM&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=2&t=3685s&ab_channel=stanfordonline)
- [Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture02-wordvecs2.pdf)
- [Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)

**Suggested Readings**
- [GloVe: Global Vectors for Word Representation (original GloVe paper)](https://nlp.stanford.edu/pubs/glove.pdf)
- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://nlp.stanford.edu/pubs/glove.pdf)
- [Evaluation methods for unsupervised word embeddings](www.aclweb.org/anthology/D15-1036)



**Additional Readings**
- [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
- [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/2018/file/b534ba68236ba543ae44b22bd110a1d6-Paper.pdf)
- [A Latent Variable Model Approach to PMI-based Word Embeddings](https://www.aclweb.org/anthology/Q16-1028/)
  
---
We have been seeing word embeddings applied in NLP to get the vector representation of the words.
But now let's try them on tabular dataset with categorical features. We will convert the categorical features in word embeddings rather than traditional approaches like one-hot encoding, label encoding, etc.
### Word Embeddings for Categorical Features
- [Code](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/Word2vec/Categorical_Embeddings.ipynb)

---

Let's now move on to the deep learning part of NLP.

## Refresher on Neural Networks
### CS224N - Lecture-3 & 4 (Neural Networks and Backpropagation)

- [Lecture-3](https://www.youtube.com/watch?v=kEMJRjEdNzM&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=3)
- [Lecture-4](https://www.youtube.com/watch?v=yLYHDSv-288&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=4&ab_channel=stanfordonline)
- [Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture03-neuralnets.pdf)
- [Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf)

**Suggested Readings**
- [Matrix Calculus Notes](https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf)
- [Review of differential calculus](https://web.stanford.edu/class/cs224n/readings/review-differential-calculus.pdf)
- [CS231N notes on network architecture](https://cs231n.github.io/optimization-2/)
- [CS231N notes on backprop](http://cs231n.stanford.edu/handouts/derivatives.pdf)
- [Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf)
- [Learning representations by back-propagating errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)

**Additional Readings**
- [Yes you should understand backprop](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)
- [Natural Language Processing (Almost) from Scratch](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)
  


### RNN

This material will get you started with RNN
#### Videos
- [RNN-Intro (Krish Naik)](https://www.youtube.com/watch?v=CPl9XdIFbYA)
- [RNN-Forward Prop](https://www.youtube.com/watch?v=u8utlK_c5C8)
- [RNN-Backward Prop](https://www.youtube.com/watch?v=6EXP2-d_xQA&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=18)

#### Articles
- [Visual Intuition of RNNs](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9)
- [Intuitive explanation of RNN - personal favorite](https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/)

#### Code
- [RNN from scratch in PyTorch](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/RNN/RNN-from-scratch.ipynb)
- [PyTorch RNN example](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/RNN/PyTorch_RNN.ipynb)

### LSTM

#### Videos
- [Krish Naik](https://www.youtube.com/watch?v=rdkIOM78ZPk&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=19&ab_channel=KrishNaik)
- [Visual Interpretation](https://www.youtube.com/watch?v=8HyCNIVRbSU)
  
#### Articles
- [Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [An illustrated guide to LSTM's and GRU's](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
- [Animated RNN, LSTM, GRU](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45)  
  
#### Code


### Bidirectional RNN

#### Videos
- [B-RNN (overview) - Krish Naik](https://www.youtube.com/watch?v=D-a6dwXzJ6s&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=24&ab_channel=KrishNaik)

#### Articles
- [Bi-Directional RNN - Paperspace Blog](https://blog.paperspace.com/bidirectional-rnn-keras/)

#### Code
- [Generating Baby names with Bidirectional-RNN](https://github.com/Gladiator07/Natural-Language-Processing/blob/main/LSTM/text_generator_charachter_level_LSTM.ipynb)

---

### Sequence to Sequence Learning 

- Will add resources here soon (forgot to add while learning :( )

---

### Attention Mechanism

#### Articles
- [Seq to Seq Models with Attention - Jay Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Attention in detail](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [Attention Mechanism in Deep Learning](https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/)

#### Code
- [FloydHub Article](https://blog.floydhub.com/attention-mechanism/)
- [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

---

### Transformers

#### Articles
- [Jay Alammar Blog](https://jalammar.github.io/illustrated-transformer/)