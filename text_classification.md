# Text Classification

The task of **text classification** aims to classify text according to a number of classes.

- Spam detection
- Sentiment analysis
- Assign subject categories, topics or genres
- Authorship identification form a closed list
- Age/gender classification
- Language detection

Input:
- A document *d*
- A fixed set of classes $C =\set{c_1,c_2,...,c_n}$

Output:
- The predicted class $c \in C$ for document $d$

## Hand-coded Rules

Hand-coded rules are rules based on combinations of words or other features
- In *spam detection*: **black-list** of addresses and **keyword detection**
- Sentiment analysis : ratio of **word-polarities** appearing in a **sentiment lexicon**

The accuracy provided by this method can be high if the rules are carefully revised by an expert,although this process can be very tiresome and expensive.

## Supervised Machine Learning

Making use of **annotated datasets** through Machine Learning Algorithms

### Building a model

Input:
- A set of fixed classes $C = \set{c_1,c_2,...,c_n}$
- A training set of **m** hand-labeled documents $\set{(d_1,c_1),(d_2,c_2),(d_n,c_n)}$ where $d_i \in D \land c_i \in C$

Output Classified: $\gama: D \rarr C$
- a mapping from document to classes (or class probabilities)

## Classifiers

A **classifier**, given a document assigns a class or class probability to it

**Probability Classifier**, more than predicting one class, it assigns a probability of the input document belonging to each of classes.


**Generative Classifiers** build a model of how a class could generate some input data
- Given an observation, returns the class that has most likely produced the  input observation. Ex. *Naive Bayes*

**Discriminative Classifiers** learn what features from the input are most useful to discriminate between the different possible classes.
- Ex: Decision Trees (DTS), Logistic Regression (LR), Support Vector Machines (SVM)

## Bag Of Words Model

Machine Learning models require the data to be represented as a set of **features**, in a tabular way, but in NLP we are dealing with text, unstructured data, we need a way to massage the text into a format so that a Machine Learning Model can consume it.

- We thus need a way of going from document *d* to a vector of features *X*

The **bag of words** model is an unordered set of words only keeping their frequency in the document. It assumes that the postion of each word does not matter.

## Naive Bayes

Naive Bayes (NB) makes a simplifying (naive) assumption about how the features interact

Bayes Rule:

$$P(c|d) = \frac{P(d|c)P(c)}{P(d)}$$

Most Likely Class:

$$\hat{c} = \argmax_{c \in C} P(c|d) = \argmax_{c \in C} \frac{P(d|c)P(c)}{P(d)}$$

As $P(d)$ is a common factor of $\frac{P(d|c)P(c)}{P(d)}, \forall c \in C$, maximizing this expression is the same as maximizing $P(d|c)P(c)$

$$ \argmax_{c \in C} P(d|c)P(c)$$

## Naive Bayes Classifier

Representig a document with a set of features

$$ d = \set{f_1,f_2,...,f_n} \rarr \hat{c} = \argmax_{c \in C} P(f_1,f_2,...,f_n|c)P(c)$$  

Assuming conditional independence, i.e. features are not dependent on each other. In this context it would mean that the appearence of a word has no influency in the appearence of another word.

$$P(f_1,f_2,...,f_n|c) = P(f_1|c) P(f_2|c) .... P(f_n|c)$$

So we get a Naive Bayes Classifier:

$$ c_{NB} = \argmax_{c \in C} P(c)\prod_{f \in F} P(f|c)$$


For all the tokens in the test document we would calculate the Naive Bayes Classifier of that document.

Instead of multiplying all probabilities we can **sum in the log space** to avoid numerical underflow like it was previously discussed

$$ c_{NB} = \argmax_{c \in C} \log P(c) \sum_{t \in document} \log P(t|c)$$

Highest log probaility class is still the most likely call, has the logarithm is a strictly crescent function.

Probability of a document belonging to a class:

$$\hat{P}(c) = \frac{N_c}{N_doc}$$


Word probability per class

$$ \hat{P}(w_i |c) = \frac{count(w_i,c)}{\sum_{k \in V} count(w_k,c)}$$

Where $count(w,c)$ returns the occurances of the word $w$ in documents of class $c$.

Handling non-occuring words in a class, with **add-one Laplace Smoothing**

$$\hat{P}(w_i|c) = \frac{count(w_i,c) + 1}{V + \sum_{k\in V} count(w_k,c)}$$


## Naive Bayes... Not so Naive

- Very Fast, low storage requirements
- Robust to irrelevant features: they tend to cancel eachother without affecting results
- Very good in domains with equally important features, where Decision Trees suffer with fragmentation

- It is optimal if the **assumed independece assumptions hold**

It is used as a good **baseline** to put other models in comparison.