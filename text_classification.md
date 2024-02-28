# Text Classification

The task of **text classification** aims to classify text according to a number of classes.

- Spam detection
- Sentiment analysis
- Assign subject categories, topics or genres
- Authorship identification from a closed list
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

Output Classified: $\gamma: D \rarr C$
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

Naive Bayes (NB) makes a simplifying (naive) assumption about how the features interact, that features are independent from eachother, that we are dealing with independent events.

Bayes Rule:

$$P(c|d) = \frac{P(d|c)P(c)}{P(d)}$$

Most Likely Class:

$$\hat{c} = \argmax_{c \in C} P(c|d) = \argmax_{c \in C} \frac{P(d|c)P(c)}{P(d)}$$

As $P(d)$ is a common factor of $\frac{P(d|c)P(c)}{P(d)}, \forall c \in C$, maximizing this expression is the same as maximizing $P(d|c)P(c)$ so:

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

$$ c_{NB} = \argmax_{c \in C} \log P(c) + \sum_{t \in document} \log P(t|c)$$

Highest log probaility class is still the most likely class, has the logarithm is a strictly crescent function.(So we don't need to convert into the real probability.)

Probability of a document belonging to a class:

$$\hat{P}(c) = \frac{N_{\text{docs of class c}}}{N_{docs}}$$


Word probability per class

$$ \hat{P}(w_i |c) = \frac{count(w_i,c)}{\sum_{k \in V} count(w_k,c)}$$

Where $count(w,c)$ returns the number of occurances of the word $w$ in documents of class $c$.

Handling non-occuring words in a class, with **add-one Laplace Smoothing**

$$\hat{P}(w_i|c) = \frac{count(w_i,c) + 1}{V + \sum_{k\in V} count(w_k,c)}$$


## Naive Bayes... Not so Naive

- Very Fast, low storage requirements
- Robust to irrelevant features: they tend to cancel eachother without affecting results
- Very good in domains with equally important features, where Decision Trees suffer with fragmentation

- It is optimal if the **assumed independece assumptions hold**

It is used as a good **baseline** to put other models in comparison.

### Example Sentimental Analysis with Naive Bayes

| Category | Documents                             |
| -------- | ------------------------------------- |
| -        | just plain boring                     |
| -        | entirely predictable and lacks energy |
| -        | no surprises and very few laughs      |
| +        | very powerfull                        |
| +        | the most fun film of the summer       |
| test ?   | predictable with no fun               |

Class Distribution (Prior Distribution)

$$
P(-) = \frac{N_{\text{docs of class -}}}{N_{docs}} = \frac{3}{5} = 0.6 \\
P(+) = \frac{2}{5} = 1 - P(-) = 0.4
$$

Now let $S$ be the sentence we want to test, we will  calculate $P(S|+)$ and $P(S|-)$ and will choose the class that maximizes the probability, this is exaclty what the $\argmax$ does.

$$\hat{c} = \argmax_{c \in C} P(c)P(S|c)$$

And we are doing this using the **Naive Bayes** model which states that the features are independent, and therefore $P(f_1 \land f_2) = P(f_1)P(f_2)$.

$$hat{c} = \argmax_{c \in C} P(c)P(f_1 f_2...f_n |c) = \argmax_{c \in C}P(c)\prod_{token \in doc} P(t|c)$$
In the task of text classification the features are tokens of the document.

Now we just need to calculate $P(t|c)$ which can  be done by calculating the distribution of $t$ accross all tokens of $c$

$$P(t|c) = \frac{count(t,c)}{\sum_{t_i \in V}count(t_i,c)}$$

With Laplace 1 smoothing:

$$P(t|c) = \frac{count(t,c)+1}{|V| + \sum_{t_i \in V}count(t_i,c)}$$

So lets calculate for the negative class $|V| = 20 \land \sum_{t_i \in V}count(t_i,-) = 14$
$$
\begin{align*}
& P(predictable | -) = \frac{1+1}{14 + 20} = \frac{2}{34} = \frac{1}{17} \\

& P(with | -) = \frac{0+1}{14+20} = \frac{1}{34} \\

& P(no | -) = \frac{1+1}{14+20} = \frac{1}{17} \\

& P(fun | -) = \frac{0+1}{14+20} = \frac{1}{34} \\

& P(-)P(S|-) = 0.6 \times {\frac{1}{17}}^2 \times {\frac{1}{34}}^2 = 6.1\times 10^{-5}
\end{align*}
$$

Doing the same for the positive class we would get: $P(+)P(S|+) = 3.2\times 10^{-5}$

Therefore the chosen class would be the negative class!

Remember we are calculating $P(c|S)$ because we have the sentence and we want to know the class! And we use the Naive Bayes theorem to use the training data to calculate $ \forall c \in C , P(c|S)$  and we chose the class that maximizes this probability.


