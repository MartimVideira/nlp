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


## Word Occurance vs Word Frequency

>In how many documents of the class does the word occur?

This can even be more important than word frequency

In Binary Naive Bayes we use **binary counts** meaning that we just count the
occurrance of a word in a document, not its frequency. Therefore its
$count(w,c)$ will be equal to the number of documents of class $c$ where the
word $w$ appears.

## Dealing with negation

Looking at the sentences:
- I like this movie.
- I *did not* like this movie.

Despite meaning the complete opposite in respect to tokens these two sentences are almost the same.

In sentiment analysis we have the problem of **dealing with negation**. One way of handling negation is to
change the tokens that occur after a negation: `I did not NOT_like NOT_this  NOT_movie` there fore in terms of tokens
these two sentences would be rather different improving the capabilities of a model using Naive Bayes

The usage of **bigrams** instead of single words like we've been discussing the Naive Bayes model. It would capture the meaning of **not like**.

## Making Use Of Lexicons

Lexicons can provide external knowledge that can be useful in this task.

**Sentiment lexicons** are lists of words that capture their positive/negative **polarity**. 

With the help sentiment lexicons a **polarity** can be calculated for a given sentence, helping in the text classification task.

Other features can be used in the task of classification, like the detection of certain **keywords** or **expressions** accross the corpus.

features like the metadata of the text if it is of a format other than raw text, Or even syntatic features of the text like the use of punctuation, paragraphs if the words are capitalized or not, the use of slang...

- Distinguishing a boomer from a zoomer.

## Ngrams Come to the Help of Naive Bayes... Maybe

We have seen that the bigger the n-grams the greater context they provide.

And we've also seen that the Naive Bayes assumption that features/words are independent of eachother can hinder us.

So if we instead of single words use n-grams we can use the Naive Bayes Model,
while maintaining more context than if we used just single words. Like understanding **not like** .

This comes with the cost that the feature set will be very sparse, some n-grams present in the testing data will never have
appeared in the trainning data... Text is creative...

## Classification with More than 2 Classes

- Sentiment analysis with `Very Negative, Negative, Indiferent, Positive, Very Positive`
- Topic classification (not a binary decision) where a document can have more than 1 topic

So we enter the topic of **Multi-class** classification tasks:
- **Multi-label (any-of)** classification where each item can be assigned more than one label.
- **Multinomial (one-of)** classification, classes are mutually exclusive, only one classification.

So the sentiment analysis would be **multinomial** and the topic classification would be **multi-label**

- **Any of**: one binary classifier for each class, this may assign multiple labels
- **One of**: use a multiclass classifier like the multinomial **Naive Bayes** model we've seen or a Multinomial Logistic Regression
    - Or run a **binary classifier** for all the possible classes and choose the one with higher probability.

Where a **binary classifier** can tell us the probability of a the subject being testes belongin to the class.

We can run a **binary classifier** for each class and have a **threshold**. We accept that the document is from a class if the result from the classifier is greater than the **threshold** therefore we can use a **binary classifier** to get **any of** a set of classes.


