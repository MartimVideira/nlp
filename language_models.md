# Language Models

In this document we will talk about:

- N-grams
- Markov Assumption
- Evaluation of Language Models
- Perplexity
- Smoothing

## Probabilistic Language Models

When we talk about Language Models (LM) we refer to **probabilistic language models** that assign probabilies to a sequence of words.

1. Given a sequence of words, predict the next word by assigning a **probability to each possible word in the vocabulary**
1. Or assign a probability to an entire sequence (*sentence*) to see how likely it is to be generated.

## Probability of a Word given a History

The probability of having a word *w* given an history h

$$ P(w|h)$$

$$P(\text{you}|\text{I love}) = \frac{C(\text{I love you})}{C(\text{I love})} $$

We make use of the frequency of these possible sentences, but this is likely to
produce bad results as if we got a word that didn't exist in our dataset we
would get a probability of zero, meaning that this model is not creative cannot,
like language create new knowledge.


## Probability of a Sequence

The probability of the sentence `I drank coffee` is given by:

$$P(\text{I}) \times P(\text{drank} | \text{I}) \times P(\text{coffee} | \text{I drank})$$

So given a sequence of *N* words $w_1 ... w_n$ lets call it $w_1^n$ we can say that the probability of that sequence of words $P(w_1^n)$

$$P(w_1^n) = P(w_1)P(w_2|w_1)P(w_3|w_1^2)...P(w_n|w_1^{n-1})$$

$$\Pi_{k=1}^{n} P(w_k|w_1^{k-1})$$
Computing the probability of a word given its entire history is hard, we would like to be able to use just the last few words. This will result in an aproximation, and  the **model will lose some context** but it can speed up calculations.

## N-Grams

An N-Gram is a sequence of n words.

Given the sentence `Please turn your homework` would be split into:

- 3 **bigrams** `please turn`, `turn your`, `your homework`.
- And 2 **trigrams**: `please turn your`, `turn your homework`.

N-gram Models can be used to estimate the probability of the last word of an n-gram given the previous n-1 words.

## Markov Models

The **Markov Assumption** states that we can predict the probabilty of some future unit without looking too far into the past.

**Bigram Model** approximate $P(w_n|w_1^{n-1})\sim P(w_n|w_{n-1})$ using only the preceding word!. So the Probability of seeing `the` after seeing `its water is so transparent that`  would be approximate to the probability of seeing `the` after `that`

- Probability of a sequence based on **bigrams**: $P(w_1)P(w_2|w_1)P(w_3|w_2)...P(w_n|w_{n-1})$
$$P(w_1^n) \sim \Pi_{k=1}^{n}P(w_k|w_{k-1})$$

Generalizing to any **n-gram** model, the intuition is instead of all the words of the sequence we just care about the words inside the **n-gram** using the markov assumption.

For a sequence of $n$ tokens using the n-gram model of size $t$

$$P(w_1^n) \sim \Pi_{k=1}^{n}P(w_k|w_{k-t+1}^{k-1})$$

$$w_{k - t +1}{k-1}$$

| Index | Token       |
| ----- | ----------- |
| 1     | its         |
| 2     | water       |
| 3     | is          |
| 4     | so          |
| 5     | transparent |
| 6     | that        |

Using a **tri-gram model** where $t=3$ if we wanted to predict the sequence `its water [is]`

Here $k=1$ then  
$$P(w_1^3) \sim P(w_3|w_{3-3+1}^{3-1})$$
Doing the math:
$$P(w_1^3) \sim P(w_3|w_{1}^{2})$$
Substituting the sequences in:
$$P(\text{its water is}) \sim P(\text{is}|\text{its water})$$

## Maximum Likelihood Estimation (MLE)

Getting normalized counts, relative frequencies from a corpus ,i.e the probability

Bigram Model: 
$$P(w_n|w_{n-1}) = \frac{C(w_{n-1}w_n)}{C(w_{n-1})}$$

N-Gram Model

$$P(w_n|w_{n-N+1}^{n-1}) = \frac{C(w_{n-N+1}^{n-1}w_n)}{C(w_{n-N+1})}$$

I wrote them myself so I *think* i understand them?

Given the mini-corpus: 
```
<s>I am Sam</s>
<s>Sam I am</s>
<s>I do not like green eggs and ham</s>
```
Where the tokens `<s>` and `</s>` represent the start and end of the sentence.

Tokens: `<s>I`,`I am`, `am Sam`, `Sam <s/>`, `<s>Sam`, `Sam I`,`am </s>`,`I do`, `do not`, `not like`,`like green`, `green eggs` ,`eggs and`,`and ham`, `ham </s>`


