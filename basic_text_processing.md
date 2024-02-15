# Basic Text Processing


## Regular Expressions

A regular expression is a sequence of characters that define a search pattern

- **False Negatives**: prediction says it is false when in reality it is true
- **False Positives** prediciton says it is true when in reality it is False

| | True| False|
|--|--  |--|
|True|True Positives| False Positives |
|False|False Negatives | True Negatives|

Reducing the error rate for an application often involves two **antagonistic** efforts

- Increasing **accuracy** or **precision** (minimizing false positives)

- Increasing **coverage** or **recall** (minimizing false negatives)

Regular Expressions are often a first model for many text processing tasks. But as tasks get harder or increased performance, we use **machine learning classifiers**

# Lemmas, Wordforms, Types and Tokens

Dictionary Word Variations:
- **Capitalization**
- **Inflected forms**: the same word with a slight change to better fit the sentence's needs

Inflection is a process of word formation in which a word is modified to express different **gramatical categories**

- tense
- case
- voice 
- aspect
- person
- number
- gender
- mood
- animacy 
- definiteness

> An inflection expresses grammatical categories with affixation (such as prefix, suffix, infix, circumfix, and transfix), apophony (as Indo-European ablaut), or other modifications.[3] For example, the Latin verb ducam, meaning "I will lead", includes the suffix -am, expressing person (first), number (singular), and tense-mood (future indicative or present subjunctive). The use of this suffix is an inflection. In contrast, in the English clause "I will lead", the word lead is not inflected for any of person, number, or tense; it is simply the bare form of a verb. The inflected form of a word often contains both one or more free morphemes (a unit of meaning which can stand by itself as a word), and one or more bound morphemes (a unit of meaning which cannot stand alone as a word). For example, the English word cars is a noun that is inflected for number, specifically to express the plural; the content morpheme car is unbound because it could stand alone as a word, while the suffix -s is bound because it cannot stand alone as a word. These two morphemes together form the inflected word cars.

- **Lemma**: same stem, part-of-speech and word sense.
- **Wordforms**: include the inflected forms of words

So *cat* and *cats* are the same **lemma** but are different **wordforms** 

- **Types**: distinct words in the vocabulary used in a corpus so the set of words is the vocabulary and the type.
- **Token** an instance of a type in running text


# Corpora

The available text,or set of documents.

The corpora can have the following important characteristics:

- **Languages** and their **variations** 
- **Code Variation** in the middle switching languages
- **Genre**: scientific publication, pink magazines, religious texts, fiction...
- **Author Demographics**: age,sex,gender, socioeconomic class, era.
- **Time** as the language changes over the time.

# Sentence Segmentation

Text is typically splitted according to punctuantion `.?!` but we need to decide if the punctuation is part of the word or a sentence boundary marker.

# Text Normalization

It is usefull to reduce the vocabulary size and it helps machine learning models to generalize, though some of the technique can stip meaning from the words, like upper case `US` (United State) vs `us` (pronoun).

Converting text into a more convenient form using techniques such as:

- Case folding
- Lemmatization
- Stemming

# Word Tokenization

An initial approach of using `spaces` and `punctuation` with **Regular Expressions** is easy to think about but it quickly starts to get out of hand once:
- Abbreviation, compound words basically words that include punctuation
- Numbers, Number with decimal places, how to denote them
- Dates and Places
- Links, emails...

And certain languages don't use spaces to split words.


# Sub Word Tokenization

Main idea: What if we tokenize words by pieces?
Problems It solves:
- languages with no space splitting
- contracted forms
- Dealing with unknown forms of words if `lowest` and `low` appear in the trainning corpus but `lower` only appears in the test corpus.
- Robustness to misspeling
- Dealing with multi-lingual data 
- Reducing the vocabulary size 
- Capturing the shared meaning of parts of words (`suffixes` `prefixes`...)

# Multi Word Expressions

**Futebol Clube Do Porto** is a multiword expression, this problem is tied with **Named Entity Recognition** 

The solutions are:
- A Multi Word Expression Dictionary
- A statistical approach of detecting **frequently used n-grams** and sticking them together


# Lemmatization

Determining the **root**  of the word
- be:am,are,is
- "He is reading detective stories" -> "He be read detective story"

**Morphological Parsing**: words are built from morphemes -> result in inflected forms of the same word
- **Stem** the central morpheme of a word, supplying the main meaning
- **Affix**  adding additional meaning

- cats is an inflected form: cat(stem) + s(suffix) denoting plural
- iremos is an inflected form: ir(stem) + 1 person plural + future tense morphologic features

# Stemming

Lemmatization algorithms can be complicated

**Stemming**  is a simpler and cruder method that simply **cuts off word final affixes** it is subject to over or undergeneralization

These stemmers just cut off affixes thererfore `organs`  and `organization` will have the same **stem**  therefore the model will overgeneralize

There are special Stemmers for various languages like the **Porter Stemmer**  for English
```text
ational -> ate (configurational configurate)
ING -> epsilon (helping -> help)
sses -> ss (grasses -> grass)
```

