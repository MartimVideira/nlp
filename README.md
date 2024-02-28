
## Installing the requirements

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Duvidas

Why Sub word tokenization helps with:
- Robustness to misspelling
- Dealing with multi-lingual data 
- Reducing the vocabulary size 

Sub-Word Tokenization Algorithms/ Byte Pair Encoding

- Why do we need to use  **Multinomial Naive Bayes** in problems where we want **any-of** a set of classes? Instead of using the $\argmax$ why not pick all the classes whose probability surpasses a threshold?

## Use

- Use a subword tokenizer that detects contracted forms

