
## Installing the requirements

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Duvidas

Why Sub word tokenization helps with:
- Robustness to misspeling
- Dealing with multi-lingual data 
- Reducing the vocabulary size 

Sub-Word Tokenization Algorithms/ Byte Pair Encoding

## Use

- `MWETokenizer` and provide a Multi Word Expression Dictionary
- Use a subword tokenizer that detects contracted forms

