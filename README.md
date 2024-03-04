
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

Gradient Descent:
- In the gradient descent why is each partial derivative with respect to $w_i = (\hat{y} - y) x_i$, why is it scaled by the input feature?
- In the gradient descent is should it be $L_{CE}$ or the average of $L_{CE}$ for all data points?
- So calculating the partial derivative is just wiggling the weights and bias and recalculating $\hat{y}$?
## Use

- Use a subword tokenizer that detects contracted forms

- Copy Slide 17 into my notes.


