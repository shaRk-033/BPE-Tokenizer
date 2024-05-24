# Basic and GPT Tokenizer

This repository contains two classes, `BasicTokenizer` and `GPTTokenizer`, that implement basic tokenization techniques for text processing. These tokenizers can be used to encode and decode text based on a simple byte-pair encoding (BPE) algorithm. The `GPTTokenizer` class also includes a sophisticated text splitting method based on regular expressions, inspired by GPT-4's tokenization pattern.

## Table of Contents
- [Basic and GPT Tokenizer](#basic-and-gpt-tokenizer)
  - [Table of Contents](#table-of-contents)
  - [BasicTokenizer Class](#basictokenizer-class)
    - [Initialization](#initialization)
    - [Methods](#methods)
  - [GPTTokenizer Class](#gpptokenizer-class)
    - [Initialization](#initialization-1)
    - [Methods](#methods-1)
  - [Usage](#usage)
  - [Installation](#installation)
  - [License](#license)

## BasicTokenizer Class

The `BasicTokenizer` class provides a simple implementation of a byte-pair encoding (BPE) tokenizer.

### Initialization

```python
def __init__(self, vocab_size):
```
- `vocab_size`: The desired vocabulary size after training.

### Methods

- `freq(tokens)`: Computes frequency statistics of bigrams in the token list.
- `replace(tokens, pair, idx)`: Replaces occurrences of a bigram pair with a new token index.
- `train(txt)`: Trains the tokenizer on the given text.
- `encode(txt)`: Encodes a text string into a list of tokens.
- `decode(tokens)`: Decodes a list of tokens back into a text string.

## GPTTokenizer Class

The `GPTTokenizer` class extends the basic BPE tokenizer with a more advanced text splitting method using regular expressions, aiming to emulate GPT-4's tokenization strategy.

### Initialization

```python
def __init__(self, vocab_size):
```
- `vocab_size`: The desired vocabulary size after training.

### Methods

- `split_text(text)`: Splits the text into chunks using a regular expression pattern.
- `freq(tokens, stats)`: Computes frequency statistics of bigrams in the token list and updates the given stats dictionary.
- `replace(tokens, pair, idx)`: Replaces occurrences of a bigram pair with a new token index.
- `train(text)`: Trains the tokenizer on the given text.
- `_encode_chunk(tokens)`: Encodes a chunk of tokens by repeatedly replacing bigrams.
- `encode(text)`: Encodes a text string into a list of tokens.
- `decode(tokens)`: Decodes a list of tokens back into a text string.

## Usage

Here's an example of how to use both tokenizers:

```python
# Initialize the tokenizers
basic_tokenizer = BasicTokenizer(vocab_size=300)
gpt_tokenizer = GPTTokenizer(vocab_size=300)

# Train the tokenizers
text = "This is an example sentence for tokenization."
basic_tokenizer.train(text)
gpt_tokenizer.train(text)

# Encode the text
basic_encoded = basic_tokenizer.encode(text)
gpt_encoded = gpt_tokenizer.encode(text)

# Decode the tokens back to text
basic_decoded = basic_tokenizer.decode(basic_encoded)
gpt_decoded = gpt_tokenizer.decode(gpt_encoded)

# Output the results
print("Basic Tokenizer Encoded:", basic_encoded)
print("Basic Tokenizer Decoded:", basic_decoded)
print("GPT Tokenizer Encoded:", gpt_encoded)
print("GPT Tokenizer Decoded:", gpt_decoded)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
