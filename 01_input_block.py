# Import necessary libraries
import torch

# Device assignment
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tiny_shakespeare data file.
with open('tiny_shakespeare.txt', 'r') as f:
    data = f.read()

# Prepare vocabulary by taking all unique characters from the data
vocab = sorted(list(set(data)))

# Add additional tokens for LLaMA 3 model
vocab.extend(['<|begin_of_text|>', '<|end_of_text|>', '<|pad_id|>'])
vocab_size = len(vocab)

# Create mapping between characters and integer indexes in the vocabulary
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

# Tokenizer encode function: convert string to list of integers
def encode(s):
    return [stoi[ch] for ch in s]

# Tokenizer decode function: convert list of integers back to string
def decode(l):
    return ''.join(itos[i] for i in l)

# Token variables for training
token_bos = torch.tensor([stoi['<|begin_of_text|>']], dtype=torch.int, device=device)
token_eos = torch.tensor([stoi['<|end_of_text|>']], dtype=torch.int, device=device)
token_pad = torch.tensor([stoi['<|pad_id|>']], dtype=torch.int, device=device)

# Example prompt
prompts = "Hello World"
encoded_tokens = encode(prompts)
decoded_text = decode(encoded_tokens)

# Test (remove comments to test the code):
# print(f"Lenth of shakespeare in characters: {len(data)}")
# print(f"Vocabulary: {''.join(vocab)}")
# print(f"Vocab size: {vocab_size}")
# print(f"Encoded tokens: {encoded_tokens}")
# print(f"Decoded text: {decoded_text}")

#Expected output :-
"""
Length of shakespeare in characters: 1115394
Vocabulary: !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz<|begin_of_text|><|end_of_text|><|pad_id|>
Vocab size: 68
Encoded tokens: [20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42]
Decoded text: Hello World
"""