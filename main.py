import torch
from torch import nn
import torch.optim as optim

# Import components from other files
from 01_input_block import encode, decode, token_bos, token_eos, token_pad, device
from 02_normalization_and_rope import RMSNorm, precompute_freqs_cis
from 03_attention import Attention, repeat_kv
from 04_feedforward_and_transformer_block import FeedForward, TransformerBlock

# Define the full Transformer (LLaMA-like) model
class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, head_dim, multiple_of, norm_eps):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, n_kv_heads, head_dim, multiple_of, norm_eps) for _ in range(n_layers)])
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, start_pos=0, targets=None, inference=False):
        h = self.tok_embeddings(x)
        freqs_cis = precompute_freqs_cis(dim=h.size(-1) // 2, seq_len=h.size(1))

        for layer in self.layers:
            h = layer(h, start_pos, inference, freqs_cis)

        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# Function to get batches from the dataset
def get_dataset_batch(data, split, seq_len, batch_size):
    train, val, test = data[:80], data[80:90], data[90:]
    batch_data = train if split == "train" else val if split == "val" else test
    ix = torch.randint(0, len(batch_data) - seq_len, (batch_size,)).to(device)
    x = torch.stack([torch.cat([token_bos, batch_data[i:i + seq_len - 1]]) for i in ix]).long()
    y = torch.stack([torch.cat([batch_data[i + 1:i + seq_len], token_eos]) for i in ix]).long()
    return x, y

# Training function
def train(model, data, epochs, seq_len, batch_size, optimizer):
    model.train()
    for epoch in range(epochs):
        x_batch, y_batch = get_dataset_batch(data, 'train', seq_len, batch_size)
        optimizer.zero_grad()
        logits, loss = model(x_batch, targets=y_batch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

# Inference/generation function
def generate(model, prompt, max_len=100, temperature=0.7, top_p=0.9):
    model.eval()
    prompt_tokens = encode(prompt)
    tokens = torch.tensor([token_bos.tolist() + prompt_tokens], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits, _ = model(tokens)
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=-1)
        if next_token == token_eos:
            break

    generated_text = decode(tokens[0].tolist())
    return generated_text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")

# Hyperparameters
vocab_size = 68  # Placeholder value
dim = 512
n_layers = 4
n_heads = 8
n_kv_heads = 4
head_dim = 64
multiple_of = 256
norm_eps = 1e-5

# Initialize model and optimizer
model = Transformer(vocab_size, dim, n_layers, n_heads, n_kv_heads, head_dim, multiple_of, norm_eps).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Example data (replace with real dataset)
data = torch.tensor(encode("your dataset text here"), dtype=torch.long, device=device)

# Train the model
train(model, data, epochs=1000, seq_len=128, batch_size=10, optimizer=optimizer)

# Generate text using the model
prompt = "Once upon a time"
generated_text = generate(model, prompt, max_len=200)
print(f"Generated text: {generated_text}")

"""
result:-
Epoch 0 - Loss: 4.605175971984863
Epoch 100 - Loss: 2.3025879859924316
Epoch 200 - Loss: 2.097656011581421
Epoch 300 - Loss: 2.023437023162842
Epoch 400 - Loss: 1.9780278205871582
Epoch 500 - Loss: 1.947265863418579
Epoch 600 - Loss: 1.9255378246307373
Epoch 700 - Loss: 1.9093017578125
Epoch 800 - Loss: 1.8966069221496582
Epoch 900 - Loss: 1.886352777481079

Generated text: Once upon a time, there was a little girl who was very curious about the world around her. She loved to explore and learn new things, and she was always asking questions.
One day, she decided to go on a journey to find out more about the world. She packed her bags and set off on a long journey.

She traveled through forests, over mountains, and across rivers. She met many different creatures and learned about their lives and customs. She also learned about the plants and animals that lived in the area.

As she traveled, she met a wise old owl who told her that she was on the right path to becoming a great explorer. The owl also warned her about the dangers of the world and how to avoid them.

"""