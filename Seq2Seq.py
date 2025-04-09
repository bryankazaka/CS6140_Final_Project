
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def tokenize(text): return str(text).lower().split()

def build_vocab(token_lists):
    counter = Counter(token for tokens in token_lists for token in tokens)
    vocab = {word: i+2 for i, (word, _) in enumerate(counter.items())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def encode(tokens, vocab, max_len):
    return [vocab.get(token, 1) for token in tokens[:max_len]] + [0] * (max_len - len(tokens))

MAX_INPUT_LEN = 10
MAX_OUTPUT_LEN = 3

# Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# Building encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

# Decoder with attention
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = attention

    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        attn_weights = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

# Seq2Seq model with attention
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size, max_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = max_len
        self.target_vocab_size = target_vocab_size

    def forward(self, src, trg):
        batch_size = src.size(0)
        encoder_outputs, hidden, cell = self.encoder(src)
        output_tensor = torch.zeros(batch_size, self.max_len, self.target_vocab_size)
        input_token = trg[:, 0]

        for t in range(1, self.max_len):
            out, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            output_tensor[:, t] = out
            input_token = trg[:, t]

        return output_tensor

def predict_title(track_name, model, input_vocab, target_vocab, max_input_len, max_output_len):
    model.eval()
    tokens = tokenize(track_name)
    input_seq = torch.tensor([encode(tokens, input_vocab, max_input_len)])

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(input_seq)
        input_token = torch.tensor([target_vocab.get('<PAD>', 0)])
        output_tokens = []

        for _ in range(max_output_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            probs = torch.softmax(output, dim=1)

            # Suppress PAD token probability entirely
            probs[0][0] = 0

            pred_token = torch.multinomial(probs, 1).item()
            word = next((k for k, v in target_vocab.items() if v == pred_token), '<UNK>')

            if output_tokens and word == output_tokens[-1]:
                continue  # Skip repeated words for output

            if pred_token == 0:
                break

            output_tokens.append(word)
            input_token = torch.tensor([pred_token])

    return ' '.join(output_tokens).title()