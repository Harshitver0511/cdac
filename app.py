import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import json
import re

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Parameters
MAX_LEN = 50  # Same max length as used during training

# Tokenizer
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return text.split()

# Encode function
def encode(text, vocab, max_len=MAX_LEN):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab.get('<PAD>', 0)] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# Model definition (must match your training model)
class EmailClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # note hidden_dim*2 here

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        # Concatenate the last forward and backward hidden states
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.fc(h_n)
        return out


# Load model
model = EmailClassifier(len(vocab))
model.load_state_dict(torch.load('email_classifier.pth', map_location=device))
model.to(device)
model.eval()

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email_text = data.get('email', '')
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400

    encoded = torch.tensor([encode(email_text, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(encoded)
        pred = torch.argmax(output, dim=1).item()
    label = "Phishing Email" if pred == 1 else "Safe Email"
    return jsonify({'prediction': label})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

