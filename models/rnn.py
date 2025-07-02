"""
RNN Text Classifier
Author: Emmanuel Ojo
"""
import torch
import torch.nn as nn

class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_prob):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        dropout_rate = dropout_prob if num_layers > 1 else 0.0
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden) 