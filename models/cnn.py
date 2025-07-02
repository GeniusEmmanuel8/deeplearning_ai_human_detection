"""
CNN Text Classifier
Author: Emmanuel Ojo
"""
import torch
import torch.nn as nn

class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, num_classes, dropout_prob):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.batch_norm = nn.BatchNorm1d(len(kernel_sizes) * num_filters)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).unsqueeze(1)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.batch_norm(cat)
        cat = self.dropout(cat)
        return self.fc(cat) 