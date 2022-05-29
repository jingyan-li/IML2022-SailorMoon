
import torch.nn as nn
import torch.nn.functional as F

class PretrainEmbeddingNet(nn.Module):
    def __init__(self, input_features: int = 1000, out_channels: int = 128, dropout: float = 0, regress_head_input_channels: int = 16):
        super(PretrainEmbeddingNet, self).__init__()
        # Input is [BatchSize, N_features]
        self.fc1 = nn.Sequential(
            nn.Linear(input_features, out_channels*4),
            nn.BatchNorm1d(out_channels*4),
            nn.PReLU()
                                )
        self.fc2 = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Linear(out_channels*4, out_channels*2),
            nn.BatchNorm1d(out_channels*2),
            nn.PReLU()
                                )
        self.embedding = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Linear(out_channels*2, out_channels),
                                )
        self.embedding_norm = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )
        #### Regression layer
        self.regress_head = nn.Sequential(
            nn.Linear(out_channels, regress_head_input_channels),
            nn.BatchNorm1d(regress_head_input_channels),
            nn.PReLU(),
            nn.Linear(regress_head_input_channels, 1)
                                )
        self.layers = [self.fc1, self.fc2, self.embedding, self.embedding_norm, self.regress_head]
        self.embedding_layers = [self.fc1, self.fc2, self.embedding]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def get_embedding(self, x):
        out = x
        for layer in self.embedding_layers:
            out = layer(out)
        return out