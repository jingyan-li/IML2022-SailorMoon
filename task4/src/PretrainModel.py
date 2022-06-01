
import torch.nn as nn
import torch.nn.functional as F

class PretrainEmbeddingNet(nn.Module):
    def __init__(self, input_features: int = 1000, out_channels: int = 128, dropout: float = 0.1, regress_head_input_channels: int = 16):
        super(PretrainEmbeddingNet, self).__init__()
        # Input is [BatchSize, N_features]
        # self.fc1 = nn.Sequential(
        #     nn.Linear(input_features, 800),
        #     nn.BatchNorm1d(800),
        #     nn.PReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(800, 600),
        #     nn.BatchNorm1d(600),
        #     nn.PReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(600, out_channels * 4),
        #     nn.BatchNorm1d(out_channels * 4),
        #     nn.PReLU()
        #                         )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(out_channels*4, out_channels*2),
        #     nn.BatchNorm1d(out_channels*2),
        #     nn.PReLU()
        #                         )
        # self.embedding = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(out_channels*2, out_channels),
        #                         )
        # self.embedding_norm = nn.Sequential(
        #     nn.BatchNorm1d(out_channels),
        #     nn.PReLU()
        # )
        # #### Regression layer
        # self.regress_head = nn.Sequential(
        #     nn.Linear(out_channels, regress_head_input_channels),
        #     nn.BatchNorm1d(regress_head_input_channels),
        #     nn.Tanh(),
        #     nn.Linear(regress_head_input_channels, 1),
        #                         )
        # self.layers = [self.fc1, self.fc2, self.embedding, self.embedding_norm, self.regress_head]
        # self.embedding_layers = [self.fc1, self.fc2, self.embedding]
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 800),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(),
            nn.Linear(800, 600),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(600, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 200),
            nn.Tanh(),
            nn.Linear(200,100),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(100,out_channels)
        )
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(out_channels, 1)
        )
        self.layers = [self.encoder, self.decoder]
        self.embedding_layers = [self.encoder]

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