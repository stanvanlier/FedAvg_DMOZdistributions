import torch
from torch import nn
import os
import pkg_resources

ROOT_DIR = pkg_resources.resource_filename("src", "..")


class LogReg(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(LogReg, self).__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim

        idx2vec = torch.load(os.path.join(ROOT_DIR, "data/processed/idx2vec.pt"))

        self.e = nn.Embedding.from_pretrained(idx2vec)
        self.l = nn.Linear(self.seq_len * embed_dim, 13)

    def forward(self, x):
        h = self.e(x)
        z = torch.zeros((h.shape[0], self.seq_len * self.embed_dim,))
        h_len = h.shape[1] * h.shape[2]
        z[:, :h_len] = h.view((-1, h_len,))
        return self.l(z)


def make_LogRegAdam(lr=0.001, max_seqlen=165, embed_dim=50):
    model = LogReg(max_seqlen, embed_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model_name = "LogRegAdam-lr{}".format(lr)

    return model, criterion, optimizer, model_name
