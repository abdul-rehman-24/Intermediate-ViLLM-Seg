import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """Projects a feature vector into a shared embedding space."""
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x):
        return self.proj(x)
