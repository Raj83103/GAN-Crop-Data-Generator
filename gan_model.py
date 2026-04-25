import torch
import torch.nn as nn

# -------------------------
# Generator Model
# -------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(10, 32),   # input noise → hidden
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 7),    # output = 7 features
            nn.Tanh()            # keeps values stable
        )

    def forward(self, x):
        return self.model(x)


# -------------------------
# Discriminator Model
# -------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(7, 64),   # input features
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()        # output = real/fake
        )

    def forward(self, x):
        return self.model(x)