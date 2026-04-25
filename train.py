import torch
import pandas as pd
from gan_model import Generator, Discriminator

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("dataset.csv")
data = torch.tensor(data.values, dtype=torch.float32)

# -------------------------
# Initialize Models
# -------------------------
G = Generator()
D = Discriminator()

# Loss + Optimizers
criterion = torch.nn.BCELoss()
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)

# -------------------------
# Training Loop
# -------------------------
epochs = 500

for epoch in range(epochs):

    # Random real data batch
    idx = torch.randint(0, len(data), (32,))
    real_data = data[idx]

    real_labels = torch.ones(32, 1)
    fake_labels = torch.zeros(32, 1)

    # -------------------------
    # Train Discriminator
    # -------------------------
    noise = torch.randn(32, 10)
    fake_data = G(noise)

    d_loss_real = criterion(D(real_data), real_labels)
    d_loss_fake = criterion(D(fake_data.detach()), fake_labels)

    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # -------------------------
    # Train Generator
    # -------------------------
    noise = torch.randn(32, 10)
    fake_data = G(noise)

    g_loss = criterion(D(fake_data), real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    # -------------------------
    # Print progress
    # -------------------------
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# -------------------------
# Save model
# -------------------------
torch.save(G.state_dict(), "generator.pth")
print("Training complete. Model saved!")