import torch
import pandas as pd
from gan_model import Generator

# -------------------------
# Load trained model
# -------------------------
G = Generator()
G.load_state_dict(torch.load("generator.pth"))
G.eval()  # important for inference

# -------------------------
# Generate fake data
# -------------------------
noise = torch.randn(100, 10)   # 100 fake samples
fake_data = G(noise).detach().numpy()

# scale to realistic range
fake_data[:,0:3] *= 100   # N, P, K
fake_data[:,3] = fake_data[:,3]*15 + 20   # temperature
fake_data[:,4] = fake_data[:,4]*50 + 40   # humidity
fake_data[:,5] = fake_data[:,5]*2 + 5.5   # pH
fake_data[:,6] = fake_data[:,6]*150 + 50  # rainfall

# -------------------------
# Convert to DataFrame
# -------------------------
columns = ["N","P","K","temperature","humidity","ph","rainfall"]
df = pd.DataFrame(fake_data, columns=columns)

# -------------------------
# Save to CSV
# -------------------------
df.to_csv("synthetic_data.csv", index=False)

print("✅ Synthetic data generated and saved!")