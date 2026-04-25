import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
real = pd.read_csv("dataset.csv")
fake = pd.read_csv("synthetic_data.csv")

st.title(" GAN Synthetic Data Dashboard")

# Dropdown
feature = st.selectbox("Select Feature", real.columns)

# Plot
fig, ax = plt.subplots()

sns.kdeplot(real[feature], label="Real", fill=True, ax=ax)
sns.kdeplot(fake[feature], label="Synthetic", fill=True, ax=ax)

ax.set_title(f"{feature} Distribution Comparison")

st.pyplot(fig)

# Show data tables
if st.checkbox("Show Data"):
    st.write("Real Data", real.head())
    st.write("Synthetic Data", fake.head())
