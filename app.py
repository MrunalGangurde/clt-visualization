import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, norm
import time

# Title and Sidebar
st.title("Central Limit Theorem Visualization")
st.sidebar.header("Settings")

# User Input for Population Distribution
dist_options = {
    "Normal": np.random.normal,
    "Exponential": np.random.exponential,
    "Uniform": np.random.uniform,
    "Poisson": np.random.poisson,
    "Binomial": np.random.binomial,
    "Log Normal": np.random.lognormal,
    "Beta": np.random.beta,
    "Student T": np.random.standard_t,
    "Chi Squared": np.random.chisquare
}
dist_name = st.sidebar.selectbox("Population Distribution:", list(dist_options.keys()))

sample_size = st.sidebar.slider("Sample size:", 2, 500, 30)
num_samples = st.sidebar.slider("Number of samples:", 100, 10000, 1000)
bins = st.sidebar.slider("Number of bins:", 10, 100, 50)

# Distribution Parameters
if dist_name in ["Normal", "Log Normal"]:
    param1 = st.sidebar.number_input("Mean:", value=0.0)
    param2 = st.sidebar.number_input("Standard Deviation:", value=1.0, min_value=0.01)
elif dist_name == "Exponential":
    param1 = st.sidebar.number_input("Rate:", value=1.0, min_value=0.01)
elif dist_name == "Uniform":
    param1 = st.sidebar.number_input("Lower Bound:", value=0.0)
    param2 = st.sidebar.number_input("Upper Bound:", value=1.0)
elif dist_name == "Poisson":
    param1 = st.sidebar.number_input("Lambda:", value=1.0, min_value=0.01)
elif dist_name == "Binomial":
    param1 = st.sidebar.number_input("Trials:", value=10, min_value=1)
    param2 = st.sidebar.number_input("Probability of Success:", value=0.5, min_value=0.0, max_value=1.0)
elif dist_name == "Beta":
    param1 = st.sidebar.number_input("Alpha:", value=2.0, min_value=0.01)
    param2 = st.sidebar.number_input("Beta:", value=2.0, min_value=0.01)
elif dist_name == "Student T":
    param1 = st.sidebar.number_input("Degrees of Freedom:", value=10, min_value=1)
elif dist_name == "Chi Squared":
    param1 = st.sidebar.number_input("Degrees of Freedom:", value=10, min_value=1)

# Generate Population
if dist_name in ["Normal", "Log Normal"]:
    population = dist_options[dist_name](param1, param2, 100000)
elif dist_name == "Exponential":
    population = dist_options[dist_name](1 / param1, 100000)
elif dist_name == "Uniform":
    population = dist_options[dist_name](param1, param2, 100000)
elif dist_name == "Poisson":
    population = dist_options[dist_name](param1, 100000)
elif dist_name == "Binomial":
    population = dist_options[dist_name](param1, param2, 100000)
elif dist_name == "Beta":
    population = dist_options[dist_name](param1, param2, 100000)
elif dist_name == "Student T":
    population = dist_options[dist_name](param1, 100000)
elif dist_name == "Chi Squared":
    population = dist_options[dist_name](param1, 100000)

# Sampling
samples = np.random.choice(population, (num_samples, sample_size))
sample_means = samples.mean(axis=1)

# Population Distribution
st.header("Population Distribution")
fig, ax = plt.subplots()
sns.histplot(population, kde=True, bins=bins, ax=ax)
ax.axvline(np.mean(population), color='red', linestyle='--', label='Population Mean')
ax.set_title("Population Distribution")
ax.legend()
st.pyplot(fig)

# Distribution of Sample Means
st.header("Distribution of Sample Means")
fig, ax = plt.subplots()
sns.histplot(sample_means, kde=True, bins=bins, ax=ax)
ax.axvline(np.mean(sample_means), color='green', linestyle='--', label='Sample Mean')
ax.axvline(np.mean(population), color='red', linestyle='--', label='Population Mean')
ax.set_title("Distribution of Sample Means")
ax.legend()
st.pyplot(fig)

# Theoretical Normal Curve
st.header("Theoretical Normal Curve Comparison")
fig, ax = plt.subplots()
sns.histplot(sample_means, kde=True, bins=bins, ax=ax, label="Sample Means")
x = np.linspace(min(sample_means), max(sample_means), 1000)
y = norm.pdf(x, np.mean(sample_means), np.std(sample_means))
ax.plot(x, y * len(sample_means) * (bins / (max(sample_means) - min(sample_means))), color='orange', label="Theoretical Curve")
ax.set_title("Theoretical Curve vs Sample Means")
ax.legend()
st.pyplot(fig)

# QQ Plot
st.header("QQ Plot of Sample Means")
fig, ax = plt.subplots()
probplot(sample_means, dist="norm", plot=ax)
ax.set_title("QQ Plot")
st.pyplot(fig)

# Galton Board Visualization
st.header("Visualization")
cols = st.columns(sample_size)

# Initialize counts and animation
num_particles = st.sidebar.slider("Number of Particles:", 100, 5000, 500)
animation_speed = st.sidebar.slider("Animation Speed (ms):", 10, 200, 50)
bin_counts = np.zeros(sample_size + 1, dtype=int)

placeholder = st.empty()

for particle in range(num_particles):
    position = 0
    for _ in range(sample_size):
        position += np.random.choice([-1, 1])  # Move left or right
    final_position = (position + sample_size) // 2
    bin_counts[final_position] += 1

    # Update visualization
    fig, ax = plt.subplots()
    ax.bar(range(sample_size + 1), bin_counts, color='blue', alpha=0.7)
    ax.set_title(f"Visualization (Particle {particle + 1})")
    ax.set_xlabel("Final Position")
    ax.set_ylabel("Count")
    placeholder.pyplot(fig)
    time.sleep(animation_speed / 1000.0)

# Final Galton Board Visualization
st.header("Final Visualization Results")
fig, ax = plt.subplots()
ax.bar(range(sample_size + 1), bin_counts, color='green', alpha=0.7)
ax.set_title("Final Visualization Simulation")
ax.set_xlabel("Final Position")
ax.set_ylabel("Count")
st.pyplot(fig)

# Summary Table
st.header("Summary Statistics")
st.write(pd.DataFrame({
    "Metric": ["Population Mean", "Sample Mean", "Population Std Dev", "Sample Std Dev"],
    "Value": [np.mean(population), np.mean(sample_means), np.std(population), np.std(sample_means)]
}))