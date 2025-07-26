# Neural Network Coffee Roast Classifier

This project demonstrates a simple neural network classifier that predicts whether a coffee roast is "Good" or "Bad" based on two key features: **roast temperature** and **roast duration**. It also includes visualizations to understand how the network processes and classifies input data at each layer.

---

## 🧠 Project Overview

- **Model**: A small feedforward neural network with one hidden layer (3 neurons) and sigmoid activations.
- **Input Features**:
  - Roast temperature (in °C)
  - Roast duration (in minutes)
- **Goal**: Learn a non-linear boundary to classify roasts as *Good* or *Bad*.

---

## 🎯 Objectives

- Simulate a real-world classification problem using synthetic coffee roast data.
- Train a neural network to separate "good" and "bad" roasts.
- Visualize:
  - Decision boundaries
  - Neuron activations
  - Final network output in 3D

---

## 🧪 Model Architecture

- **Input Layer**: 2 inputs (Temperature, Duration)
- **Hidden Layer**: 3 neurons with sigmoid activations
- **Output Layer**: 1 neuron (sigmoid) giving roast quality score (0–1)

---

## 📈 Visual Examples

### Layer 1 - Neuron Activations
Each unit in the first layer learns to respond to different boundaries within the roast space.

- **Light Blue Areas**: High activation (neuron "fires")
- **Dark Blue Areas**: Low activation

### Output Layer - Decision Function (3D)
- The final layer combines the neuron outputs.
- Areas with **low values** (in blue) correspond to **"Good Roasts"**
- Areas with **high values** (lighter blue) are **"Bad Roasts"**

---

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**

---
