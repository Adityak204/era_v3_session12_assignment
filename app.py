import streamlit as st
import torch
from pathlib import Path
import math
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from src.gpt_base import GPT
import json


# Config class for model parameters
@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 65
    num_layer: int = 12  # number of layers
    num_head: int = 12  # number of heads
    emb_dim: int = 768  # embedding dimension
    dropout: float = 0.1  # dropout rate


# Copy all the model classes (GPT, MultiHeadAttention, FeedForward, TransformerBlock) here
# [Previous model code goes here]

# Load stoi and itos from docs
with open("docs/stoi.json") as f:
    stoi = json.load(f)

with open("docs/itos.json") as f:
    itos = json.load(f)


# Encoding/Decoding functions
def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


def predict_next_word(text, model, seq_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(seq_len):
        xb = torch.tensor(encode(text)).unsqueeze(0).to(device)
        yb = model(xb)
        next_word = yb[0, -1].argmax().item()
        text += itos[str(next_word)]
    return text


# Streamlit app
st.title("GPT Text Generation")
# Add some usage instructions
st.markdown(
    """
### How to use:
1. Enter your text prompt in the text box above
2. Adjust the sequence length using the slider
3. Click 'Generate Text' to see the model's output

Note: Longer sequence lengths will take more time to generate.
"""
)

# Input text box
input_text = st.text_area("Enter your text prompt:", height=100)

# Sequence length slider
seq_length = st.slider(
    "Select sequence length for prediction:",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
)

# Model loading and prediction
if st.button("Generate Text"):
    if input_text:
        try:
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config = GPTConfig()
            model = GPT(config)
            model = model.to(device)

            # Load checkpoint
            checkpoint_path = "/Users/aditya/Documents/self_learning/ERA V3/week 12/model artifacts/gpt_model_and_loss.pth"

            with st.spinner("Loading model and generating text..."):
                _dict = torch.load(checkpoint_path, map_location=device)
                model_state_dict = _dict["model_state_dict"]
                model.load_state_dict(model_state_dict)

                # Generate text
                generated_text = predict_next_word(input_text, model, seq_length)

                # Display results
                st.subheader("Generated Text:")
                st.write(generated_text)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter some text first!")
