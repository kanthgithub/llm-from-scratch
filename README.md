# llm-from-scratch

# What is Pytorch
- Pytorch is tensor-library, Automatic differentiation engine, Deep learning library
- library that allows you to do deep-learning, training and fine-tuning a model

# LLM Model From Scratch

This Jupyter notebook demonstrates the process of building a Large Language Model (LLM) from scratch, covering essential components such as data preparation, tokenization, model architecture (embedding, attention mechanisms, feed-forward networks), training, and evaluation. It aims to provide a fundamental understanding of LLM construction without relying on high-level frameworks for core components.

## Table of Contents

- [Introduction](#introduction)
- [Credits](#credits)
- [Setup and Dependencies](#setup-and-dependencies)
- [Dataset](#dataset)
- [Tokenization](#tokenization)
- [Model Architecture](#model-architecture)
  - [Embedding Layer](#embedding-layer)
  - [Attention Mechanism](#attention-mechanism)
  - [Multi-Head Attention](#multi-head-attention)
  - [Feed Forward Network (FFN)](#feed-forward-network-ffn)
  - [Transformer Block](#transformer-block)
  - [Decoder Only Transformer](#decoder-only-transformer)
- [Training](#training)
- [Inference](#inference)

## Introduction

This notebook provides a step-by-step guide to understanding the internal workings of a transformer-based LLM. It focuses on implementing the foundational blocks of such a model from first principles, making it an excellent resource for those who want to grasp the core concepts behind modern LLMs.

## Credits

A few images and concepts in this notebook are inspired by Sebastian Raschka's blog.

## Setup and Dependencies

To run this notebook, you will need the following libraries:

- `torch`: For building and training the neural network.
- `torch.nn`: Neural network modules.
- `torch.nn.functional` (as `F`): Common neural network functions.
- `torch.utils.data`: Utilities for data loading.
- `tiktoken`: For advanced tokenization (specifically, OpenAI's `cl100k_base` tokenizer is used).
- `tqdm`: For progress bars during training.

You can install these dependencies using pip:

```bash
pip install torch tiktoken tqdm
```

## Dataset

The notebook utilizes a simple dataset for demonstration purposes. The dataset consists of pairs of input and target sequences.

 - Input Sample: [1, 5, 2, 6, 3, 7, 4, 8]

 - Target Sample: [5, 2, 6, 3, 7, 4, 8, 9]

This setup is characteristic of sequence-to-sequence or next-token prediction tasks common in LLMs.

## Tokenization

The notebook implements a custom tokenization process and also leverages tiktoken for more robust tokenization.

 - Custom Tokenizer: A basic tokenizer is created to map characters to integers and vice versa. It builds a vocabulary from the input text and provides encode and decode functions.

 - tiktoken Integration: The cl100k_base tokenizer from tiktoken (used by OpenAI's models like gpt-4, gpt-3.5-turbo, text-embedding-ada-002) is also demonstrated for more practical tokenization.

## Model Architecture

The LLM is built component by component, detailing the implementation of each key layer:

### Embedding Layer

- Token Embeddings: Maps input tokens (integers) to dense vector representations.

- Positional Embeddings: Adds positional information to the token embeddings, crucial for transformers to understand the order of tokens in a sequence. This notebook implements fixed positional embeddings (sine and cosine functions).

## Attention Mechanism

The core of the Transformer architecture. This notebook details the SelfAttention mechanism.

  - Query, Key, Value (QKV): Explains how input sequences are transformed into Query, Key, and Value matrices.

  - Scaled Dot-Product Attention: Calculation of attention scores using the formula: softmax(Q * K^T / sqrt(d_k)) * V.

  - Masked Self-Attention: For decoder-only models, a look-ahead mask is applied to prevent tokens from attending to future tokens during training.

## Multi-Head Attention

Extends the single attention mechanism by performing multiple attention calculations in parallel.

  - Combines the outputs of multiple attention heads to capture diverse relationships within the sequence.


## Feed Forward Network (FFN)

 - A simple two-layer neural network with a ReLU activation, applied independently to each position in the sequence.

## Transformer Block

- Combines Multi-Head Attention and a Feed Forward Network, along with Layer Normalization and residual connections.

## Decoder Only Transformer

- The complete LLM architecture, stacking multiple Transformer Blocks. This architecture is suitable for generative tasks like text completion.

    - Input: Token IDs.

    - Output: Logits for the next token in the sequence.

## Training

The training process involves:

    - Loss Function: Cross-entropy loss is used, suitable for multi-class classification (predicting the next token from the vocabulary).

    - Optimizer: AdamW optimizer is employed for efficient training.

    - Batching: Data is processed in batches to improve training stability and speed.

    - Training Loop: Iterates over the dataset, performs forward and backward passes, and updates model weights.

    - Evaluation: The model is evaluated on a test set to monitor performance.

## Inference

After training, the notebook demonstrates how to use the trained model for inference to generate new sequences.

  - Greedy Decoding: The model predicts the next token with the highest probability at each step.
  - Generating Text: Shows how to seed the model with an initial sequence and have it generate a continuation.

This README provides a high-level overview of the llm-model-from-scratch.ipynb notebook. For detailed implementation and further understanding, please refer to the notebook itself.
