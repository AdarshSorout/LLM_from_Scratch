 ## Building a Large Language Model (LLM) From Scratch

A hands-on end-to-end implementation of transformers, tokenizers, training loops, and dataset preparation.

This project demonstrates how a modern Large Language Model works internally, by building every core component from scratch using Python + PyTorch.
It is designed both for learning and for interview demonstration.
## Features

✔ Custom Byte-Pair Encoding (BPE) tokenizer
✔ Transformer architecture built from scratch
✔ Multi-Head Self-Attention implementation
✔ Positional embeddings
✔ Causal language modeling
✔ End-to-end training on OpenWebText or a small custom dataset
✔ CPU/GPU compatibility
✔ Jupyter notebook training pipeline

This project is a fully hands-on implementation of the core building blocks of modern LLMs.
I did everything from scratch, without relying on HuggingFace or any high-level APIs.

Here is what I personally built:

## 1. Custom Tokenizer (BPE)

Implemented Byte Pair Encoding manually

Generated vocabulary from text corpus

Created encode/decode pipelines

Handled unknown tokens, merges, and special tokens

## 2. Complete Transformer Architecture

I implemented the full transformer decoder stack:

Scaled Dot-Product Attention

Multi-Head Self Attention

Positional Embeddings

Residual Connections

Layer Normalization

Feedforward MLP block

Causal masking for next-token prediction

No shortcuts. Everything implemented line-by-line.

## 3. Dataset Pipeline

Downloaded and prepared OpenWebText (or mini dataset)

Cleaned and chunked data

Created PyTorch dataset + dataloader

Built train/val split logic

## 4. Training Loop (Full LM Training)

Wrote a custom training loop (AdamW optimizer, LR scheduling)

Implemented batching, gradient clipping

Perplexity tracking

GPU/CPU autodetection

Model saving + checkpointing

## 5. Text Generation Engine

Temperature sampling

Top-K sampling

Greedy decoding

Prompt feeding

Auto-regressive next-token generation

🚀 Result:

A working, fully custom LLM pipeline capable of training from text and generating text.
## Research Papers:
Attention is All You Need - https://arxiv.org/pdf/1706.03762.pdf

A Survey of LLMs - https://arxiv.org/pdf/2303.18223.pdf

QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/pdf/2305.14314.pdf
