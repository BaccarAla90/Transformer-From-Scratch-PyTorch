# Custom PyTorch Transformer: Architecture & Benchmarking

This repository contains a from-scratch PyTorch implementation of a Transformer Encoder, trained on the IMDB Movie Review dataset for sentiment analysis. 

Rather than relying on pre-built `nn.Transformer` modules, the architecture is broken down into raw components (`attention.py`, `layers.py`) to demonstrate a deep understanding of self-attention mechanics and matrix transformations.

## The Benchmark: Custom vs. Pre-trained
To evaluate the model, I wrote a custom testing loop to benchmark its accuracy directly against an industry-standard, pre-trained Hugging Face pipeline (`bert-base-uncased` fine-tuned for sentiment).

*   **Custom Transformer Accuracy (Trained from scratch):** ~76.3%
*   **Pre-Trained Hugging Face Accuracy:** ~88.0%+

This performance gap demonstrates the immense value of pre-training on massive corpuses (like Wikipedia) before fine-tuning on a specific classification task like IMDB reviews. 

## Repository Structure
*   `attention.py`: The Multi-Head Attention math
*   `layers.py`: The Encoder blocks and FeedForward networks
*   `transformer.py`: The final Model class
*   `notebook.ipynb`: Data loading, PyTorch training loop, and the side-by-side Hugging Face benchmark.

* (Mini project for learning purposes)
