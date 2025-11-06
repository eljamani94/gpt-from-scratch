# GPT from Scratch

A complete implementation of a GPT-style decoder-only transformer model built from first principles using TensorFlow and Keras. This project demonstrates the core architecture of modern language models by implementing multi-head attention, positional embeddings, and autoregressive text generation.

## Overview

This project implements a text generation model trained on technical support ticket descriptions. Rather than using pre-built transformer libraries, all core components (attention mechanisms, transformer blocks, positional encodings) are implemented from scratch to provide deep insight into how GPT-style models work.

### Key Features

- **Custom Multi-Head Attention**: Hand-coded scaled dot-product attention mechanism
- **Transformer Decoder Block**: Complete implementation with layer normalization and feed-forward networks
- **Token & Position Embeddings**: Combined embedding layer for semantic and positional information
- **Autoregressive Generation**: Greedy decoding strategy for text generation
- **Training Pipeline**: End-to-end data preprocessing, training, and evaluation

## Architecture

The model follows the decoder-only transformer architecture similar to GPT-2:

1. **Input Processing**: Text tokenization with custom vocabulary
2. **Embedding Layer**: Token embeddings combined with positional encodings
3. **Transformer Block**: Multi-head self-attention with feed-forward network
4. **Output Layer**: Softmax classification over vocabulary

### Model Specifications

- **Vocabulary Size**: ~1,150 tokens
- **Embedding Dimension**: 512
- **Attention Heads**: 4
- **Feed-Forward Dimension**: 2,048
- **Max Sequence Length**: 56 tokens
- **Total Parameters**: ~2.8M

## Technical Stack

- **Framework**: TensorFlow 2.x / Keras
- **NLP Library**: Keras NLP
- **Tokenization**: TextVectorization layer
- **Architecture**: Custom transformer decoder

## Project Structure
```
.
├── gpt_notebook.ipynb           # Main implementation notebook
├── data/
│   └── tickets.txt              # Training corpus (downloaded)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone the repository**
```bash
   git clone https://github.com/eljamani94/gpt-from-scratch.git
   cd gpt-from-scratch
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Launch Jupyter**
```bash
   jupyter notebook gpt_notebook.ipynb
```

## Dependencies
```
tensorflow>=2.13.0
keras-nlp>=0.6.0
numpy>=1.24.0
jupyter>=1.0.0
```

## Usage

### Training the Model

The notebook walks through the complete pipeline:
```python
# 1. Data preprocessing
with open("data/tickets.txt", "r") as f:
    text = f.read()
tickets = [sentence + " EOS " for sentence in text.split(" --- ")]

# 2. Tokenization
vectorize_layer = TextVectorization(
    standardize="lower",
    output_mode="int",
    output_sequence_length=max_len
)
vectorize_layer.adapt(tickets)

# 3. Create training data
X, y = create_training_data(tickets)

# 4. Build model
model = create_model(
    max_sequence_length=56,
    vocab_size=1150,
    embedding_dimension=512
)

# 5. Train
model.fit(X, y, batch_size=32, epochs=25)
```

### Generating Text
```python
def generate(starter_string):
    """Generate text autoregressively from a starter prompt"""
    for _ in range(max_len - len(starter_string.split())):
        tokens = vectorize_layer(starter_string)
        token_expanded = tf.expand_dims(tokens, 0)
        pred = model.predict(token_expanded)
        next_word = vocab[pred.argmax()]

        if next_word == "eos":
            break

        starter_string += f" {next_word}"

    return starter_string

# Example
generate("I need help with")
# Output: "I need help with data preprocessing and handling missing values..."
```

## Implementation Details

### Attention Mechanism

The core of the model is the scaled dot-product attention:
```python
def coded_attention(query, key, value):
    # Q * K^T
    score = tf.matmul(query, key, transpose_b=True)

    # Scale by sqrt(d_k)
    scaled_score = score / tf.cast(22.6, tf.float32)

    # Softmax normalization
    weights = tf.nn.softmax(scaled_score, axis=-1)

    # Weighted sum of values
    output = tf.matmul(weights, value)

    return output, weights
```

### Data Preprocessing

The training data uses a clever masking technique to create multiple training examples from each sentence:

- Input: `[w1, w2, w3, w4, w5]`
- Creates pairs: `([w1] → w2), ([w1, w2] → w3), ([w1, w2, w3] → w4), ...`
- Uses `tf.linalg.band_part()` for efficient masking

### Training Strategy

- **Loss**: Sparse categorical cross-entropy
- **Optimizer**: Adam
- **Metrics**: Perplexity and accuracy
- **Batch Size**: 32
- **Epochs**: 25
- **Training Examples**: ~17,000 (after data augmentation)

## Results

After 25 epochs:
- **Final Loss**: ~0.19
- **Perplexity**: ~1.21
- **Accuracy**: ~93.7%

The model generates coherent technical descriptions:

**Input**: `"I need"`

**Output**: `"I need to detect anomalies or outliers in my irregular time series data with prediction or credit data. what are some techniques like domain adaptation, transfer learning, or using pre-trained language models such as bert or gpt-3 that can help me improve sentiment analysis performance..."`

## Model Architecture Diagram
```
Input Tokens (56,)
    ↓
Token + Position Embeddings (56, 512)
    ↓
Transformer Block:
  - Multi-Head Attention (4 heads)
  - Layer Normalization
  - Feed-Forward Network (2048 → 512)
  - Layer Normalization
    ↓
Global Average Pooling (512,)
    ↓
Dense Softmax Layer (1150,)
    ↓
Output: Next Token Probabilities
```

## Limitations

- **Small Dataset**: Only 372 training sentences limits vocabulary and generation quality
- **Simple Tokenization**: Word-level tokenization; no subword handling (BPE/WordPiece)
- **Greedy Decoding**: Uses argmax; no beam search or sampling strategies
- **No Masking**: Doesn't use causal masking in attention (relies on data preprocessing)
- **Limited Context**: Max sequence length of 56 tokens

## Future Enhancements

- [ ] Implement subword tokenization (BPE/SentencePiece)
- [ ] Add causal attention masking
- [ ] Implement sampling strategies (temperature, top-k, nucleus)
- [ ] Scale to larger datasets (e.g., 10K+ documents)
- [ ] Add beam search for generation
- [ ] Implement model checkpointing
- [ ] Add learning rate scheduling
- [ ] Visualize attention weights
- [ ] Compare with HuggingFace transformers

## Learning Outcomes

This implementation demonstrates:

1. **Attention Mechanisms**: How queries, keys, and values enable context-aware representations
2. **Positional Encoding**: Why position information is critical for transformers
3. **Autoregressive Generation**: How language models generate text token-by-token
4. **Transformer Architecture**: The building blocks of modern LLMs
5. **Training Dynamics**: How perplexity correlates with generation quality

## Extending to Other Datasets

The code is modular and can be adapted to other text generation tasks:
```python
# Change data source
with open("your_data.txt", "r") as f:
    text = f.read()

# Adjust hyperparameters
model = create_model(
    max_sequence_length=128,  # Longer sequences
    vocab_size=5000,          # Larger vocabulary
    embedding_dimension=768   # Bigger model
)
```

## Comparison with Production Models

| Feature | This Implementation | GPT-2 | GPT-3 |
|---------|---------------------|-------|-------|
| Parameters | 2.8M | 1.5B | 175B |
| Layers | 1 | 48 | 96 |
| Attention Heads | 4 | 16 | 96 |
| Training Data | 372 sentences | 40GB | 570GB |
| Tokenization | Word-level | BPE | BPE |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - GPT-1
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation

## Contributing

Contributions are welcome! Areas for improvement:

- Better tokenization strategies
- More efficient attention implementations
- Additional generation strategies
- Visualization tools
- Extended documentation

Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Training data: Technical support ticket descriptions
- Built following transformer architecture principles from "Attention Is All You Need"
- Uses TensorFlow/Keras ecosystem for implementation

## Contact

For questions or discussions, please open a GitHub issue.

---
