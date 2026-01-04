# Addition Task: Training Transformers from Scratch

This repository implements training a small Transformer model from scratch to perform k-digit addition. The task serves as a testbed for understanding how neural networks learn arithmetic operations and their generalization capabilities.

## Task Overview

The model learns to perform addition of two k-digit numbers:
- **Input format**: `"1234 + 5678 ="`
- **Output format**: `"6912"` (no spaces, no thousands separator)

### Evaluation

- **In-Distribution (ID)**: Same format as training (`"1234 + 5678 ="` → `"6912"`)
- **Out-of-Distribution (OOD)**: Thousands separator in input (`"1 234 + 5 678="` → accepts `"6912"` or `"6 912"`)

## Results

From the latest training run:

- **ID Test Accuracy**: 91.63% (9163/10000)
- **OOD Test Accuracy**: 0.00% (0/10000)

The model successfully learns the addition task but fails to generalize to different input formatting, demonstrating format-dependent learning.

## Project Structure

```
cbai_tt/
├── src/
│   ├── tokenizer.py          # Character-level tokenizer (0-9, +, =, space)
│   ├── dataset.py            # Data generation and evaluation
│   └── helpers.py             # Utility functions
├── config/
│   ├── config.yaml           # Main configuration
│   └── experiment/
│       └── scratch.yaml      # Training from scratch config
├── train_colab.ipynb         # Colab notebook for training and evaluation
├── addition_task_outputs/    # Trained checkpoints and results
│   ├── checkpoints/          # Model checkpoints
│   ├── results.json         # Evaluation results
│   └── training_details.json # Training metadata
└── README.md                 # This file
```

## Quick Start

### Training in Google Colab

1. Open `train_colab.ipynb` in Google Colab
2. Upload the `src/` directory when prompted
3. Run all cells sequentially

The notebook includes:
- Task setup documentation
- Data generation and diagnostics
- Training with progress tracking
- Evaluation on ID and OOD test sets
- Results visualization and export

### Environment Setup (Colab)

The notebook installs dependencies automatically:
```python
!pip install -q torch transformers datasets trl wandb hydra-core matplotlib seaborn pandas
```

## Model Architecture

- **Base**: GPT-2 architecture
- **Layers**: 4
- **Hidden Size**: 128
- **Attention Heads**: 4
- **Intermediate Size**: 512
- **Vocab Size**: 13 (character-level: 0-9, +, =, space)

## Tokenizer

Custom character-level tokenizer (`src/tokenizer.py`):
- Vocabulary: `["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", " "]`
- Each character is a separate token
- Inherits from `transformers.PreTrainedTokenizer`

## Data Generation

The dataset generator (`src/dataset.py`):
- Generates unique addition problems
- Ensures non-overlapping train/val/test splits
- Supports both ID (no separators) and OOD (with thousands separators) formats
- Uniform sampling of operands in range [10^(k-1), 10^k - 1]

## Key Findings

1. **Model learns addition**: Achieves 91.63% accuracy on in-distribution test set
2. **Format sensitivity**: Fails completely (0%) on out-of-distribution format with thousands separators
3. **Training efficiency**: Trains in ~9 minutes on Tesla T4 GPU

## Files

- `train_colab.ipynb`: Complete training and evaluation notebook with results
- `addition_task_outputs/`: Contains trained checkpoints and results
  - `checkpoints/`: Model checkpoints (saved every epoch)
  - `results.json`: Evaluation results
  - `training_details.json`: Training configuration and metadata
