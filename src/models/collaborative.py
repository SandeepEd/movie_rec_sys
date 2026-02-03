"""
Collaborative filtering models using PyTorch.

Implements matrix factorization and neural collaborative filtering:
- User and item embeddings
- Bias terms
- Neural architecture for learning embeddings

Classes:
    MatrixFactorization: Basic MF model with embeddings and biases
    NeuralCF: Neural collaborative filtering with MLP layers
    RatingDataset: PyTorch Dataset for ratings data

Training utilities:
    - train_epoch: Single epoch training
    - evaluate: Model evaluation
    - negative_sampling: Generate negative samples
"""
