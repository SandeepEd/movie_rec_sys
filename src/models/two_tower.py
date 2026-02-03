"""
Two-tower architecture for recommendation retrieval.

Implements separate neural networks for users and items that output embeddings
in the same space, enabling efficient similarity-based retrieval.

Classes:
    TwoTowerModel: Main two-tower neural network
    UserTower: MLP for user feature encoding
    ItemTower: MLP for item feature encoding
    TwoTowerDataset: PyTorch Dataset for two-tower training

Key Components:
    - Separate towers for users and items
    - L2 normalized embeddings
    - Cosine similarity for scoring
    - Efficient retrieval with pre-computed item embeddings

Training:
    - In-batch softmax loss or contrastive loss
    - Negative sampling strategies
    - Learning rate scheduling
"""
