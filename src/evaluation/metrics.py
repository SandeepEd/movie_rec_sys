"""
Evaluation metrics for recommendation systems.

Implements standard ranking and recommendation metrics:
- Precision@K and Recall@K
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP)
- Coverage (catalog and user coverage)
- Diversity metrics
- Novelty metrics

Functions:
    precision_at_k: Calculate precision at K
    recall_at_k: Calculate recall at K
    ndcg_at_k: Calculate NDCG at K
    mean_average_precision: Calculate MAP
    coverage: Calculate catalog coverage
    diversity: Measure recommendation diversity
    evaluate_model: Comprehensive evaluation suite
"""
