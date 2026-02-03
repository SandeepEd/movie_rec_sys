"""
Content-based filtering recommender.

Uses movie features (genres, title, release year) to compute item-item similarity
and recommend movies similar to ones the user has liked.

Classes:
    ContentBasedRecommender: Item-item similarity based recommender

Key Methods:
    - compute_similarity: Calculate cosine similarity between items
    - build_user_profile: Create user profile from rated items
    - recommend: Generate recommendations based on content similarity
"""
