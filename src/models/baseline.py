# a smart dumb algorithm that recommends based on global movie popularity
from src.data.loader import MovieLensLoader

loader = MovieLensLoader()

class baseline_recommender:
    def __init__(self):
        pass

    def recommend(self, user_id, top_n=12):
        ratings = loader.load_ratings()
        return ratings