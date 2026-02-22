import numpy as np
import pandas as pd
import os
import requests
from io import BytesIO, StringIO

class TubingenLoader:
    """
    Helper class to load Tubingen Cause-Effect Pairs
    Reference: https://is.tuebingen.mpg.de/cause-effect
    """
    BASE_URL = "https://webdav.tuebingen.mpg.de/cause-effect"

    def __init__(self, target_dir="datasets/tubingen"):
        self.target_dir = target_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

    def load_pair(self, pair_id=1):
        """Loads a specific pair by ID (1-108)"""
        filename = f"pair{pair_id:04d}.txt"
        url = f"{self.BASE_URL}/{filename}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = pd.read_csv(StringIO(response.text), sep=' ', header=None).values
                return data
            else:
                print(f"Failed to download pair {pair_id}")
                return None
        except Exception as e:
            print(f"Error loading Tubingen: {e}")
            return None

    def get_ground_truth(self):
        """Downloads the ground truth direction mapping"""
        url = f"{self.BASE_URL}/pairmeta.txt"
        # 1st col: pair index, 2nd: start, 3rd: end, 4th: weight
        # Usually X is 2nd col, Y is 3rd col
        # If ground truth is 1, then X -> Y
        pass 

if __name__ == "__main__":
    loader = TubingenLoader()
    pair = loader.load_pair(1)
    if pair is not None:
        print(f"Loaded Tubingen Pair 1: {pair.shape} samples")
