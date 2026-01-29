"""
Testing script to read generated user features from synthetic generator + feature extractor setup.
@author: Attabra Benjamin Ekow
@date: 2025-01-28
"""

import pandas as pd

df = pd.read_csv("data/user_features.csv", index_col="user_id")

print(df.head())

print(80 * "=")

df.info()
