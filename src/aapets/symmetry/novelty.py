import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import Config


class NoveltyArchive:
    def __init__(self, config: Config):
        self.k = config.novelty_knn
        self.add_threshold = config.novelty_add_threshold
        self.archive = []  # list of behavior descriptors

    def novelty(self, footprint):
        assert all(0 <= x <= 1 for x in footprint), "Footprint must be normalized in [0, 1]"
        k = min(len(self.archive), self.k)
        if k == 0:
            return float("inf")

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.archive)
        dist, _ = nn.kneighbors([footprint])
        n = dist.mean()

        if n > self.add_threshold:
            self.archive.append(n)

        return n
