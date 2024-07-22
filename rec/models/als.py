import pyarrow.parquet as pq
import glob

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import logging
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from rec.types.types import Recommendation, RecommendedItem
import threadpoolctl

class ALS:
    def __init__(self, factors=20, use_gpu=False, use_cg=False, iterations=10, logger=None):

        threadpoolctl.threadpool_limits(12, "blas")

        self.logger = logger
        self.logger.name = "als"
        self.model = AlternatingLeastSquares(
            factors=factors,
            use_gpu=use_gpu,
            use_cg=use_cg,
            iterations=iterations
        )

    def load_data(self, path, nested=False, limit=-1):
        i = 0
        if nested:
            dfs = []
            for file in glob.glob(path + "/**/*.parquet", recursive=True):
                if i > limit and limit != -1:
                    break
                df = pq.read_table(file).to_pandas()
                self.logger.debug("Loaded file: " + file + " with shape: " + str(df.shape))
                dfs.append(df)
                i += 1
            self.data = pd.concat(dfs, ignore_index=True)
        else:
            self.data = pq.read_table(path).to_pandas()

    def _bm25(self, uim, K1=3.0, B=1.0):

        return bm25_weight(uim, K1=K1, B=B)

    def preprocess(self):
        self.sessions = self.data[["profileId", "itemId", "durationSec"]] \
            .rename(index=str, columns={'profileId': 'userId', 'durationSec': 'score'}) \
            .groupby(["userId", "itemId"]).sum() \
            .reset_index()

    def fit(self, K1=1.2, B=0.75):
        # set types for user and item IDs
        self.sessions['userId'] = self.sessions['userId'].astype("category")
        self.sessions['itemId'] = self.sessions['itemId'].astype("category")

        # Dicts of item/usery IDX to ID
        self.items = dict(enumerate(self.sessions['itemId'].cat.categories))
        self.users = dict(enumerate(self.sessions['userId'].cat.categories))

        # Reverse dicts of user/item mappings
        self.items_rev = {val: key for key, val in self.items.items()}
        self.users_rev = {val: key for key, val in self.users.items()}
        # Build Item-User interaction matrix
        self.uim = coo_matrix(
            (self.sessions['score'].astype(np.float32),
             (self.sessions['userId'].cat.codes,
              self.sessions['itemId'].cat.codes))
        ).tocsr()

        # Fit model

        self._bm25(self.uim, K1, B)
        self.model.fit(self.uim, show_progress=True)

    def recommend(self, user_id, N=5):
        u = self.users_rev.get(user_id, None)
        try:
            i = self.uim[self.users_rev[user_id]]
        except Exception as e:
            self.logger.error(e)
            i = None
        if u is None:
            self.logger.error("User not found")
            return None
        if i is None:
            self.logger.error("Item not found")
            return None
        return self.model.recommend(u, self.uim[self.users_rev[user_id]], N=N)

    def recommend_items(self, user_id, N=5):
        u = self.users_rev.get(user_id, None)
        try:
            i = self.uim[self.users_rev[user_id]]
        except Exception as e:
            self.logger.error(e)
            i = None
        if u is None:
            self.logger.error("User not found")
            return None
        if i is None:
            self.logger.error("Item not found")
            return None

        try:
            recs = self.model.recommend(u, self.uim[self.users_rev[user_id]], N=N)[0]
            return [self.items.get(rec) for rec in recs]
        except Exception as e:
            self.logger.error(e)
            return None

    def recommend_standard(self, user_id, N=5) -> Recommendation:
        recommendation = Recommendation(user_id, None, {}, [], [])
        u = self.users_rev.get(user_id, None)
        try:
            i = self.uim[self.users_rev[user_id]]
        except Exception as e:
            self.logger.error(e)
            i = None
        if u is None:
            self.logger.error("User not found")
            return None
        if i is None:
            self.logger.error("Item not found")
            return None

        try:
            recs = self.model.recommend(u, self.uim[self.users_rev[user_id]], N=N)
            items, scores = recs[0], recs[1]
            scores_np = np.array(scores)
            if len(scores_np) > 1:
                scores = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min())
            else:
                scores = np.array([1.0])
            recommendation.items = [RecommendedItem(self.items.get(items[i]), scores[i], "CF") for i in range(len(items))]
            return recommendation
        except Exception as e:
            self.logger.error(e)
            return None
