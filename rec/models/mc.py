import pyarrow.parquet as pq
import glob
import pandas as pd
import numpy as np
import logging
from rec.types.types import Recommendation, RecommendedItem

class MC():
    def __init__(self, minScore=0.1, maxScore=1.0, bridgeThresholds=2, method='frequencyScoreNormalized', logger=None):
        self.logger = logger
        self.logger.name = "MC"
        self.method = method
        self.minScore = minScore
        self.maxScore = maxScore
        self.bridgeThresholds = bridgeThresholds
        self.model = None
        self.data = None

    def load_data(self, path, nested=False, limit=-1):
        i = 0
        if nested:
            dfs = []
            for file in glob.glob(path + "/**/*.parquet", recursive=True):
                if i > limit and limit != -1:
                    break
                df = pq.read_table(file).to_pandas()
                # print("Loaded file: " + file + " with shape: " + str(df.shape))
                dfs.append(df)
                i += 1
            self.data = pd.concat(dfs, ignore_index=True)
        else:
            self.data = pq.read_table(path).to_pandas()

    def remove_self_links(self):
        self.logger.debug("Removing self-links...")
        self.data = self.data[self.data['itemId'] != self.data['nextItemId']]

    def aggregate_counts(self):
        self.logger.debug("Aggregating counts...")
        self.data = self.data.groupby(['itemId', 'nextItemId']).agg(count=('count', 'sum')).reset_index()

    def calculate_frequency_score(self):
        self.logger.debug("Calculating frequency score...")
        self.data['sumCount'] = self.data.groupby('itemId')['count'].transform('sum')
        self.data['maxCount'] = self.data.groupby('itemId')['count'].transform('max')
        self.data['minCount'] = self.data.groupby('itemId')['count'].transform('min')
        self.data['numItems'] = self.data.groupby('itemId')['itemId'].transform('count')
        self.data = self.data[self.data['numItems'] >= self.bridgeThresholds]
        self.data['frequencyScore'] = self.data['count'] / self.data['sumCount']

    def log_transformation(self):
        self.logger.debug("Performing log transformation...")
        self.data['log2TransformedCount'] = np.log2(self.data['count'] + 1)
        self.data['log10TransformedCount'] = np.log10(self.data['count'] + 1)
        self.data['minLog2Score'] = self.data.groupby('itemId')['log2TransformedCount'].transform('min')
        self.data['maxLog2Score'] = self.data.groupby('itemId')['log2TransformedCount'].transform('max')
        self.data['minLog10Score'] = self.data.groupby('itemId')['log10TransformedCount'].transform('min')
        self.data['maxLog10Score'] = self.data.groupby('itemId')['log10TransformedCount'].transform('max')

    def linear_normalization(self):
        self.logger.debug("Performing linear normalization...")
        self.data['frequencyScoreNormalized'] = self.minScore + (self.data['count'] / self.data['maxCount']) * (self.maxScore - self.minScore)

    def log_normalization(self):
        self.logger.debug("Performing log normalization...")
        self.data['frequencyScoreNormalizedLog2'] = ((self.data['log2TransformedCount'] - self.data['minLog2Score']) /
                                                     (self.data['maxLog2Score'] - self.data['minLog2Score']) *
                                                     (self.maxScore - self.minScore)) + self.minScore
        self.data['frequencyScoreNormalizedLog10'] = ((self.data['log10TransformedCount'] - self.data['minLog10Score']) /
                                                      (self.data['maxLog10Score'] - self.data['minLog10Score']) *
                                                      (self.maxScore - self.minScore)) + self.minScore

    def rank_and_score(self):
        self.logger.debug("Ranking and scoring...")
        self.data['rank'] = self.data.groupby('itemId')['frequencyScore'].rank(method='first', ascending=False)
        self.data['rankScaledScoreLin'] = (self.minScore +
                                           ((self.data['numItems'] - self.data['rank']) *
                                            (self.maxScore - self.minScore) / (self.data['numItems'] - 1)))
        self.data['rankScaledScoreLog'] = (self.minScore *
                                           np.exp((self.data['numItems'] - self.data['rank']) *
                                                  np.log(self.maxScore / self.minScore) / (self.data['numItems'] - 1)))
    def set_data_to_dict(self):
        self.model = {}
        for row in self.data.to_dict(orient='records'):
            if row['itemId'] not in self.model:
                self.model[row['itemId']] = []
            self.model[row['itemId']].append((row['nextItemId'], row[self.method]))
        for key in self.model:
            self.model[key] = sorted(self.model[key], key=lambda x: x[1], reverse=True)

    def change_method(self, method):
        self.method = method
        self.set_data_to_dict()

    def fit(self, path, nested=False, limit=-1):
        self.load_data(path, nested, limit)
        self.remove_self_links()
        self.aggregate_counts()
        self.calculate_frequency_score()
        self.log_transformation()
        self.linear_normalization()
        self.log_normalization()
        self.rank_and_score()
        self.set_data_to_dict()
        self.logger.debug("Model fitting completed.")

    def recommend(self, itemId):
        result = self.model[self.model['itemId'] == str(itemId)].sort_values('frequencyScore', ascending=False)
        if result.empty:
            return None
        else:
            return result

    def recommend_items(self, itemId, method, K):
        result = self.model[self.model['itemId'] == str(itemId)].sort_values(method, ascending=False).head(K)
        if result.empty:
            return None
        else:
            return result[['itemId', 'nextItemId', method]]

    def has_item(self, itemId):
        return str(itemId) in self.model.keys()

    def recommend_standard(self, itemId, N=-1) -> Recommendation:
        recs = Recommendation(item_id=itemId, user_id=None, items_map={}, items=[], item_ids=[])
        result = self.model.get(str(itemId), None)
        if not result:
            return None
        else:
            for row in result[:N]:
                r = RecommendedItem(row[0], row[1], "BR")
                recs.items_map[r.item_id] = r
                recs.items.append(r)
            return recs
