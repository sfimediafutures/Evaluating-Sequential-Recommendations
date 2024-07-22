import pandas as pd
import pyarrow.parquet as pq
import glob
from datetime import datetime, timedelta
import logging

class PopularityScore:
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.data = None
        self.popularity_scores = {}
        self.type = None

    def load_data(self, path, nested=False, limit=-1, type=None):
        if type is None:
            raise ValueError("Type must be set before loading data, accepted values are 'viewing' and 'sessions'")
        if type not in ['viewing', 'sessions']:
            raise ValueError("Type must be either 'viewing' or 'sessions'")
        self.type = type
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


    def calculate_popularity_scores(self, days):
        if self.type is None:
            raise ValueError("Type must be set before calculating popularity scores")
        if self.type == 'sessions':
            raise ValueError("This method is not supported for session data")
        if self.data is None:
            raise ValueError("Data must be loaded before calculating popularity scores")
        
        latest_date = self.data['firstStart'].max()
        cutoff_date = latest_date - timedelta(days)

        filtered_df = self.data[self.data['firstStart'] >= cutoff_date]
        filtered_df = filtered_df[filtered_df['contentType'].isin(['SERIES', 'MOVIE'])]

        # total_watch_time = filtered_df['durationSec'].sum()
        count = filtered_df['itemId'].value_counts().to_dict()
        duration = filtered_df.groupby('itemId')['durationSec'].sum().to_dict()

        # Min-max normalization for count
        min_count = min(count.values())
        max_count = max(count.values())
        count_scores_normalized = {item: (value - min_count) / (max_count - min_count) for item, value in count.items()}

        # Min-max normalization for duration
        min_duration = min(duration.values())
        max_duration = max(duration.values())
        duration_scores_normalized = {item: (value - min_duration) / (max_duration - min_duration) for item, value in duration.items()}

        # Combine normalized scores into popularity_scores
        self.popularity_scores = {
            item: {
                "count_score": count_scores_normalized.get(item, 0),
                "duration_score": duration_scores_normalized.get(item, 0)
            } for item in set(count) | set(duration)
        }

    def calculate_popularity_scores_sessions(self):
        if self.type is None:
            raise ValueError("Type must be set before calculating popularity scores")
        if self.type == 'viewing':
            raise ValueError("This method is not supported for viewing data")
        if self.data is None:
            raise ValueError("Data must be loaded before calculating popularity scores")
        
        grouped_df = self.data.groupby('itemId')['count'].sum()
        a = grouped_df.to_dict()

        next_grouped_df = self.data.groupby('nextItemId')['count'].sum()
        b = next_grouped_df.to_dict()

        # Combine the dictionaries
        combined_dict = {key: a.get(key, 0) + b.get(key, 0) for key in set(a) | set(b)}

        # Calculate min and max values
        min_count = min(combined_dict.values())
        max_count = max(combined_dict.values())

        # Apply min-max normalization
        self.popularity_scores = {
            item: (count - min_count) / (max_count - min_count) if max_count != min_count else 0
            for item, count in combined_dict.items()
        }
