## Authors note:
# This code is far from pretty, but it does the job.

import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
from typing import List
from collections import Counter
from rec.models.hseq import HSEQ
from rec.types.types import EvaluationCase, RecommendedItem, Recommendation

class Evaluation:
    def __init__(self, sample=False, sample_size=10000, out_path='./data/evaluations', logger=None, popularity_scores=None, session_popularity_scores=None, slack=None):
        self.sample = sample
        self.slack = slack
        self.sample_size = sample_size
        self.logger = logger
        self.logger.name = "evaluator"
        self.out_path = out_path
        self.profile_id_key = 'profile_id'
        self.item_id_key = 'item_id'
        self.next_item_id_key = 'next_item_id'
        self.measure_date_key = 'measure_date'
        self.data = {}
        self.popularity_scores = popularity_scores
        self.session_popularity_scores = session_popularity_scores
        self.ALS = None
        self.MC = None
        self.HSEQ = None

        self.missing_recommendations = 0

    def load_data(self, path):
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        # We have to do some major changes to ensure that there is no floatingpoint .0s
        df['item_id'] = df[self.item_id_key].astype(int)
        df['next_item_id'] = df[self.next_item_id_key].astype(int)
        df['next_item_id'] = df['next_item_id'].astype(str)
        if self.sample:
            df = df.sample(n=self.sample_size, random_state=42123)
        # Convert the sampled DataFrame to a list of dictionaries
        self.data = df.to_dict(orient='records')

    def setup(self, ALS, MC, HSEQ, path):
        # We always load the data first
        self.load_data(path)
        # We dont evaluate the ALS model here, only bridges, so we dont need to fit the model
        self.ALS = ALS
        self.MC = MC
        self.HSEQ = HSEQ

    def prepare_reranker_evaluations(self,models:List[str], methods: List[str], w1s: List[float], Ks: List[int], Ns: List[int]):
        # We take the cartesian product of the methods, w1s, Ks and Ns to get all possible combinations
        self.logger.debug("Preparing reranker evaluation cases...")
        self.evaluation_cases = []
        for model in models:
            for method in methods:
                for w1 in w1s:
                    for K in Ks:
                        for N in Ns:
                            self.evaluation_cases.append(EvaluationCase(model, method, w1, 1-w1, K, N))

        self._check_cases()
        self.logger.debug(f"Number of evaluation cases: {len(self.evaluation_cases)}")

    def _check_cases(self):
        # We pruen the cases that are not needed
        for case in self.evaluation_cases:
            if case.model == "mc":
                case.w1 = 0
                case.w2 = 0
            if case.model == "als":
                case.w1 = 1
                case.w2 = 0
                case.method = "not_used"
        self.evaluation_cases = list(set(self.evaluation_cases))

    def prepare_bridges_evaluations(self, methods: List[str], Ns: List[int]):
        self.logger.debug("Preparing bridges evaluation cases...")
        self.evaluation_cases = []
        for method in methods:
            for N in Ns:
                self.evaluation_cases.append(EvaluationCase("mc", method, 0, 0, 0, N))

    def _store_recs(self, model, method, w1, w2, K, N, map, accuracy, avgctr, \
                    missing_bridge_count, missing_cf_count, not_enough_bridge_count,\
                    not_enough_cf_count, experiement_id, avg_popularity_score, avg_count_popularity_score, avg_session_popularity_score, coverage, gini_index):
        file_path = f"{self.out_path}{experiement_id}.csv"
        if not os.path.exists(file_path):
            with open(file_path, 'a+') as f:
                f.write("model,method,w1,w2,K,N,MAP,avgmrr,avgctr,missing_bridges,missing_cf,not_enough_bridges,not_enough_cf,averege_duration_popularity_scores,averege_count_popularity_scores,avg_session_popularity_score,coverage,gini_index\n")
                f.write(f"{model},{method},{w1},{w2},{K},{N},{map},{accuracy},{avgctr},{missing_bridge_count},{missing_cf_count},{not_enough_bridge_count},{not_enough_cf_count},{avg_popularity_score},{avg_count_popularity_score},{avg_session_popularity_score},{coverage},{gini_index}\n")
        else:
            with open(file_path, 'a+') as f:
                f.write(f"{model},{method},{w1},{w2},{K},{N},{map},{accuracy},{avgctr},{missing_bridge_count},{missing_cf_count},{not_enough_bridge_count},{not_enough_cf_count},{avg_popularity_score},{avg_count_popularity_score},{avg_session_popularity_score},{coverage},{gini_index}\n")

    def click_through_rate(self, actual_clicks, recommendations: List[RecommendedItem]):
        return len(set(actual_clicks) & set(recommendations) / len(set(actual_clicks)))

    def evaluate_reranker(self, experiment_id):
        self.logger.debug("Starting evaluation...")
        # MC can be different based on the method, so we need to fit the model for each method
        for case in self.evaluation_cases:
            if case.model != "als" and case.method != self.MC.method:
                self.logger.debug("Changing method...")
                self.MC.change_method(case.method)
                # Refit reranker with new method.
                self.HSEQ = HSEQ(self.MC, self.ALS, logger=self.logger)
            self.logger.debug(f"Model: {case.model}, Method: {case.method}, w1: {case.w1}, w2: {case.w2}, K: {case.K}, N: {case.N}")
            self._evaluate_reranker(case.method, case.w1, case.w2, case.K, case.N, experiment_id, case.model)

    def gini(self, array, min_lenght=None):
        """Calculate the Gini coefficient of a numpy array."""
        # All values are treated equally, arrays must be 1d:
        if min_lenght:
            pad_size = max(min_lenght - len(array), 0)
            array = np.pad(array, (0, pad_size), 'constant', constant_values=(0))

        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        # np.add(array, 0.0000001, out=array, casting="unsafe")
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1, array.shape[0] + 1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


    def _evaluate_reranker(self, method, w1, w2, K, N, experiment_id, model):
        ctrs = []
        mrrs = []
        itemIds = []
        # Reset metrics:
        self.missing_recommendations = 0
        self.HSEQ.missing_bridge_count = 0
        self.HSEQ.missing_cf_count = 0
        self.HSEQ.not_enough_bridge_count = 0
        self.HSEQ.not_enough_cf_count = 0
        # Train a new instance of our model
        i = 0
        recommendations = {}
        with tqdm(total=len(self.data), desc='Processing recommendations') as pbar:
            for i, case in enumerate(self.data):
                # get recs from the reranker
                if model == "hseq":
                    recs = self.HSEQ.recommend(case[self.profile_id_key], str(case[self.item_id_key]), N=N, w1=w1, w2=w2, K=K)
                elif model == "als":
                    recs = self.ALS.recommend_standard(case[self.profile_id_key], N=N)
                elif model == "mc":
                    recs = self.MC.recommend_standard(case[self.item_id_key], N=N)
                else:
                    self.logger.error("Model not found.")
                    continue
                if recs is None:
                    self.missing_recommendations += 1
                    continue

                # transform recommended items to string
                try:
                    recommended_items = [str(rec.item_id) for rec in recs.items]
                except Exception as e:
                    self.logger.error(e)
                    continue

                ctr_score = 1 if str(int(case[self.next_item_id_key])) in recommended_items else 0
                ctrs.append(ctr_score)

                # Calculate MRR score
                try:
                    rank = recommended_items.index(str(int(case[self.next_item_id_key]))) + 1
                    mrr_score = 1 / rank
                except ValueError:
                    mrr_score = 0
                mrrs.append(mrr_score)

                # Add the actual and recommended items to the recommendations dictionary
                p = recommendations.get(case[self.profile_id_key], None)
                if not p:
                    recommendations[case[self.profile_id_key]] = {'actual': [], 'recommended': []}
                recommendations[case[self.profile_id_key]]['actual'].append(case[self.next_item_id_key])
                recommendations[case[self.profile_id_key]]['recommended'].extend(recommended_items)

                # Add items to the user agnostic pool of recommended items for Gini index
                itemIds.extend(recommended_items)

                # Update the progress bar every 10,000 iterations
                if (i + 1) % 10000 == 0:
                    pbar.update(10000)

        # Calculate precision for each user
        recommended_for_popularity = []
        precision_scores = {}
        for user_id, user_data in recommendations.items():
            actual_next_items = set(user_data['actual'])
            recommended_items = set(user_data['recommended'])

            # Calculate the number of correct recommendations
            correct_recommendations = len(actual_next_items.intersection(recommended_items))
            recommended_for_popularity.extend(recommended_items)
            # Calculate precision for this user
            precision = correct_recommendations / len(recommended_items) if recommended_items else 0

            precision_scores[user_id] = precision

        # Calculate Mean Average Precision (MAP)
        mean_avg_precision = sum(precision_scores.values()) / len(precision_scores) if precision_scores else 0

        # Calculate average popularity score, general average popularity score and general duration popularity score
        avg_popularity_score = None
        avg_count_popularity_score = None
        avg_session_popularity_score = None
        if self.popularity_scores is not None and self.session_popularity_scores is not None:
            # get all popularity scores for the recommended items
            duration_popularity = []
            count_popularity = []
            session_count_popularity = []

            for item in recommended_for_popularity:
                session_popularity_score = self.session_popularity_scores.get(item, None)
                viewing_popularity_scores = self.popularity_scores.get(item, None)
                # For GAPS and GDPS
                if viewing_popularity_scores:
                    duration_popularity.append(viewing_popularity_scores['duration_score'])
                    count_popularity.append(viewing_popularity_scores['count_score'])
                # For APS
                if session_popularity_score:
                    session_count_popularity.append(session_popularity_score)
            # calculate the average popularity score
            avg_popularity_score = sum(duration_popularity) / len(duration_popularity) if duration_popularity else 0
            avg_count_popularity_score = sum(count_popularity) / len(count_popularity) if count_popularity else 0
            avg_session_popularity_score = sum(session_count_popularity) / len(session_count_popularity) if session_count_popularity else 0

        avg_ctr = sum(ctrs) / len(ctrs) if ctrs else 0
        average_mrr = sum(mrrs) / len(mrrs) if mrrs else 0


        # Calculate Gini Index
        popularity_counter = Counter(itemIds)
        popularity_scores = np.array(list(popularity_counter.values()))
        # the minimum length of the array is set to 1040, which is the number of items recommended by the ALS in a preivous experiment
        gini_index = self.gini(popularity_scores, 1040)

        # Coverage:
        # Calculate the number of unique items recommended
        unique_items = len(set(recommended_for_popularity))
        items_count = len(self.popularity_scores)
        coverage = unique_items / items_count if unique_items else 0

        self._store_recs(model, method, w1, w2, K, N, mean_avg_precision, average_mrr, avg_ctr, self.HSEQ.missing_bridge_count, self.HSEQ.missing_cf_count, self.HSEQ.not_enough_bridge_count, \
                         self.HSEQ.not_enough_cf_count, experiment_id, avg_popularity_score, avg_count_popularity_score, avg_session_popularity_score, coverage, gini_index)
        self.logger.info(f"Missing recommendations: {self.missing_recommendations}")
        self.logger.info(f"Average CTR: {avg_ctr}")
        self.logger.info(f"Average MRR: {average_mrr}")
        self.logger.info(f"Mean Average Precision: {mean_avg_precision}")
        self.logger.info(f"Average Duration Popularity Score: {avg_popularity_score}")
        self.logger.info(f"Average Count Popularity Score: {avg_count_popularity_score}")
        self.logger.info(f"Coverage: {coverage}")
        self.logger.info(f"Gini Index: {gini_index}")
        self.logger.info(f"Rerank info: missing_bridges:{self.HSEQ.missing_bridge_count}, missing_cf:{self.HSEQ.missing_cf_count}, missing_enough_bridges:{self.HSEQ.not_enough_bridge_count}, missing_enough_cf:{self.HSEQ.not_enough_cf_count}")
        if self.slack:
            self.slack.send_results(
            f"{model},{method},{w1},{w2},{K},{N}",
            avg_ctr=avg_ctr,
            mean_avg_precision=mean_avg_precision,
            avg_popularity_score=avg_popularity_score,
            avg_count_popularity_score=avg_count_popularity_score,
            coverage=coverage,
            gini_index=gini_index
        )
