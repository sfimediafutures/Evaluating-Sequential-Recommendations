from rec.models.als import ALS
from rec.models.mc import MC
from rec.models.hseq import HSEQ
import logging
import colorlog
from rec.evaluator.evaluator import Evaluation
from rec.utils.popularity import PopularityScore
import threadpoolctl
import os
from rec.utils.slack import Slack
import traceback
import sys

def beep(n=1, type='Blow'):
    for i in range(n):
        os.system(f'afplay /System/Library/Sounds/{type}.aiff')


# Initialize the models

if __name__ == '__main__':
	threadpoolctl.threadpool_limits(12, "blas")
	logger = colorlog.getLogger()
	logger.setLevel(logging.DEBUG)

	# slack = Slack()
	slack = None
	try:
		if slack is not None:
			slack.send_message("Starting the evaluation script")
		# Create a StreamHandler to output logs to the terminal
		stream_handler = colorlog.StreamHandler()
		stream_handler.setLevel(logging.DEBUG)

		# Create a formatter with colors
		formatter = colorlog.ColoredFormatter(
		'%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
		log_colors={
			'DEBUG': 'cyan',
			'INFO': 'white',
			'WARNING': 'yellow',
			'ERROR': 'red',
			'CRITICAL': 'bold_red',
		}
		)
		# Add the formatter to the handler
		stream_handler.setFormatter(formatter)

		# Add the handler to the logger
		logger.addHandler(stream_handler)
		logger.info("Calculating popularity scores...")
		P = PopularityScore(logger=logger)
		P.load_data('./data/als/train', nested=True, limit=-1, type='viewing')
		P.calculate_popularity_scores(1000) # 1000 to consider all data

		PS = PopularityScore()
		PS.load_data('./data/mc/train', nested=True, limit=-1, type='sessions')
		PS.calculate_popularity_scores_sessions()

		logger.info("Fitting ALS model...")
		CFR = ALS(factors=1, use_gpu=False, use_cg=False, iterations=1, logger=logger)
		CFR.load_data('./data/als/train', nested=True, limit=-1)
		CFR.preprocess()
		CFR.fit()

		logger.info("Fitting MC model...")
		B = MC(method='frequencyScoreNormalizedLog2', logger=logger)
		B.fit(path='./data/mc/train-short', nested=True, limit=-1)

		logger.info("Fitting HSEQ model...")
		R = HSEQ(B, CFR, logger=logger)

		if slack is not None:
			slack.send_message("Models are trained, starting the evaluation...")
		experiment_id = 'gini_index'
		out_path = './data/evaluations/'
		E = Evaluation(sample=True, sample_size=100000, out_path=out_path, logger=logger, popularity_scores=P.popularity_scores, session_popularity_scores=PS.popularity_scores, slack=slack)
		R = HSEQ(B, CFR, logger=logger)
		E.setup(CFR, B, R, path='./data/testdata/test_dataset_filtered_als_mc.csv')
		E.prepare_reranker_evaluations(["hseq", "mc", "als"],['frequencyScoreNormalizedLog2'], [0.1], [20], [3])
		E.evaluate_reranker(experiment_id)


	except Exception:
		if slack is not None:
			slack.send_exception(sys.exc_info())
		logger.error("An exception occurred", exc_info=True)
		sys.exit(1)  # Close the app
