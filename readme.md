## Evaluating Sequential Recommendations Repository
The code in this repository is connected to the paper:
Evaluating Sequential Recommendations “In the Wild”:
A Case Study on Offline Accuracy, Click Rates, and Consumption

Abstract:
Sequential recommendation problems have received increased research interest in recent years. Our knowledge about the effectiveness of sequential algorithms in practice is however limited. In this paper, we report on the outcomes of an A/B test on a video and movie streaming platform, where we benchmarked a sequential model against a non-sequential personalized recommendation model and a popularity-based baseline. Contrary to what we had expected from a preceding offline experiment, we observed that the popularity-based and the non-sequential model lead to the highest click-through rates. However, in terms of the adoption of the recommendations, the sequential model was the most successful one and led to the longest viewing times. While our work thus confirms the effectiveness of sequential models in practice, it also reminds us about important open challenges regarding (a) the sometimes limited predictive power of offline evaluations and (b) the dangers of optimizing recommendation models for click-through rates.

---
USAGE:
In this repository you will find the `rec` python package created to evaluate an ALS Collaborative Filtering model, Markov Model, and a Hybrid of the two.
It contains tools to calculate popularity in datasets, and evaluate MRR, CTR, Coverage and Popularity measures. See main.py for how it can be used.

Unfortunatly we cannot provide the datasets used in this paper, but the code is written to be able to handle any dataset with some given column names. For the ALS model and the viewing datataset, the columns `["profileId", "itemId", "durationSec"]` must be present, and for the sequential model it expects the mined frequency rules with the columns `["itemId", "nextItemId", "count"]`. For this case our data was in the `.parquet` format, but the code can be adjusted for csv if wished by simply altering the `load_data` functions found in the model python files, e.g `rec/models/als.py`. The testing data has the columns
`["profile_id", "item_id", "next_item_id"]`, here its important that the profiles and items exists in the training dataset, no logic is implemented to handle this internally at runtime. You can adjust paths to the data in the main.py file or follow this file hierarchy:

```
├── als
│   ├── test
│   └── train
├── evaluations # add me before running.
│   └── results.csv # this will come after running the main.py file
├── mc
│   ├── test
│   ├── train
│   └── train-short
└── testdata
    ├── may
    └── test_dataset_filtered_als_mc.csv # to illustrate the filtering we did before running the evaluation.
```

There is also a tool to allow you to get notified through Slack, add the env variables:
```
SLACK_URL=...
SLACK_CHANNEL=...
```
You dont need a Slack bot for this to work, as we are only sending messages, be warned the code is messy, but gets the job done. An evaluatin of the results, as well as the actual results of the offline evaluation conducted in this thesis can be found in the `results` folder.

The required packages can be found in the `requirements.txt` file, and can be installed by running `pip install -r requirements.txt`.

The experiments was conducted on a machine with the following specs:
```
Ryzen 3900X
32GB DDR4 3200MHz RAM
Nvidia RTX 3070
```
but was also tested on an Macbook Pro M2 16" 2022.

---
[#REDACTED FOR ANONYMITY]
This work was supported by industry partners and the Research Council of Norway
with funding to MediaFutures: Research Centre for Responsible Media Technology and
Innovation, through the Centres for Research-based Innovation scheme, project number
309339.
[/#REDACTED FOR ANONYMITY]
