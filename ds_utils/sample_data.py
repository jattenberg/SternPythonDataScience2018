from .features_pipeline import pipeline_from_config
import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np



def get_car_data(classification=True):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"

    predictor_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin']

    mpg_df = pd.read_csv(url,
                         delim_whitespace=True,
                         header=None,
                         names=['mpg'] + predictor_columns + ['car_name']).dropna()

    if classification:
        median = mpg_df["mpg"].median()

        Y = mpg_df["mpg"].apply(lambda m: 1 if m > median else 0)
        X = mpg_df[predictor_columns]

        return X, Y
    else:

        Y = mpg_df["mpg"]
        X = mpg_df[predictor_columns]

        return X, Y

def get_dummy_data():
    np.random.seed(56)
    X, Y = make_blobs(n_samples=4000, n_features=3, cluster_std=4, centers=3, shuffle=False, random_state=42)
    colors = ["red"] * 3800 + ["blue"] * 200
    Y = np.array([0] * 3800 + [1] * 200)

    order = np.random.choice(range(4000), 4000, False)

    X = X[order]
    Y = Y[order]

    X = pd.DataFrame(X, columns=['earning', 'geographic', 'experience'])

    return X, Y

def get_mailing_data():
    mailing_url = "https://gist.githubusercontent.com/anonymous/5275f1f59be561ec9734c90d80d176b9/raw/f92227f9b8cdca188c1e89094804b8e46f14f30b/-"
    mailing_df = pd.read_csv(mailing_url)

    config = [
        {
            "field": "Income",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "Firstdate",
            "transformers": [
                {"name": "standard_numeric"}
                ]
            },
        {
            "field": "Lastdate",
            "transformers": [
                {"name": "standard_numeric"}
                ]
            },
        {
            "field": "Amount",
            "transformers": [
                {"name": "standard_numeric"},
                {
                    "name": "quantile_numeric",
                    "config": {"n_quantiles": 10}
                    }
                ]
            },
        {
            "field": "rfaa2",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "rfaf2",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "pepstrfl",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "glast",
            "transformers": [
                {"name": "standard_numeric"},
                {
                    "name": "quantile_numeric",
                    "config": {"n_quantiles": 10}
                    }
                ]
            },
        {
            "field": "gavr",
            "transformers": [
                {"name": "standard_numeric"},
                {
                    "name": "quantile_numeric",
                    "config": {"n_quantiles": 10}
                    }
                ]
            }
        ]


    pipeline = pipeline_from_config(config)
    X = pipeline.fit_transform(mailing_df)

    return X, mailing_df["class"]
