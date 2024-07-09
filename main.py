# load data from repo
import sys
import warnings
from sklearn.datasets import load_diabetes, load_iris
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import pandas as pd

X, y = load_iris(return_X_y=True, as_frame=True)
X["target"] = y

# setup dataloader
X_r = GenericDataLoader(
    X,
    target_column="target",
    sensitive_columns=[],
)

#setup benchmarking
score = Benchmarks.evaluate(
    [("adversarial forest", "arf", {})],
    X_r,
    task_type='classification',
    metrics={'sanity':['data_mismatch'],'detection':['detection_xgb']},
    synthetic_size=len(X),
    repeats=3,
)

print(score['adversarial forest'])

