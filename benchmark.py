from sklearn.datasets import load_iris, load_digits
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import os
import json

X, y = load_digits(return_X_y=True, as_frame=True)
X["target"] = y

# we need to set the discrete feature names as environment variables to later retrieve them easily
discrete = ["pixel_0_1"]
discrete_json = json.dumps(discrete)
os.environ["DISCRETE"] = discrete_json


# setup dataloader
X_r = GenericDataLoader(
    X,
    target_column="target",
    sensitive_columns=[],
)

# setup benchmarking
score = Benchmarks.evaluate(
    [("TVAE_FASD", "tvae", {"fasd": True}), ("TVAE", "tvae", {"fasd": False})],
    X_r,
    task_type="classification",
    metrics={"performance": ["xgb"], "detection": ["detection_xgb"]},
    synthetic_size=len(X),
    repeats=1,
    synthetic_cache=False,
    synthetic_reuse_if_exists=False,
    use_metric_cache=False,
)

print(score["TVAE_FASD"])
print(score["TVAE"])
