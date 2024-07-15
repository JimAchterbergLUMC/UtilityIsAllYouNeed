from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import os
import json
from ucimlrepo import fetch_ucirepo
from utils import preprocess, plot_df
from matplotlib import pyplot as plt

ds = "adult"
with open("datasets.json", "r") as f:
    config = json.load(f)
config = config[ds]
dataset = fetch_ucirepo(id=config["id"])
X = dataset.data.features
y = dataset.data.targets

df = preprocess(X=X, y=y, config=config)
# print(df.nunique())
# plot_df(df)

# we need to set the discrete feature names as environment variables to later retrieve them easily
discrete = config["discrete"]
discrete_json = json.dumps(discrete)
os.environ["DISCRETE"] = discrete_json


# setup dataloader
X_r = GenericDataLoader(
    df,
    target_column="target",
    sensitive_columns=config["sensitive"],
)


# setup benchmarking
score = Benchmarks.evaluate(
    [
        (
            "TVAE_FASD",
            "tvae",
            {
                "fasd": True,
                "batch_size": 2**7,
                "n_iter": 2**8,
                "fasd_args": {
                    "hidden_dim": 2**5,
                    "num_epochs": 2**7,
                    "batch_size": 2**7,
                },
            },
        ),
        ("TVAE", "tvae", {"fasd": False, "batch_size": 2**7, "n_iter": 2**8}),
    ],
    X_r,
    task_type="classification",
    metrics={"performance": ["xgb"], "detection": ["detection_xgb"]},
    synthetic_size=len(X),
    repeats=1,
    synthetic_cache=False,
    synthetic_reuse_if_exists=False,
    use_metric_cache=False,
)

if not os.path.exists("results"):
    os.makedirs("results")
score["TVAE_FASD"].to_csv("results/tvae_fasd.csv")
score["TVAE"].to_csv("results/tvae.csv")

print(score["TVAE_FASD"])
print(score["TVAE"])
