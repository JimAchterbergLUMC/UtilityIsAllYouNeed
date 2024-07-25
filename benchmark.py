from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import os
import json
from ucimlrepo import fetch_ucirepo
from utils import preprocess, plot_df

ds = "adult"
with open("datasets.json", "r") as f:
    config = json.load(f)
config = config[ds]
dataset = fetch_ucirepo(id=config["id"])
X = dataset.data.features
y = dataset.data.targets

df = preprocess(X=X, y=y, config=config)
df = df[:5000]
# print(df.nunique())
# plot_df(df)


# we need to set the discrete feature names as environment variables to later retrieve them easily
discrete = config["discrete"]
discrete_json = json.dumps(discrete)
os.environ["DISCRETE"] = discrete_json


# setup dataloader
X_r = GenericDataLoader(
    data=df,
    sensitive_features=config["sensitive"],
    target_column="target",
    random_state=123,
    train_size=0.8,
)


tvae_kwargs = {
    "batch_size": 500,
    "n_iter": 10,
    "n_units_embedding": 128,
    "decoder_n_layers_hidden": 1,
    "decoder_n_units_hidden": 128,
    "decoder_nonlin": "relu",
    "decoder_dropout": 0,
    "encoder_n_layers_hidden": 1,
    "encoder_n_units_hidden": 128,
    "encoder_nonlin": "relu",
    "encoder_dropout": 0,
    "loss_factor": 2,
}
# tvae_kwargs = {}
# setup benchmarking
score = Benchmarks.evaluate(
    [
        (
            "TVAE_FASD",
            "tvae",
            {
                "fasd": True,
                "fasd_args": {
                    "hidden_dim": 64,
                    "num_epochs": 10,
                    "batch_size": 64,
                },
                **tvae_kwargs,
            },
        ),
        (
            "TVAE",
            "tvae",
            {"fasd": False, **tvae_kwargs},
        ),
    ],
    X_r,
    task_type="classification",
    metrics={
        # "stats": ["jensenshannon_dist"],
        # "performance": ["xgb"],
        # "detection": ["detection_xgb"],
        # "privacy": ["identifiability_score"],
        "attack": ["data_leakage_xgb"],
    },
    synthetic_size=len(df),
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
