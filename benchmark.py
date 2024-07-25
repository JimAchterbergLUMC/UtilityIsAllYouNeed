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
    random_state=0,
    train_size=0.8,
)


tvae_kwargs = {}
score = Benchmarks.evaluate(
    [
        (
            "TVAE_FASD",
            "tvae",
            {
                "fasd": True,
                "fasd_args": {
                    "hidden_dim": 64,
                    "num_epochs": 1000,
                    "batch_size": 200,
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
        # "sanity": [
        #     "data_mismatch",
        #     "common_rows_proportion",
        #     "nearest_syn_neighbor_distance",
        #     "close_values_probability",
        #     "distant_values_probability",
        # ],
        "stats": [
            "jensenshannon_dist",
            # "chi_squared_test",
            # "feature_corr",
            # "inv_kl_divergence",
            # "ks_test",
            "max_mean_discrepancy",
            "wasserstein_dist",
            # "prdc",
            "alpha_precision",
            # "survival_km_distance",
        ],
        "performance": ["linear_model", "mlp", "xgb", "feat_rank_distance"],
        "detection": [
            "detection_xgb",
            "detection_mlp",
            "detection_gmm",
            "detection_linear",
        ],
        "privacy": [
            "delta-presence",
            "k-anonymization",
            "k-map",
            "distinct l-diversity",
            "identifiability_score",
            # "DomiasMIA_BNAF",
            # "DomiasMIA_KDE",
            # "DomiasMIA_prior",
        ],
        "attack": ["data_leakage_linear", "data_leakage_xgb", "data_leakage_mlp"],
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
