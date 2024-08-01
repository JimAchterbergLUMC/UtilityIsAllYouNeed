import seaborn as sns
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

for i, filename in enumerate(os.listdir("results")):
    if i == 0:
        results = []
    results.append(
        pd.read_csv("results/" + filename).assign(model=filename.split(".")[0])
    )
results = pd.concat(results)
results = results.rename({"Unnamed: 0": "name"}, axis=1)
results = results[["name", "mean", "stddev", "direction", "model"]]

# remove metrics which are unwanted in the plot
remove = [
    "stats.alpha_precision.delta_precision_alpha_naive",
    "stats.alpha_precision.delta_coverage_beta_naive",
    "stats.alpha_precision.authenticity_naive",
    "performance.xgb.gt",
    "performance.xgb.syn_id",
    "privacy.k-anonymization.gt",
    "privacy.distinct l-diversity.gt",
    "privacy.identifiability_score.score",
]  # all metrics ending in these words are unwanted
results = results[results["name"].apply(lambda x: x not in remove)]

# map metric names to interpretable names
metric_mapper = {
    "stats.jensenshannon_dist.marginal": "JS",
    "stats.max_mean_discrepancy.joint": "MMD",
    "stats.alpha_precision.delta_precision_alpha_OC": "a_prc",
    "stats.alpha_precision.delta_coverage_beta_OC": "B_rec",
    "stats.alpha_precision.authenticity_OC": "Auth",
    "performance.xgb.syn_ood": "TSTR",
    "detection.detection_xgb.mean": "Detc",
    "privacy.delta-presence.score": "d_pres",
    "privacy.k-anonymization.syn": "k_ano",
    "privacy.k-map.score": "k_map",
    "privacy.distinct l-diversity.syn": "l_div",
    "privacy.identifiability_score.score_OC": "idtf",
}
# attach dataset specific AIA metric
for idx, row in results[
    results["name"].str.contains("leakage", case=False, na=False)
].iterrows():
    metric_mapper[row["name"]] = "AIA_" + row["name"].split(".")[-1]
results["name"] = results["name"].map(metric_mapper)

metric_info = {
    "utility": ["TSTR"],
    "privacy": [
        "d_pres",
        "k_ano",
        "k_map",
        "l_div",
        "idtf",
        "Auth",
    ],
    "fidelity": ["JS", "MMD", "a_prc", "B_rec", "Detc"],
}
# attach dataset specific AIA metric
metric_info["privacy"].extend(
    [value for value in metric_mapper.values() if "AIA_" in value]
)

# add metric ranges
results["metric_range"] = [(0, 1)] * len(results)
special_ranges = {
    "d_pres": (
        0,
        results["mean"][results["name"] == "d_pres"].max(),
    ),
    "k_ano": (
        1,
        results["mean"][results["name"] == "k_ano"].max(),
    ),
    "k_map": (
        1,
        results["mean"][results["name"] == "k_map"].max(),
    ),
    "l_div": (
        0,
        results["mean"][results["name"] == "l_div"].max(),
    ),
}  # all non (0,1) ranges
results["metric_range"] = results.apply(
    lambda row: special_ranges.get(row["name"], row["metric_range"]), axis=1
)

# scale all to (0,1)
results["mean"] = results.apply(
    lambda row: (row["mean"] - row["metric_range"][0])
    / (row["metric_range"][1] - row["metric_range"][0]),
    axis=1,
)

# for minimize rows, perform (1-x)
results["mean"][results["direction"] == "minimize"] = (
    1 - results["mean"][results["direction"] == "minimize"]
)


fig, axes = plt.subplots(3, 1, figsize=(10, 7))
bar_width = 0.2
for j, (cat, metrics) in enumerate(metric_info.items()):
    data = results[results["name"].isin(metrics)]
    ax = sns.barplot(
        data=data,
        x="name",
        y="mean",
        hue="model",
        ax=axes[j],
        width=bar_width,
        palette="viridis",
    )

    # Align error bars with bars
    for i, model in enumerate(data["model"].unique()):
        model_data = data[data["model"] == model]
        bar_positions = [
            bar.get_x() + bar.get_width() / 2
            for bar in ax.patches
            if bar.get_x() in model_data["name"].values
        ]

        # Map bar positions to actual data positions
        positions = [
            model_data["name"].tolist().index(name) for name in model_data["name"]
        ]

        ax.errorbar(
            x=np.array(positions)
            + (i - 0.5) * (0.5 * bar_width),  # Adjust positions to center with bars
            y=model_data["mean"],
            yerr=model_data["stddev"],
            fmt="none",  # No marker for the error bars
            color="black",
            capsize=5,
        )

    axes[j].set_xlabel("")
    axes[j].set_ylabel("")
    axes[j].set_ylim((0, 1))
    axes[j].set_title(cat)
    axes[j].legend(loc="upper right")

plt.suptitle("Adult Census")
plt.tight_layout()
plt.show()

# TBD:
# the OneClass representation of some metrics is still vague
#   -> it seems to project data into a hypersphere before calculating distance metrics
#   -> is this sensible?
# where is wasserstein distance? its lost when its incredibly big for some reason. in small samples (so lower distance) its maintained
