import seaborn as sns
import pandas as pd
import os
from matplotlib import pyplot as plt


for i, filename in enumerate(os.listdir("results")):
    if i == 0:
        results = []
    results.append(pd.read_csv("results/" + filename).assign(model=filename))
results = pd.concat(results)
results = results.rename({"Unnamed: 0": "name"}, axis=1)

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

# add metric ranges
results["metric_range"] = [(0, 1)] * len(results)

special_ranges = {
    "privacy.delta-presence.score": 0,
    "privacy.k-anonymization.syn": 0,
    "privacy.k-map.score": 0,
    "privacy.distinct l-diversity.syn": 0,
}
print(results)
quit()


# truncate names for better viz
def truncate_words(s):
    return ".".join([word[:10] for word in s.split(".")])


results["name"] = results["name"].apply(truncate_words)

cats = ["utility", "privacy", "fidelity"]
cats_labels = [["per"], ["pri", "att"], ["sta", "det"]]

fig, axes = plt.subplots(len(cats), 1, figsize=(10, 7))

for j, (cat, labels) in enumerate(zip(cats, cats_labels)):
    data = results[results["name"].str.startswith(tuple(labels))]
    sns.barplot(data=data, x="name", y="mean", hue="model", ax=axes[j])
    axes[j].set_xlabel("")
    axes[j].set_ylabel("")
    axes[j].set_title(cat)
    axes[j].legend(loc="upper right")

plt.suptitle("Adult Census")
plt.tight_layout()
plt.show()

# TBD:
# the OneClass representation of some metrics is still vague
#   -> it seems to compress data into a hypersphere before calculating distance metrics
#   -> is this sensible?
# add metric ranges at each row
# min max scale each row based on metric range where; REVERSE scale if direction=minimize to ensure higher=better
