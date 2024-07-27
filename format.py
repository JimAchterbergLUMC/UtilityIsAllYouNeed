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

print(results)


cats = ["utility", "privacy", "fidelity"]
cats_labels = [["performance"], ["privacy", "attack"], ["stats", "detection"]]

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
# ensure we are only showing relevant metrics (i.e. remove syn_id)
# add metric ranges at each row
# min max scale each row based on metric range where; REVERSE scale if direction=minimize to ensure higher=better
