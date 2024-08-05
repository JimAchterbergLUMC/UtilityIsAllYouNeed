# individual tables for all fidelity, utility and privacy metrics

# fidelity table:
# - separate table for each metric
# - columns are datasets
# - rows are models
# -> then in Latex, stack the different tables on top of each other!

import pandas as pd
import os

# loop through the datasets in results
results = []
for ds_dir in os.listdir("results"):
    path = f"results/{ds_dir}"
    for filename in os.listdir(path):
        results.append(
            pd.read_csv(f"{path}/{filename}").assign(
                model=filename.split(".")[0], ds=ds_dir
            )
        )

results = pd.concat(results)
results = results.rename({"Unnamed: 0": "name"}, axis=1)
results = results[["name", "mean", "stddev", "model", "ds"]]
df = results

# create the pivot tables for each metric:

# Group the DataFrame by 'name'
grouped = df.groupby("name")

# Create a dictionary to hold the DataFrames
dfs = {}

# Process each group
for name, group in grouped:
    # Pivot the table
    pivot = group.pivot(index="model", columns="ds", values=["mean", "stddev"])

    # Round the values to three decimal places
    mean_rounded = pivot["mean"].round(3)
    stddev_rounded = pivot["stddev"].round(3)

    # Format for LaTeX
    formatted_df = (
        "$"
        + mean_rounded.applymap(lambda x: f"{x:.3f}")
        + "_{\pm "
        + stddev_rounded.applymap(lambda x: f"{x:.3f}")
        + "}$"
    )

    # Store in the dictionary
    dfs[name] = formatted_df


result_path = "results_formatted/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

for metric, table in dfs.items():
    table.to_csv(f"{result_path}/{metric}.csv")
