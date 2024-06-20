# load data from repo
import sys
import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes, load_iris

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

X, y = load_iris(return_X_y=True, as_frame=True)
X["target"] = y

# split data into folds here?

# setup dataloader
loader = GenericDataLoader(
    X,
    target_column="target",
    sensitive_columns=[],
)

# generate data
syn_model = Plugins().get("arf")
syn_model.fit(loader)
syn_loader = syn_model.generate(count=100)


# evaluate
from synthcity.metrics.eval_performance import PerformanceEvaluatorXGB
from xgboost import XGBClassifier, XGBRegressor


# - split data into 10 folds at start
# - do performance_eval with specific model_arg sets, 3-fold CV
# - select best model_arg set
# - fit best model_arg set on entire train dataset (1 fold) and get resulting score on full test dataset
# - save this score in a list, to get the mean and std scores at the end

result = PerformanceEvaluatorXGB(
    n_folds=3,
    reduction="mean",
    task_type="classification",
    workspace="results",
    n_histogram_bins=10,
    use_cache=False,
).evaluate(X_gt=loader, X_syn=syn_loader)
print(result)
