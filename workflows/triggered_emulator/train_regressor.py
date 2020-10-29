# coding: utf-8
import argparse

import config
import fv3fit._shared
import fv3fit.sklearn
import joblib
import sklearn.compose
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.pipeline
import trigger
import xarray as xr
from loaders.batches._sequences import Local

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("output")
parser.add_argument(
    "--classifier",
    type=str,
    default="",
    help="path to classifier to use for training. Use predictions from this",
)
parser.add_argument("--n-estimators", type=int, default=10)
parser.add_argument("--max-depth", type=int, default=10)
parser.add_argument("--only-triggered", action="store_true")

args = parser.parse_args()

input_ = args.data
output = args.output

data = Local(input_)
ds = xr.concat(data, dim="sample")
ds["active"] = trigger.is_active(ds)

x_packer = fv3fit._shared.ArrayPacker("sample", config.input_variables)
y_packer = fv3fit._shared.ArrayPacker("sample", config.output_variables)
label_packer = fv3fit._shared.ArrayPacker("sample", ["active"])

X_train, y_train, label = (
    x_packer.to_array(ds),
    y_packer.to_array(ds),
    label_packer.to_array(ds),
)

rf = sklearn.ensemble.RandomForestRegressor(
    max_depth=args.max_depth, n_estimators=args.n_estimators, n_jobs=4, verbose=10
)
model = sklearn.compose.TransformedTargetRegressor(
    regressor=rf, transformer=sklearn.preprocessing.StandardScaler()
)

if args.classifier:
    classifier = joblib.load(args.classifier)
    label = classifier.predict(X_train)
elif args.only_triggered:
    print("Training with all points")
    label = label.ravel().astype(bool)
else:
    print("Training with all points")
    label = slice(None, None)

model.fit(X_train[label], y_train[label])
joblib.dump(
    {
        "model": model,
        "input_variables": config.input_variables,
        "output_variables": config.output_variables,
        "args": args,
    },
    output,
)
