# coding: utf-8
import sys

import config
import fv3fit._shared
import fv3fit.sklearn
import joblib
import loaders.batches
import loaders.mappers
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import trigger
import xarray as xr
from loaders.batches._sequences import Local


def get_test_data():
    url = "gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free"
    mapper = loaders.mappers.open_baseline_emulator(url)
    input_variables = [
        "air_temperature",
        "specific_humidity",
        "cos_zenith_angle",
        "surface_geopotential",
    ]
    output_variables = ["dQ1", "dQ2"]

    # In[4]:

    sequence = loaders.batches.batches_from_mapper(
        mapper,
        variable_names=input_variables + output_variables,
        timesteps_per_batch=5,
    )
    test = sequence[10]
    test["active"] = trigger.is_active(test)
    return test


input_ = sys.argv[1]
output = sys.argv[2]

data = Local(input_)
ds = xr.concat(data, dim="sample")

x_packer = fv3fit._shared.ArrayPacker("sample", config.input_variables)
y_packer = fv3fit._shared.ArrayPacker("sample", ["active"])

ds["active"] = trigger.is_active(ds)
X_train, y_train = x_packer.to_array(ds), y_packer.to_array(ds)

test = get_test_data()
X_test, y_test = x_packer.to_array(test), y_packer.to_array(test)


nn = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=(128, 128, 128), max_iter=100, verbose=10
    ),
)
nn.fit(X_train, y_train)

joblib.dump({"model": nn, "input_variables": config.input_variables}, output)
