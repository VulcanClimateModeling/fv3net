# coding: utf-8
import os
import socket
import subprocess
import sys

import diagnostics
import fv3fit._shared
import fv3fit.sklearn
import joblib
import loaders.batches
import loaders.mappers
import matplotlib.pyplot as plt
import report
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import trigger

url = "gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free"


def get_test_data():
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
regressor_path = sys.argv[2]
dir_ = sys.argv[3]

os.makedirs(dir_, exist_ok=True)

classifier = joblib.load(input_)
regressor = joblib.load(regressor_path)
model = classifier["model"]

sections = {}

test = get_test_data()
x_packer = fv3fit._shared.ArrayPacker("sample", classifier["input_variables"])
label_packer = fv3fit._shared.ArrayPacker("sample", ["active"])
y_packer = fv3fit._shared.ArrayPacker("sample", regressor["output_variables"])
X_test, y_test, label = (
    x_packer.to_array(test),
    y_packer.to_array(test),
    label_packer.to_array(test),
)

figures = sections.setdefault("classifier", [])
sklearn.metrics.plot_precision_recall_curve(model, X_test, label)
filename = "precision-recall.png"
plt.savefig(os.path.join(dir_, filename))
figures.append(filename)

y_pred = regressor["model"].predict(X_test)
label_pred = classifier["model"].predict(X_test)
y_pred_classifier = y_pred * label_pred.reshape((-1, 1))

figures = sections.setdefault("means", [])
diagnostics.plot_compare_means(y_pred_classifier, y_test, label)
filename = "means.png"
plt.savefig(os.path.join(dir_, filename))
figures.append(filename)


figures = sections.setdefault("r2", [])
plt.figure()
diagnostics.plot_r2(y_pred_classifier, y_test)
plt.title("Triggered RF")
filename = "r2-with-classifier.png"
plt.savefig(os.path.join(dir_, filename))
figures.append(filename)

plt.figure()
diagnostics.plot_r2(y_pred, y_test)
plt.title("Untriggered RF")
filename = "r2-without-classifier.png"
plt.savefig(os.path.join(dir_, filename))
figures.append(filename)

html = report.create_html(
    title="Emulator Offline Report",
    metadata={
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "argv": " ".join(sys.argv),
        "git-sha": subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("UTF-8")
        .strip(),
        "regressor_path": regressor_path,
        "regressor": repr(regressor["model"]),
        "regressor_inputs": repr(regressor["input_variables"]),
        "classifier_path": input_,
        "classifier": repr(classifier["model"]),
        "classifer_inputs": repr(classifier["input_variables"]),
        "test-data": url,
    },
    sections=sections,
)

with open(os.path.join(dir_, "index.html"), "w") as f:
    f.write(html)
