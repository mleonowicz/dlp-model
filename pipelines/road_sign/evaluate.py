"""Evaluation script for measuring mean squared error."""
import glob
import json
import logging
import pathlib
import tarfile

import mxnet as mx
import numpy as np
import pandas as pd
from sagemaker.model import Model
from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")

    logger.debug("Loading mxnet model.")

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    param_file = glob.glob("./*.params")
    epoch = int(param_file[0][-11:-7])
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        "image-classification", epoch
    )

    model_shapes_json_file = open("./model-shapes.json")
    model_shapes_dict = json.load(model_shapes_json_file)[0]
    train_batch_size = model_shapes_dict["shape"][0]
    train_data_shape = tuple(model_shapes_dict["shape"][1:])

    logger.debug("Reading test data.")

    DATASET_NAME = "ImageData"
    test_path = "/opt/ml/processing/test/imgs.rec"

    test = mx.io.ImageRecordIter(
        path_imgrec=test_path,
        data_name="data",
        label_name="softmax_label",
        batch_size=train_batch_size,
        data_shape=train_data_shape,
        rand_crop=False,
        rand_mirro=False,
    )

    logger.info("Inference on the test data.")

    mod = mx.mod.Module(symbol=sym, context=mx.cpu())

    mod.bind(
        for_training=False,
        data_shapes=test.provide_data,
        label_shapes=test.provide_label,
    )

    mod.set_params(arg_params, aux_params)

    logger.debug("Calculating accuracy on test set.")

    test_accuracy = mod.score(eval_data=test, eval_metric="acc")[0][1]

    print(f"Test Accuracy: {test_accuracy}")

    report_dict = {
        "classification_metrics": {
            "accuracy": {"value": test_accuracy},
        },
    }

    logger.info("Dumping evaluation report")

    output_dir = pathlib.Path("/opt/ml/processing/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
