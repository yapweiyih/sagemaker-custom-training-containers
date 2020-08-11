"""
Reference:
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode.ipynb
https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html


Test:
Local test on train.py
python train.py --train "../../test_data/train/" --validation "../../test_data/val/" --model-dir "../../test_data/"

vscode launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${fileDirname}",
            "args": [
                "--train",
                "../../test_data/train/",
                "--validation",
                "../../test_data/val/",
                "--model-dir",
                "../../test_data/"
            ]
        }
    ]
}

"""

import os
import argparse
from xgboost import XGBClassifier
from utils import print_files_in_path
import pickle
from multiprocessing import cpu_count
from sklearn.metrics import classification_report, confusion_matrix, precision_score
import pandas as pd


def model_fn(model_dir):
    filename = os.path.join(model_dir, "model.pth")
    with open(filename, "rb") as f:
        model = pickle.load(f)

    print(model)
    return model


def save_model(model, model_dir):
    """Save xgb's Booster. Model function should return a xgboost.Booster object."""
    print("save booster")
    filename = os.path.join(model_dir, "model.pth")
    with open(filename, "wb") as f:
        pickle.dump(model._Booster, f)


def get_data(train_channel, validation_channel):
    """Retrieve data based on channel dir provided."""
    train_df = pd.read_csv(os.path.join(train_channel, "train.csv"), header=0, index_col=None)
    test_df = pd.read_csv(os.path.join(validation_channel, "test.csv"), header=0, index_col=None)
    X_train, y_train = train_df.iloc[:, 1:], train_df.iloc[:, 0]
    X_test, y_test = test_df.iloc[:, 1:], test_df.iloc[:, 0]
    print(f"X_train: {X_train.shape}, y_train:{y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test:{y_test.shape}")
    return X_train, X_test, y_train, y_test


def train(train_channel, validation_channel, model_dir):
    """
    SM_CHANNEL does not contain backward slash:
        SM_CHANNEL_TRAIN=/opt/ml/input/data/train
        SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation

    Training job name:
        script-mode-container-xgb-2020-08-10-13-29-15-756

    """
    print("\nList of files in train channel: ")
    print_files_in_path(train_channel)

    print("\nList of files in validation channel: ")
    print_files_in_path(validation_channel)

    X_train, X_test, y_train, y_test = get_data(train_channel, validation_channel)

    n_jobs = cpu_count() - 1

    parameters = {
        "min_child_weight": 5,
        "max_depth": 5,
        "learning_rate": 0.0001,
        "objective": "multi:softprob",
        "n_estimators": 100,
    }

    model = XGBClassifier(
        base_score=0.5,
        booster="gbtree",
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=1,
        gamma=0,
        max_delta_step=0,
        missing=None,
        n_jobs=n_jobs,  # From version 1.1.1, cant use -1 for all cores
        nthread=None,
        random_state=0,
        reg_alpha=0,
        reg_lambda=1,
        # scale_pos_weight=1,
        subsample=1,
        verbosity=1,
        **parameters,
    )
    print(model)
    fit_params = {
        # "sample_weight": df_train_w["sample_weight"],
        "early_stopping_rounds": 10,
        "eval_metric": "mlogloss",
        "eval_set": [(X_train, y_train), (X_test, y_test)],
    }
    model.fit(X_train, y_train, **fit_params)

    # Evaluation
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds, labels=[0, 1, 2]))
    print(precision_score(y_test, preds, average="weighted"))

    save_model(model, model_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument("--hp1", type=str)
    parser.add_argument("--hp2", type=int, default=50)
    parser.add_argument("--hp3", type=float, default=0.1)

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", None))

    args = parser.parse_args()
    print(args)
    train(args.train, args.validation, args.model_dir)
