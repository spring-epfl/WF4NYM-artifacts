#!/usr/bin/env python3

import argparse
import pickle
import os
import json
import logging
import classifiers
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical

import common
import classifiers


def train_tiktok_model(X_train, y_train, n_packets=10000, epochs=100):
    """
    Train a TikTok (SirinamDF) model on X_train, y_train and return it.
    """
    logging.info("→ converting training data to TikTok repr")
    X_tt = common.convert_records_to_tiktok_repr(X_train, n_packets)
    n_classes = np.max(y_train) + 1
    y_tt = to_categorical(y_train, num_classes=n_classes)

    model = classifiers.SirinamDF(X_tt.shape[1], epochs, n_classes)
    model.metadata["input"] = "tt"
    model.fit(X_tt, y_tt)
    return model


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    return obj


def evaluate_model(model, X_test, y_test, out_path, n_packets=10000):
    """
    Evaluate `model` on X_test, y_test; dump accuracy+confusion to JSON.
    """
    logging.info("→ converting test data to TikTok repr")
    X_tt = common.convert_records_to_tiktok_repr(X_test, n_packets)
    y_pred = model.predict(X_tt)

    acc = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred).tolist()
    binary = len(set(y_test)) == 2

    results = classifiers.compute_stats(y_test, y_pred, binary=binary)
    results["conf"] = conf

    # Convert all numpy arrays to lists for JSON serialization
    results = to_serializable(results)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"✔ evaluation saved to {out_path}")


def run_cv_for_file(pkl_path, output_dir, n_folds=5, random_state=42):
    """
    1) Load (X_train, X_test, y_train, y_test) from pickle
    2) Concatenate train+test into X_all, y_all
    3) 5-fold stratified CV over X_all, y_all
    4) For each fold: train on fold’s train split, evaluate on fold’s test split
    """
    logging.info(f"➤ loading `{pkl_path}`")
    with open(pkl_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # combine
    X_all = X_train + X_test
    y_all = y_train + y_test

    basename = Path(pkl_path).stem
    skf = StratifiedKFold(n_splits=n_folds,
                          shuffle=True,
                          random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X_all, y_all), start=1
    ):
        tag = f"{basename}_fold{fold}"
        logging.info(f"\n=== Fold {fold}/{n_folds} (tag={tag}) ===")

        X_fold_train = [X_all[i] for i in train_idx]
        y_fold_train = [y_all[i] for i in train_idx]
        X_fold_test  = [X_all[i] for i in test_idx]
        y_fold_test  = [y_all[i] for i in test_idx]

        # train
        model = train_tiktok_model(X_fold_train, y_fold_train)

        # eval
        eval_path = Path(output_dir) / f"tt_{tag}_eval.json"
        evaluate_model(model, X_fold_test, y_fold_test, eval_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="5-fold stratified CV on combined X_train+X_test/y_train+y_test"
    )
    p.add_argument("pickle_files", nargs="+",
                   help="Pickle files with (X_train, X_test, y_train, y_test)")
    p.add_argument("output_dir",
                   help="Directory to save evaluation JSONs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    for pkl in args.pickle_files:
        run_cv_for_file(pkl, args.output_dir)


if __name__ == "__main__":
    main()
