import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
import sys
import os
import csv
import numpy as np
import pandas as pd
import pickle

#### Parameters ####
num_Trees = 1000
SEED = 1


def load_data(directory, extension='.features', delimiter=' '):
    """
    Load feature files from feature directory.
    :return X - numpy array of data instances w/ shape (n,f)
    :return Y - numpy array of data labels w/ shape (n,1)
    """
    X = []  # feature instances
    Y = []  # site labels
    for root, dirs, files in os.walk(directory):

        # filter for feature files
        files = [fi for fi in files if fi.endswith(extension)]

        # read each feature file as CSV
        for file in files:
            cls, ins = file.split("-")
            with open(os.path.join(root, file), "r") as csvFile:
                # Handle np.float64(...) and np.int64(...) and convert to float
                raw_features = list(csv.reader(csvFile, delimiter=delimiter))[0]
                features = []
                for f in raw_features:
                    if 'np.float64' in f or 'np.int64' in f:
                        # Safely evaluate and extract the number
                        value = eval(f.split('(')[1].rstrip(')'))
                        features.append(float(value))
                    elif f:
                        features.append(float(f))
                #features = features[:13] + features[37:2813] + features[2939:]  # remove time features
                X.append(features)
                Y.append(int(cls))

    return np.array(X), np.array(Y)


def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i in range(truths.shape[0]):
        if truths[i] in best_n[i, :]:
            successes += 1
    return float(successes)/truths.shape[0]


def load_features(path_to_features, tr_split):
    """
    Prepare monitored data for training and test sets.
    """

    # load features dataset
    #X_tr, Y_tr = load_data(os.path.join(path_to_features, 'train'), ".features", " ")
    #X_ts, Y_ts = load_data(os.path.join(path_to_features, 'test'), ".features", " ")

    X, Y = load_data(path_to_features, ".features", " ")
    # shuffle features
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, Y = X[s], Y[s]

    # split into training and testing
    cut = int(tr_split*Y.shape[0])
    X_tr, Y_tr, X_ts, Y_ts = X[:cut], Y[:cut], X[cut:], Y[cut:]

    return X_tr, Y_tr, X_ts, Y_ts


# Feature groups mapping
feature_groups = [
    ('pkt_count', (0, 13)),
    ('time', (13, 37)),
    ('ngram', (37, 161)),
    ('transposition', (161, 765)),
    ('interval-I', (765, 1365)),
    ('interval-II', (1365, 1967)),
    ('interval-III', (1967, 2553)),
    ('pkt_distribution', (2553, 2778)),
    ('burst', (2778, 2789)),
    ('first_20', (2789, 2809)),
    ('pkt_per_second', (2813, 2939)),
    ('CUMUL', (2939, 3043))
]

def train_and_evaluate(X_tr, Y_tr, X_ts, Y_ts):
    results = []
    
    for name, (start, end) in feature_groups:
        if end > X_tr.shape[1]:  # Ensure feature indices do not exceed dataset dimensions
            print(f"Skipping {name}: Feature index out of bounds!")
            continue

        X_tr_subset = X_tr[:, start:end]
        X_ts_subset = X_ts[:, start:end]

        # Ensure subsets have features before training
        if X_tr_subset.shape[1] == 0 or X_ts_subset.shape[1] == 0:
            print(f"Skipping {name}: Feature subset has zero features!")
            continue

        # Train a separate model for each feature group
        model = RandomForestClassifier(n_jobs=2, n_estimators=1000, oob_score=True, random_state=1)
        model.fit(X_tr_subset, Y_tr)

        pred = model.predict_proba(X_ts_subset)
        top_1 = np.mean(np.argmax(pred, axis=1) == Y_ts) * 100
        top_2 = np.mean([Y_ts[i] in np.argsort(pred[i])[-2:] for i in range(len(Y_ts))]) * 100
        top_5 = np.mean([Y_ts[i] in np.argsort(pred[i])[-5:] for i in range(len(Y_ts))]) * 100

        results.append([name, f"{top_1:.1f}%", f"{top_2:.1f}%", f"{top_5:.1f}%"])

    # Save results to a pickle file
    results_df = pd.DataFrame(results, columns=["Features", "Top-1", "Top-2", "Top-5"])
    pickle_filename = "table_XII_results.pkl"
    with open(pickle_filename, "wb") as f:
        pickle.dump(results_df, f)

    print(f"Results saved in {pickle_filename}")
    return results_df
    
    

def classify(feature_directory, tr_split, out):
    """
    Closed world RF classification of data
    - only uses sk.learn classification - does not do additional k-nn.
    """

    # load dataset
    X_tr, Y_tr, X_ts, Y_ts = load_features(feature_directory, tr_split)
    print(X_tr.shape[1])
    #
    # train random forest model
    #
    print("Training ...")
    #model = RandomForestClassifier(n_jobs=20, n_estimators=num_Trees, oob_score=True)
    #model.fit(X_tr, Y_tr)
    
    table_XII = train_and_evaluate(X_tr, Y_tr, X_ts, Y_ts)
    print(table_XII.to_string(index=False))
    #
    # test performance
    #
    #acc = model.score(X_ts, Y_ts)
    #print("accuracy = ", acc)

    #pred = model.predict_proba(X_ts)
    #acc_2 = top_n_accuracy(pred, Y_ts, 2)
    #print("top_2 accuracy = ", acc_2)

    #
    # rank feature importance
    #
    #print("Top 100 features:")
    #importance = zip(model.feature_importances_, range(0, len(model.feature_importances_)))
    #sorted_importance = sorted(importance, key=lambda tup: tup[0], reverse=True)
    #index = 0
    #for score, label in sorted_importance:
    #    index += 1
    #    print("\t%d. Feature #%s (%f)" % (index, label, score))
    #    if index == 100:
    #        break

    #
    # cross validation score
    #
    #scores = cross_val_score(model, np.array(X_tr), np.array(Y_tr))
    #print("cross_val_score = ", scores.mean())
    #print("OOB score = ", model.oob_score_)

   


def parse_arguments():
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description='RF benchmarks')
    parser.add_argument('-f', '--features',
                        type=str,
                        help="Path to feature dictionary.",
                        required=True)
    parser.add_argument('-t', '--train',
                        default=0.8,
                        type=float,
                        help="Percentage of dataset to use for training.",
                        required=False)
    parser.add_argument('-o', '--output',
                        default=None,
                        help="Output file to store results.")
    return parser.parse_args()


def main():
    """
    Run RF classification using WeFDE features.
    """
    args = parse_arguments()

    # Example command line:
    # $ python rf.py --features /path/to/features --train 0.8
    classify(args.features, args.train, args.output)

    return 0


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
