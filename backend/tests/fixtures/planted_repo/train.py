"""Train a one-hidden-layer MLP on the scikit-learn digits dataset.

PLANTED DISCREPANCIES (ReproScope test fixture; see planted_truth.json):
  P1  --lr defaults to 0.01, the paper states 0.001
  P2  test_size=0.3, the paper states an 80/20 split
  P3  no random seed anywhere
  P4  f1 uses average="micro", the paper reports macro-F1
"""
import argparse
import json

import yaml
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description="MLP on digits")
    ap.add_argument("--lr", type=float, default=0.01, help="learning rate")
    ap.add_argument("--hidden", type=int, default=64, help="hidden units")
    ap.add_argument("--epochs", type=int, default=200, help="max training iterations")
    ap.add_argument("--config", default=None, help="optional YAML config overriding the defaults")
    ap.add_argument("--out", default="results.json", help="where to write metrics")
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            setattr(args, key, value)

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(args.hidden,), learning_rate_init=args.lr,
                        max_iter=args.epochs)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    results = {
        "accuracy": accuracy_score(yte, pred),
        "f1": f1_score(yte, pred, average="micro"),
    }
    print(f"Test accuracy: {results['accuracy'] * 100:.2f}")
    print(f"Test F1: {results['f1']:.4f}")
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
