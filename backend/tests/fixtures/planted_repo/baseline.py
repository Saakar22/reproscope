"""Logistic-regression baseline on digits. Seeded, so it is deterministic (control case)."""
import argparse

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    print(f"Test accuracy: {accuracy_score(yte, clf.predict(Xte)) * 100:.2f}")


if __name__ == "__main__":
    main()
