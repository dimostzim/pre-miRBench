#!/usr/bin/env python3
import argparse
import csv
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier

DINUCLEOTIDES = [a + b for a in 'ACGU' for b in 'ACGU']

def dinucleotide_features(seq):
    """Compute frequencies of all 16 dinucleotides."""
    seq = seq.upper().replace('T', 'U')
    counts = {di: 0 for di in DINUCLEOTIDES}
    total = 0
    for i in range(len(seq) - 1):
        di = seq[i:i+2]
        if di in counts:
            counts[di] += 1
            total += 1
    return [counts[di] / total for di in DINUCLEOTIDES]

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--json', required=True)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--C', type=float, default=1.0, help='Regularization strength (default: 1.0, lower=more regularization)')
parser.add_argument('--solver', default='lbfgs', choices=['lbfgs', 'liblinear', 'saga'], help='Solver algorithm (default: lbfgs)')
parser.add_argument('--n_trees', type=int, default=100, help='Number of trees for Random Forest (default: 100)')
args = parser.parse_args()

# load data
rows = []
with open(args.input) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# extract features and labels
X = []
y = []
for row in rows:
    features = dinucleotide_features(row['sequence'])
    X.append(features)
    label = 1 if row['label'] == 'positive' else 0
    y.append(label)

X = np.array(X)
y = np.array(y)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

results = {}

# Baseline 1: Random classifier (should be ~0.5)
print("Training random baseline...")
dummy = DummyClassifier(strategy='uniform', random_state=args.random_state)
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
y_pred_proba = dummy.predict_proba(X_test)[:, 1]
results['random'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

# Baseline 2: Majority class (should be ~0.5 for balanced data)
print("Training majority class baseline...")
dummy_majority = DummyClassifier(strategy='most_frequent', random_state=args.random_state)
dummy_majority.fit(X_train, y_train)
y_pred = dummy_majority.predict(X_test)
y_pred_proba = dummy_majority.predict_proba(X_test)[:, 1]
results['majority_class'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

# Baseline 3: Logistic Regression on full balanced dataset
print("Training logistic regression on balanced dataset...")
lr_balanced = LogisticRegression(C=args.C, solver=args.solver, random_state=args.random_state, max_iter=1000)
lr_balanced.fit(X_train, y_train)
y_pred = lr_balanced.predict(X_test)
y_pred_proba = lr_balanced.predict_proba(X_test)[:, 1]
results['lr_balanced'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

# Baseline 4: Random Forest on full balanced dataset
print("Training random forest on balanced dataset...")
rf = RandomForestClassifier(n_estimators=args.n_trees, random_state=args.random_state, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
results['rf_balanced'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

# Save results
with open(args.json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {args.json}")
print("\nBaseline Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
