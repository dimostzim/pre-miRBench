#!/usr/bin/env python3
import argparse
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.dummy import DummyClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "output")

DINUCLEOTIDES = [a + b for a in 'ACGU' for b in 'ACGU']

def dinucleotide_features(seq):
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
parser.add_argument('--json', default=os.path.join(OUTPUT_DIR, 'models_output', 'baseline_metrics.json'))
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--C', type=float, default=1.0, help='Regularization strength (default: 1.0, lower=more regularization)')
parser.add_argument('--solver', default='lbfgs', choices=['lbfgs', 'liblinear', 'saga'], help='Solver algorithm (default: lbfgs)')
parser.add_argument('--n_trees', type=int, default=100, help='Number of trees for RF/GB (default: 100)')
parser.add_argument('--svm_C', type=float, default=1.0, help='SVM regularization (default: 1.0)')
parser.add_argument('--svm_kernel', default='rbf', choices=['rbf', 'linear', 'poly'], help='SVM kernel (default: rbf)')
args = parser.parse_args()

rows = []
with open(args.input) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

X_dinuc = []
y = []
for row in rows:
    X_dinuc.append(dinucleotide_features(row['sequence']))
    label = 1 if row['label'] == 'positive' else 0
    y.append(label)

X_dinuc = np.array(X_dinuc)
y = np.array(y)

X_train_dinuc, X_test_dinuc, y_train, y_test = train_test_split(
    X_dinuc, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

results = {}
plot_data = {}

print("Training random baseline...")
dummy = DummyClassifier(strategy='uniform', random_state=args.random_state)
dummy.fit(X_train_dinuc, y_train)
y_pred = dummy.predict(X_test_dinuc)
y_pred_proba = dummy.predict_proba(X_test_dinuc)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plot_data['random'] = (precision, recall)
results['random'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

print("Training logistic regression (dinucleotide)...")
lr_dinuc = LogisticRegression(C=args.C, solver=args.solver, random_state=args.random_state, max_iter=1000)
lr_dinuc.fit(X_train_dinuc, y_train)
y_pred = lr_dinuc.predict(X_test_dinuc)
y_pred_proba = lr_dinuc.predict_proba(X_test_dinuc)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plot_data['lr_dinuc'] = (precision, recall)
results['lr_dinuc'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

print("Training random forest (dinucleotide)...")
rf_dinuc = RandomForestClassifier(n_estimators=args.n_trees, random_state=args.random_state, n_jobs=-1)
rf_dinuc.fit(X_train_dinuc, y_train)
y_pred = rf_dinuc.predict(X_test_dinuc)
y_pred_proba = rf_dinuc.predict_proba(X_test_dinuc)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plot_data['rf_dinuc'] = (precision, recall)
results['rf_dinuc'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

print("Training SVM (dinucleotide)...")
svm = SVC(C=args.svm_C, kernel=args.svm_kernel, probability=True, random_state=args.random_state)
svm.fit(X_train_dinuc, y_train)
y_pred = svm.predict(X_test_dinuc)
y_pred_proba = svm.predict_proba(X_test_dinuc)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plot_data['svm_dinuc'] = (precision, recall)
results['svm_dinuc'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

print("Training gradient boosting (dinucleotide)...")
gb = GradientBoostingClassifier(n_estimators=args.n_trees, random_state=args.random_state)
gb.fit(X_train_dinuc, y_train)
y_pred = gb.predict(X_test_dinuc)
y_pred_proba = gb.predict_proba(X_test_dinuc)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plot_data['gb_dinuc'] = (precision, recall)
results['gb_dinuc'] = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'auprc': float(average_precision_score(y_test, y_pred_proba)),
    'aps': float(average_precision_score(y_test, y_pred_proba)),
    'auroc': float(roc_auc_score(y_test, y_pred_proba))
}

json_dir = os.path.dirname(args.json) or '.'
if json_dir != '.':
    os.makedirs(json_dir, exist_ok=True)

with open(args.json, 'w') as f:
    json.dump(results, f, indent=2)

plot_file = args.json.replace('.json', '.png')
plt.figure(figsize=(8, 6))
colors = {'random': 'gray', 'lr_dinuc': 'blue', 'rf_dinuc': 'green', 'svm_dinuc': 'orange', 'gb_dinuc': 'red'}
for model_name, (precision, recall) in plot_data.items():
    auprc = results[model_name]['auprc']
    plt.plot(recall, precision, label=f'{model_name} (AUPRC={auprc:.3f})', color=colors.get(model_name, 'black'), linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()

print(f"\nResults saved to {args.json}")
print(f"Plot saved to {plot_file}")
print("\nBaseline Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
