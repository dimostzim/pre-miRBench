#!/usr/bin/env python3
import argparse
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import pickle

def sequence_features(seq):
    # nucleotide counts
    n = len(seq)
    a = seq.count('A')
    u = seq.count('U')
    g = seq.count('G')
    c = seq.count('C')
    return [a/n, u/n, g/n, c/n]

def structure_features(struct):
    # paired/unpaired counts
    n = len(struct)
    paired = struct.count('(') + struct.count(')')
    unpaired = struct.count('.')
    return [paired/n, unpaired/n]

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--n_trees', type=int, default=100)
parser.add_argument('--random_state', type=int, default=42)
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
    seq_feat = sequence_features(row['sequence'])
    struct_feat = structure_features(row['structure'])
    mfe = float(row['mfe'])
    features = seq_feat + struct_feat + [mfe]
    X.append(features)
    y.append(1 if row['label'] == 'positive' else 0)

X = np.array(X)
y = np.array(y)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

# train
clf = RandomForestClassifier(
    n_estimators=args.n_trees,
    random_state=args.random_state,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred_proba)
auprc = average_precision_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# save model
with open(args.model, 'wb') as f:
    pickle.dump(clf, f)

# save report
with open(args.report, 'w') as f:
    f.write("random forest classification\n")
    f.write("="*50 + "\n\n")
    f.write(f"dataset: {len(rows)} samples\n")
    f.write(f"positives: {sum(y)} | negatives: {len(y) - sum(y)}\n")
    f.write(f"train: {len(y_train)} | test: {len(y_test)}\n\n")

    f.write(f"model: {args.n_trees} trees\n")
    f.write(f"features: {X.shape[1]} (sequence: 4, structure: 2, mfe: 1)\n\n")

    f.write("test set performance\n")
    f.write("-"*50 + "\n")
    f.write(f"accuracy:  {acc:.4f}\n")
    f.write(f"precision: {prec:.4f}\n")
    f.write(f"recall:    {rec:.4f}\n")
    f.write(f"f1-score:  {f1:.4f}\n")
    f.write(f"auroc:     {auroc:.4f}\n")
    f.write(f"auprc:     {auprc:.4f}\n\n")

    f.write("confusion matrix\n")
    f.write("-"*50 + "\n")
    f.write(f"              predicted\n")
    f.write(f"              neg    pos\n")
    f.write(f"actual  neg   {cm[0,0]:<6} {cm[0,1]:<6}\n")
    f.write(f"        pos   {cm[1,0]:<6} {cm[1,1]:<6}\n\n")

    f.write("feature importances\n")
    f.write("-"*50 + "\n")
    feature_names = ['A%', 'U%', 'G%', 'C%', 'paired%', 'unpaired%', 'mfe']
    importances = clf.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        f.write(f"{name:<12} {imp:.4f}\n")

print(f"trained on {len(y_train)} samples, tested on {len(y_test)} samples")
print(f"accuracy: {acc:.4f} | precision: {prec:.4f} | recall: {rec:.4f} | f1: {f1:.4f}")
print(f"auroc: {auroc:.4f} | auprc: {auprc:.4f}")
print(f"model saved: {args.model}")
print(f"report saved: {args.report}")
