"""Shared, leakage-safe preprocessing for the 13-channel structure-aware RNA CNN (extended from iteration 1).

Use load_samples(input_dir) in inference; load_labeled_split(split_dir) is
strictly a training/evaluation helper. Neither IDs nor species are predictors.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

CHANNELS = ['A', 'C', 'G', 'U', '(', ')', '.', 'depth_after/100',
            'partner_offset/199', 'partner_A', 'partner_C', 'partner_G', 'partner_U']
LENGTH = 200
REQUIRED = ['id', 'sequence_rna', 'structure', 'mfe']


def validate_samples(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No input samples')
    ids = frame['id']
    if ids.isna().any() or (ids.astype(str).str.len() == 0).any() or ids.duplicated().any():
        raise ValueError('IDs must be nonempty, unique strings')
    sequences, structures = [], []
    for sid, seq, struct in zip(ids, frame.sequence_rna, frame.structure):
        if not isinstance(seq, str) or not isinstance(struct, str):
            raise ValueError(f'{sid}: missing sequence/structure')
        seq = seq.upper().replace('T', 'U')
        if len(seq) != LENGTH or set(seq) - set('ACGU'):
            raise ValueError(f'{sid}: invalid RNA length/alphabet')
        if len(struct) != LENGTH or set(struct) - set('().'):
            raise ValueError(f'{sid}: invalid structure length/alphabet')
        balance = 0
        for char in struct:
            balance += (char == '(') - (char == ')')
            if balance < 0:
                raise ValueError(f'{sid}: unbalanced dot-bracket structure')
        if balance:
            raise ValueError(f'{sid}: unbalanced dot-bracket structure')
        sequences.append(seq)
        structures.append(struct)
    raw = frame.mfe.replace(r'^\s*$', np.nan, regex=True)
    try:
        mfe = pd.to_numeric(raw, errors='raise').to_numpy(dtype=np.float64, na_value=np.nan)
    except (ValueError, TypeError) as exc:
        raise ValueError('MFE must be numeric or missing') from exc
    if np.isinf(mfe).any():
        raise ValueError('Infinite MFE is invalid')
    return sequences, structures, mfe


def load_samples(input_dir):
    frame = pd.read_csv(Path(input_dir) / 'samples.tsv', sep='\t', dtype={'id': str})
    validate_samples(frame)
    return frame


def load_labeled_split(split_dir):
    split_dir = Path(split_dir)
    frame = load_samples(split_dir / 'input')
    labels = pd.read_csv(split_dir / 'labels.csv', dtype={'id': str})
    if not {'id', 'numeric_label'}.issubset(labels.columns):
        raise ValueError('Labels require id and numeric_label')
    if labels.id.isna().any() or labels.id.duplicated().any():
        raise ValueError('Labels must have unique nonmissing IDs')
    if len(frame) != len(labels) or set(frame.id) != set(labels.id):
        raise ValueError('Input and label IDs/counts must match exactly')
    y = labels.set_index('id').loc[frame.id, 'numeric_label']
    if not y.isin([0, 1]).all():
        raise ValueError('Labels must be binary 0/1')
    return frame, y.to_numpy(dtype=np.float32)


def structure_partners(structure):
    """Return symmetric partner indices (-1 if unpaired) and post-character depth."""
    partners = np.full(len(structure), -1, dtype=np.int64)
    depth = np.zeros(len(structure), dtype=np.float32)
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if not stack:
                raise ValueError('Unbalanced structure')
            j = stack.pop()
            partners[i], partners[j] = j, i
        elif char != '.':
            raise ValueError('Invalid structure alphabet')
        depth[i] = len(stack)
    if stack:
        raise ValueError('Unbalanced structure')
    return partners, depth


class RNAFeatures:
    def __init__(self, stats=None):
        self.stats = stats

    def fit(self, training_frame):
        _, _, mfe = validate_samples(training_frame)
        if np.isnan(mfe).all():
            raise ValueError('No observed training MFE values')
        median = float(np.nanmedian(mfe))
        filled = np.where(np.isnan(mfe), median, mfe)
        std = float(filled.std(ddof=0))
        self.stats = dict(schema_version=3, channels=CHANNELS.copy(), length=LENGTH,
                          partner_indices='int64_symmetric_self_if_unpaired',
                          mfe_median=median, mfe_mean=float(filled.mean()),
                          mfe_std=std if std > 0 else 1.0)
        return self

    def transform(self, frame):
        if self.stats is None:
            raise RuntimeError('Fit on training data or load saved statistics first')
        sequences, structures, mfe = validate_samples(frame)
        x = np.zeros((len(frame), len(CHANNELS), LENGTH), dtype=np.float32)
        partner_indices = np.broadcast_to(np.arange(LENGTH, dtype=np.int64), (len(frame), LENGTH)).copy()
        for i, (seq, struct) in enumerate(zip(sequences, structures)):
            for c, symbol in enumerate(CHANNELS[:4]):
                x[i, c] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8) == ord(symbol)
            for c, symbol in enumerate(CHANNELS[4:7], 4):
                x[i, c] = np.frombuffer(struct.encode('ascii'), dtype=np.uint8) == ord(symbol)
            partners, depth = structure_partners(struct)
            paired = np.flatnonzero(partners >= 0)
            partner_indices[i, paired] = partners[paired]
            x[i, 7] = depth / 100.0
            x[i, 8, paired] = (partners[paired] - paired) / 199.0
            for c in range(4):
                x[i, 9 + c, paired] = x[i, c, partners[paired]]
        filled = np.where(np.isnan(mfe), self.stats['mfe_median'], mfe)
        scalar = ((filled - self.stats['mfe_mean']) / self.stats['mfe_std']).astype(np.float32).reshape(-1, 1)
        return x, scalar, partner_indices

    def save(self, path):
        if self.stats is None:
            raise RuntimeError('No fitted statistics to save')
        Path(path).write_text(json.dumps(self.stats, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        stats = json.loads(Path(path).read_text())
        if (stats.get('schema_version') != 3 or stats.get('channels') != CHANNELS
                or stats.get('length') != LENGTH
                or stats.get('partner_indices') != 'int64_symmetric_self_if_unpaired'):
            raise ValueError('Incompatible feature schema')
        if not all(np.isfinite(stats[k]) for k in ['mfe_median', 'mfe_mean', 'mfe_std']) or stats['mfe_std'] <= 0:
            raise ValueError('Invalid preprocessing statistics')
        return cls(stats)
