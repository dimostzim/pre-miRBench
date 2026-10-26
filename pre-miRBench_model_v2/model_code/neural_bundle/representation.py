"""Iteration 11 two-input representation; iteration 4 features.py is unchanged.
Inference callers pass only input/ to load_inputs. No labels or species enter inputs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import features

CHANNEL_NAMES = ['nt_A','nt_C','nt_G','nt_U','db_dot','db_open','db_close',
                 'paired','post_character_depth_div200','signed_partner_displacement_div200',
                 'centered_pair_midpoint_div200','maximal_contiguous_stack_length_div200',
                 'pair_AU','pair_UA','pair_GC','pair_CG','pair_GU','pair_UG']
PAIR_TYPES = ('AU','UA','GC','CG','GU','UG')

def positional_one(sequence, structure):
    seq = features.normalize_sequence(sequence)
    if len(seq) != 200 or not isinstance(structure, str) or len(structure) != 200:
        raise ValueError('Sequence and structure must have exactly 200 positions')
    if set(seq) - set('ACGUN'):
        raise ValueError('Sequence contains characters outside A,C,G,U,N (after normalization)')
    partner = np.full(200, -1, dtype=np.int64)
    depth = np.zeros(200, dtype=np.float32)
    opens = []
    for i, char in enumerate(structure):
        if char == '(':
            opens.append(i)
        elif char == ')':
            if not opens:
                raise ValueError('Unmatched closing parenthesis')
            j = opens.pop()
            partner[i] = j
            partner[j] = i
        elif char != '.':
            raise ValueError('Unsupported dot-bracket symbol')
        depth[i] = len(opens)
    if opens:
        raise ValueError('Unmatched opening parenthesis')
    stacks = np.zeros(200, dtype=np.float32)
    for i, j in enumerate(partner):
        if j <= i:
            continue
        # Only start at the outermost pair in a maximal contiguous stack.
        if i > 0 and j < 199 and partner[i-1] == j+1:
            continue
        length = 1
        while i+length < j-length and partner[i+length] == j-length:
            length += 1
        for k in range(length):
            stacks[i+k] = stacks[j-k] = length
    out = np.zeros((18, 200), dtype=np.float32)
    chars = np.array(list(seq))
    brackets = np.array(list(structure))
    for c, base in enumerate('ACGU'):
        out[c] = chars == base
    for c, symbol in enumerate('.()', 4):
        out[c] = brackets == symbol
    paired = partner >= 0
    indices = np.arange(200)
    out[7] = paired
    out[8] = depth / 200.0
    out[9, paired] = (partner[paired] - indices[paired]) / 200.0
    out[10, paired] = ((indices[paired] + partner[paired]) / 2.0 - 99.5) / 200.0
    out[11] = stacks / 200.0
    for i in np.flatnonzero(paired):
        pair = seq[i] + seq[partner[i]]
        if pair in PAIR_TYPES:
            out[12 + PAIR_TYPES.index(pair), i] = 1.0
    return out

def extract_inputs(samples):
    positional = np.empty((len(samples), 18, 200), dtype=np.float32)
    for row, (seq, structure) in enumerate(zip(samples['sequence_rna'], samples['structure'])):
        positional[row] = positional_one(seq, structure)
    global_features = features.extract_features(samples)
    if global_features.shape != (len(samples), 1837):
        raise ValueError('Global feature dimensions differ from iteration 4')
    return global_features, positional

def load_inputs(input_dir):
    samples = features.load_samples(input_dir)
    global_features, positional = extract_inputs(samples)
    return samples['id'].to_numpy(), global_features, positional

def make_preprocessor():
    # Never drop constant or all-missing columns.
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scaler', StandardScaler())])

def transform_global(preprocessor, matrix):
    result = np.clip(preprocessor.transform(matrix), -8, 8).astype(np.float32)
    if result.shape != (len(matrix), 1837) or not np.isfinite(result).all():
        raise ValueError('Invalid standardized global features')
    return result

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def validate_config(directory):
    directory = Path(directory)
    config = json.loads((directory / 'feature_config.json').read_text())
    payload = dict(config)
    expected = payload.pop('schema_sha256')
    if digest(payload) != expected:
        raise ValueError('Representation schema checksum mismatch')
    if config['feature_names'] != features.FEATURE_NAMES or config['positional_channel_names'] != CHANNEL_NAMES:
        raise ValueError('Feature/channel ordering mismatch')
    if config['feature_names_sha256'] != features.FEATURE_CONFIG['feature_names_sha256']:
        raise ValueError('Global feature name checksum mismatch')
    if config['positional_channel_names_sha256'] != digest(CHANNEL_NAMES):
        raise ValueError('Channel checksum mismatch')
    for name, checksum in config['source_sha256'].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Representation source checksum mismatch: ' + name)
    features.validate_feature_config(directory / 'iteration4_feature_config.json')
    return config
