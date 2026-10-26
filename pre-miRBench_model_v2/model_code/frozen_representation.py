"""Independent frozen representations; never fit preprocessing here."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import threading
import joblib
import numpy as np

_IMPORT_LOCK = threading.RLock()

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class FrozenRepresentations:
    def __init__(self, directory=None):
        root = Path(directory) if directory is not None else Path(__file__).resolve().parent
        manifest = json.loads((root/'representation_manifest.json').read_text())
        for bundle, entry in manifest.items():
            for name, expected in entry['files_sha256'].items():
                if Path(name).name != name:
                    raise ValueError('Invalid artifact filename')
                if hashlib.sha256((root/bundle/name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Checksum mismatch: {bundle}/{name}')
        tree = root/'tree_bundle'; neural = root/'neural_bundle'
        self.tree = _load('_frozen_tree_features', tree/'features.py')
        self.neural = _load('_frozen_neural_features', neural/'features.py')
        # Original source uses `import features`; bind only during its execution.
        # The resulting module keeps its own reference; restore the process namespace.
        with _IMPORT_LOCK:
            sentinel = object(); previous = sys.modules.get('features', sentinel)
            try:
                sys.modules['features'] = self.neural
                self.rep = _load('_frozen_neural_representation', neural/'representation.py')
            finally:
                if previous is sentinel:
                    sys.modules.pop('features', None)
                else:
                    sys.modules['features'] = previous
        self.tree.validate_feature_config(tree/'feature_config.json')
        self.rep.validate_config(neural)
        self.tree_preprocessor = joblib.load(tree/'imputer.joblib')
        self.neural_preprocessor = joblib.load(neural/'global_preprocessor.joblib')
        if len(self.tree.FEATURE_NAMES) != 13357 or len(self.neural.FEATURE_NAMES) != 1837:
            raise ValueError('Unexpected feature dimensions')
        if self.tree_preprocessor.n_features_in_ != 13357 or self.neural_preprocessor.n_features_in_ != 1837:
            raise ValueError('Unexpected preprocessor dimensions')

    def load_samples(self, input_dir):
        return self.tree.load_samples(input_dir)

    def transform(self, samples):
        if len(samples) > 256:
            raise ValueError('Representation batches must contain at most 256 rows')
        # Positional validation rejects invalid length, alphabet, or structure.
        global_raw, positional = self.rep.extract_inputs(samples)
        tree = self.tree_preprocessor.transform(self.tree.extract_features(samples)).astype(np.float32)
        global_scaled = self.rep.transform_global(self.neural_preprocessor, global_raw)
        if tree.shape != (len(samples), 13357) or positional.shape != (len(samples),18,200):
            raise ValueError('Representation shape mismatch')
        if not all(np.isfinite(x).all() for x in (tree, global_scaled, positional)):
            raise ValueError('Nonfinite transformed features')
        return tree, global_scaled, positional

    def batches(self, samples, batch_size=256):
        if not 1 <= batch_size <= 256:
            raise ValueError('batch_size must be in [1,256]')
        for start in range(0, len(samples), batch_size):
            batch = samples.iloc[start:start+batch_size]
            yield batch['id'].to_numpy(copy=True), self.transform(batch)
