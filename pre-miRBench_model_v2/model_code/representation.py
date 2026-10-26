"""Iteration 17: frozen neural inputs, inherited tree prefix, new loop block."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
from frozen_representation import FrozenRepresentations as InheritedRepresentations, _load

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ordered_id_sha(ids):
    return hashlib.sha256(json.dumps(list(ids),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

class Representations:
    def __init__(self,directory=None,require_fitted=True):
        self.root=Path(directory) if directory is not None else Path(__file__).resolve().parent
        self.base=InheritedRepresentations(self.root)
        self.loop=_load('_loop_aligned_features',self.root/'loop_features.py')
        self.feature_names=self.base.tree.FEATURE_NAMES+self.loop.APPENDED_NAMES
        assert len(self.feature_names)==16354
        config_path=self.root/'augmented_feature_config.json'
        if config_path.exists():
            config=json.loads(config_path.read_text())
            if config['feature_names']!=self.feature_names: raise ValueError('Tree feature order mismatch')
        manifest=self.root/'augmented_manifest.json'
        if manifest.exists():
            for name,expected in json.loads(manifest.read_text())['files_sha256'].items():
                if sha(self.root/name)!=expected: raise ValueError('Artifact checksum mismatch: '+name)
        fitted=self.root/'augmented_tree_imputer.joblib'
        self.tree_preprocessor=joblib.load(fitted) if fitted.exists() else None
        if require_fitted and self.tree_preprocessor is None: raise ValueError('New tree imputer is not fitted')
        if self.tree_preprocessor is not None and self.tree_preprocessor.n_features_in_!=16354: raise ValueError('Tree imputer dimension')
    def load_samples(self,input_dir): return self.base.load_samples(input_dir)
    def load_labeled_split(self,split_dir):
        frame=self.base.tree.load_labeled_split(split_dir)
        # Existing one-to-one merge is left-order-preserving; assert explicitly.
        import pandas as pd
        source=pd.read_csv(Path(split_dir)/'input'/'samples.tsv',sep='\t',usecols=['id'],dtype={'id':str},keep_default_na=False)
        if frame.id.tolist()!=source.id.tolist(): raise ValueError('Label join changed input order')
        return frame
    def extract_tree(self,samples):
        # Appended extraction validates strict alphabet/structure even before neural extraction.
        appended=self.loop.extract_appended(samples)
        return np.concatenate((self.base.tree.extract_features(samples),appended),axis=1)
    def transform(self,samples):
        if len(samples)>256: raise ValueError('Maximum batch size is 256')
        tree=self.tree_preprocessor.transform(self.extract_tree(samples)).astype(np.float32)
        raw,positional=self.base.rep.extract_inputs(samples)
        neural=self.base.rep.transform_global(self.base.neural_preprocessor,raw)
        if tree.shape!=(len(samples),16354): raise ValueError('Tree shape mismatch')
        if not all(np.isfinite(a).all() for a in (tree,neural,positional)): raise ValueError('Nonfinite features')
        return tree,neural,positional
    def batches(self,samples,batch_size=256):
        if not 1<=batch_size<=256: raise ValueError('Invalid batch size')
        for start in range(0,len(samples),batch_size):
            batch=samples.iloc[start:start+batch_size]
            yield batch.id.to_numpy(copy=True),self.transform(batch)

# Backward-compatible class name used by iteration 14 orchestration.
FrozenRepresentations=Representations
